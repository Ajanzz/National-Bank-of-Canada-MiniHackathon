"""Fast streaming CSV analysis optimized for large well-formed datasets."""

from __future__ import annotations

import base64
import csv
import heapq
import io
import json
import random
from collections import deque
from datetime import datetime
from time import perf_counter
from typing import Any, TextIO

BIAS_TYPES = ("REVENGE_TRADING", "LOSS_AVERSION", "RECENCY_BIAS", "OVERTRADING")
QTY_SAMPLE_SIZE = 20_000
FLAGGED_TOP_K = 500
FLAGGED_RETURN_K = 200
OVERTRADING_WINDOW_SECONDS = 600
OVERTRADING_BURST_THRESHOLD = 7
RECENCY_STREAK_MIN = 3
REVENGE_REENTRY_SECONDS = 600


def _clamp(value: float, lower: float, upper: float) -> float:
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


def _fast_parse_epoch_seconds(timestamp: str) -> int:
    # Known input: "YYYY-mm-dd HH:MM:SS" or "YYYY-mm-ddTHH:MM:SS" (optionally with Z)
    if timestamp.endswith("Z"):
        timestamp = f"{timestamp[:-1]}+00:00"
    return int(datetime.fromisoformat(timestamp).timestamp())


def _reservoir_add(sample: list[float], seen_count: int, value: float, sample_size: int) -> int:
    next_seen = seen_count + 1
    if len(sample) < sample_size:
        sample.append(value)
        return next_seen
    replace_at = random.randrange(next_seen)
    if replace_at < sample_size:
        sample[replace_at] = value
    return next_seen


def _percentile_from_sorted(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    idx = int((len(sorted_values) - 1) * pct)
    return float(sorted_values[idx])


def _to_float(raw: str | None) -> float:
    # Fast path with blank-cell tolerance for known numeric columns.
    if not raw:
        return 0.0
    raw = raw.strip()
    if not raw:
        return 0.0
    return float(raw)


def _push_flagged(
    heap: list[tuple[float, int, dict[str, Any]]],
    severity: float,
    ordinal: int,
    item: dict[str, Any],
) -> None:
    entry = (severity, ordinal, item)
    if len(heap) < FLAGGED_TOP_K:
        heapq.heappush(heap, entry)
        return
    if severity > heap[0][0]:
        heapq.heapreplace(heap, entry)


def analyze_csv_stream(text_stream: TextIO) -> dict[str, Any]:
    """Analyze CSV data in a single streaming pass and return fast contract payload."""
    timings_ms = {
        "parse_csv": 0.0,
        "aggregates": 0.0,
        "bias_features": 0.0,
        "bias_scoring": 0.0,
        "build_series": 0.0,
        "json_serialize": 0.0,
    }

    row_count = 0

    daily_pnl: dict[str, float] = {}
    monthly_pnl: dict[str, float] = {}
    hourly_pnl: dict[str, float] = {}
    hourly_trade_count: dict[str, int] = {}
    hourly_win_count: dict[str, int] = {}
    hourly_loss_count: dict[str, int] = {}
    hourly_flat_count: dict[str, int] = {}

    overtrade_deque_10m: deque[int] = deque()

    flagged_heap: list[tuple[float, int, dict[str, Any]]] = []
    flagged_ordinal = 0

    qty_sample: list[float] = []
    qty_seen = 0

    qty_running_mean = 0.0
    qty_running_count = 0

    total_pnl = 0.0
    win_count = 0
    loss_count = 0
    win_sum = 0.0
    loss_abs_sum = 0.0
    tail_loss_hits = 0

    last_loss_ts: int | None = None
    last_trade_ts: int | None = None
    prev_pnl = 0.0
    loss_streak = 0
    max_loss_streak = 0

    streak_sign = 0
    streak_len = 0
    max_abs_streak = 0

    recent_signs: deque[int] = deque(maxlen=20)

    max_burst_10m = 0
    total_bursts_over_threshold = 0
    burst_sum = 0

    overtrading_hits = 0
    revenge_hits = 0
    revenge_severity_sum = 0.0
    recency_hits = 0
    recency_severity_sum = 0.0
    loss_aversion_hits = 0
    loss_aversion_severity_sum = 0.0

    revenge_hit_quantities: list[float] = []
    recency_hit_quantities: list[float] = []

    danger_hours = [[0 for _ in range(24)] for _ in range(7)]

    reader = csv.DictReader(text_stream)
    for row in reader:
        parse_start = perf_counter()

        ts_raw = row["timestamp"]
        if not ts_raw:
            continue
        ts_raw = ts_raw.strip()
        if not ts_raw:
            continue
        asset = row["asset"]
        side = row["side"]
        quantity = _to_float(row["quantity"])
        entry_price = _to_float(row["entry_price"])
        exit_price = _to_float(row["exit_price"])
        pnl = _to_float(row["profit_loss"])
        balance = _to_float(row["balance"])

        ts_epoch = _fast_parse_epoch_seconds(ts_raw)
        day_key = ts_raw[:10]
        month_key = ts_raw[:7]
        hour_part = ts_raw[11:13] if len(ts_raw) >= 13 else "00"
        hour_key = f"{day_key} {hour_part}"

        timings_ms["parse_csv"] += (perf_counter() - parse_start) * 1000.0

        aggregate_start = perf_counter()

        row_count += 1
        total_pnl += pnl
        daily_pnl[day_key] = daily_pnl.get(day_key, 0.0) + pnl
        monthly_pnl[month_key] = monthly_pnl.get(month_key, 0.0) + pnl
        hourly_pnl[hour_key] = hourly_pnl.get(hour_key, 0.0) + pnl
        hourly_trade_count[hour_key] = hourly_trade_count.get(hour_key, 0) + 1

        if pnl > 0:
            win_count += 1
            win_sum += pnl
            hourly_win_count[hour_key] = hourly_win_count.get(hour_key, 0) + 1
        elif pnl < 0:
            loss_count += 1
            loss_abs_sum += abs(pnl)
            hourly_loss_count[hour_key] = hourly_loss_count.get(hour_key, 0) + 1
        else:
            hourly_flat_count[hour_key] = hourly_flat_count.get(hour_key, 0) + 1

        qty_seen = _reservoir_add(qty_sample, qty_seen, quantity, QTY_SAMPLE_SIZE)

        qty_running_count += 1
        qty_running_mean += (quantity - qty_running_mean) / qty_running_count

        timings_ms["aggregates"] += (perf_counter() - aggregate_start) * 1000.0

        feature_start = perf_counter()

        # OVERTRADING (rolling 10-minute window)
        overtrade_deque_10m.append(ts_epoch)
        cutoff = ts_epoch - OVERTRADING_WINDOW_SECONDS
        while overtrade_deque_10m and overtrade_deque_10m[0] < cutoff:
            overtrade_deque_10m.popleft()
        window_count = len(overtrade_deque_10m)
        burst_sum += window_count
        if window_count > max_burst_10m:
            max_burst_10m = window_count
        if window_count >= OVERTRADING_BURST_THRESHOLD:
            total_bursts_over_threshold += 1

        baseline_qty = qty_running_mean if qty_running_mean > 0 else quantity
        size_ratio = quantity / baseline_qty if baseline_qty > 0 else 1.0

        bias_flags: list[str] = []
        explanation_bullets: list[str] = []
        severity_score = 0.0

        if window_count >= OVERTRADING_BURST_THRESHOLD:
            overtrading_hits += 1
            over_sev = _clamp(35.0 + (window_count - OVERTRADING_BURST_THRESHOLD + 1) * 8.0, 0.0, 100.0)
            severity_score += over_sev
            bias_flags.append("OVERTRADING")
            explanation_bullets.append(f"{window_count} trades detected in a rolling 10-minute window.")

        # REVENGE TRADING (fast re-entry after loss + size up)
        if prev_pnl < 0.0 and last_loss_ts is not None:
            seconds_since_loss = ts_epoch - last_loss_ts
            if seconds_since_loss <= REVENGE_REENTRY_SECONDS and size_ratio >= 1.25:
                revenge_hits += 1
                revenge_hit_quantities.append(quantity)
                revenge_sev = _clamp(
                    40.0 + (max(0.0, 1.0 - (seconds_since_loss / REVENGE_REENTRY_SECONDS)) * 25.0) + (size_ratio - 1.0) * 25.0 + min(loss_streak, 5) * 6.0,
                    0.0,
                    100.0,
                )
                revenge_severity_sum += revenge_sev
                severity_score += revenge_sev
                bias_flags.append("REVENGE_TRADING")
                explanation_bullets.append("Large re-entry occurred quickly after a losing trade.")

        # RECENCY BIAS (size-up after streak context)
        prior_streak_len = streak_len
        if abs(prior_streak_len) >= RECENCY_STREAK_MIN and size_ratio >= 1.20:
            recency_hits += 1
            recency_hit_quantities.append(quantity)
            momentum = abs(sum(recent_signs)) / len(recent_signs) if recent_signs else 0.0
            recency_sev = _clamp(35.0 + abs(prior_streak_len) * 7.0 + momentum * 20.0 + (size_ratio - 1.0) * 20.0, 0.0, 100.0)
            recency_severity_sum += recency_sev
            severity_score += recency_sev
            bias_flags.append("RECENCY_BIAS")
            explanation_bullets.append("Position size increased after a strong recent streak pattern.")

        # LOSS AVERSION proxy (tail losses on running baseline)
        if pnl < 0.0 and loss_count > 5:
            running_avg_loss = loss_abs_sum / loss_count if loss_count else 0.0
            if running_avg_loss > 0 and abs(pnl) > running_avg_loss * 1.8:
                tail_loss_hits += 1
                loss_aversion_hits += 1
                loss_sev = _clamp(32.0 + (abs(pnl) / running_avg_loss) * 12.0, 0.0, 100.0)
                loss_aversion_severity_sum += loss_sev
                severity_score += loss_sev
                bias_flags.append("LOSS_AVERSION")
                explanation_bullets.append("Loss magnitude is significantly larger than the running loss baseline.")

        if bias_flags:
            flagged_ordinal += 1
            final_severity = float(_clamp(severity_score / max(1, len(bias_flags)), 0.0, 100.0))
            unique_bias_flags = sorted(set(bias_flags))
            summary = {
                "tradeId": str(row_count),
                "timestamp": ts_raw,
                "asset": asset,
                "side": side,
                "quantity": quantity,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "balance": balance,
                "biasFlags": unique_bias_flags,
                "severity": round(final_severity, 3),
                "explanationBullets": explanation_bullets,
            }
            _push_flagged(flagged_heap, final_severity, flagged_ordinal, summary)

            dt = datetime.fromtimestamp(ts_epoch)
            weekday = (dt.weekday()) % 7
            hour = dt.hour
            danger_hours[weekday][hour] += 1

        # Update streak trackers
        sign = 1 if pnl > 0 else (-1 if pnl < 0 else 0)
        if sign == 0:
            streak_len = 0
            streak_sign = 0
        elif sign == streak_sign:
            streak_len += 1
        else:
            streak_sign = sign
            streak_len = 1
        if abs(streak_len) > max_abs_streak:
            max_abs_streak = abs(streak_len)
        recent_signs.append(sign)

        if pnl < 0:
            last_loss_ts = ts_epoch
            loss_streak += 1
            if loss_streak > max_loss_streak:
                max_loss_streak = loss_streak
        elif pnl > 0:
            loss_streak = 0

        prev_pnl = pnl
        last_trade_ts = ts_epoch

        timings_ms["bias_features"] += (perf_counter() - feature_start) * 1000.0

    bias_scoring_start = perf_counter()

    qty_sorted = sorted(qty_sample) if qty_sample else [0.0]
    q50 = _percentile_from_sorted(qty_sorted, 0.50)
    q90 = _percentile_from_sorted(qty_sorted, 0.90)

    avg_burst_count = (burst_sum / row_count) if row_count else 0.0
    win_rate = (win_count / row_count) if row_count else 0.0
    avg_win = (win_sum / win_count) if win_count else 0.0
    avg_loss = (loss_abs_sum / loss_count) if loss_count else 0.0
    avg_revenge_severity = (revenge_severity_sum / revenge_hits) if revenge_hits else 0.0
    avg_recency_severity = (recency_severity_sum / recency_hits) if recency_hits else 0.0

    revenge_large_hits = sum(1 for q in revenge_hit_quantities if q >= q90) if revenge_hit_quantities else 0
    recency_large_hits = sum(1 for q in recency_hit_quantities if q >= q90) if recency_hit_quantities else 0

    overtrading_score = _clamp(
        max_burst_10m * 4.5 + (total_bursts_over_threshold / max(1, row_count)) * 700.0 + avg_burst_count * 2.0,
        0.0,
        100.0,
    )
    revenge_score = _clamp(
        (revenge_hits / max(1, row_count)) * 1000.0
        + avg_revenge_severity * 0.45
        + (revenge_large_hits / max(1, revenge_hits)) * 22.0
        + max_loss_streak * 2.0,
        0.0,
        100.0,
    )

    loss_ratio = (avg_loss / avg_win) if avg_win > 0 else 0.0
    tail_loss_ratio = tail_loss_hits / max(1, loss_count)
    high_win_low_net = 1.0 if (win_rate > 0.55 and total_pnl < (win_sum * 0.20)) else 0.0
    loss_aversion_score = _clamp(
        max(0.0, (loss_ratio - 1.0) * 38.0) + tail_loss_ratio * 130.0 + high_win_low_net * 20.0,
        0.0,
        100.0,
    )

    recency_score = _clamp(
        (recency_hits / max(1, row_count)) * 1000.0
        + avg_recency_severity * 0.40
        + (recency_large_hits / max(1, recency_hits)) * 20.0
        + max_abs_streak * 2.5,
        0.0,
        100.0,
    )

    bias_scores = {
        "REVENGE_TRADING": round(revenge_score, 2),
        "LOSS_AVERSION": round(loss_aversion_score, 2),
        "RECENCY_BIAS": round(recency_score, 2),
        "OVERTRADING": round(overtrading_score, 2),
    }

    top_biases = [name for name, _ in sorted(bias_scores.items(), key=lambda item: item[1], reverse=True)[:2]]

    timings_ms["bias_scoring"] = (perf_counter() - bias_scoring_start) * 1000.0

    series_start = perf_counter()

    daily_series = []
    cumulative_daily = 0.0
    for day in sorted(daily_pnl.keys()):
        pnl_value = daily_pnl[day]
        cumulative_daily += pnl_value
        daily_series.append(
            {
                "date": day,
                "pnl": round(pnl_value, 6),
                "cumulative": round(cumulative_daily, 6),
            }
        )

    monthly_series = []
    cumulative_monthly = 0.0
    for month in sorted(monthly_pnl.keys()):
        pnl_value = monthly_pnl[month]
        cumulative_monthly += pnl_value
        monthly_series.append(
            {
                "month": month,
                "pnl": round(pnl_value, 6),
                "cumulative": round(cumulative_monthly, 6),
            }
        )

    hourly_series = []
    cumulative_hourly = 0.0
    for hour_key in sorted(hourly_pnl.keys()):
        pnl_value = hourly_pnl[hour_key]
        cumulative_hourly += pnl_value
        date_part = hour_key[:10]
        hour_part = hour_key[11:13]
        hourly_series.append(
            {
                "timestamp": f"{date_part}T{hour_part}:00:00",
                "pnl": round(pnl_value, 6),
                "cumulative": round(cumulative_hourly, 6),
                "trade_count": int(hourly_trade_count.get(hour_key, 0)),
                "wins": int(hourly_win_count.get(hour_key, 0)),
                "losses": int(hourly_loss_count.get(hour_key, 0)),
                "flat": int(hourly_flat_count.get(hour_key, 0)),
            }
        )

    flagged_trades_sorted = [
        item[2]
        for item in sorted(flagged_heap, key=lambda entry: entry[0], reverse=True)[:FLAGGED_RETURN_K]
    ]

    timings_ms["build_series"] = (perf_counter() - series_start) * 1000.0

    result = {
        "rowCount": row_count,
        "biasScores": bias_scores,
        "topBiases": top_biases,
        "flaggedTrades": flagged_trades_sorted,
        "aggregates": {
            "dailyPnl": daily_series,
            "monthlyPnl": monthly_series,
            "hourlyPnl": hourly_series,
        },
        "dangerHours": danger_hours,
        "stats": {
            "winCount": int(win_count),
            "lossCount": int(loss_count),
            "flatCount": int(max(0, row_count - win_count - loss_count)),
            "totalPnl": round(float(total_pnl), 6),
        },
        "sizeThresholds": {"q50": round(q50, 6), "q90": round(q90, 6)},
        "perf": {
            "rows": row_count,
            "timings_ms": {k: round(v, 3) for k, v in timings_ms.items()},
        },
    }

    serialize_start = perf_counter()
    json.dumps(result, separators=(",", ":"))
    timings_ms["json_serialize"] = (perf_counter() - serialize_start) * 1000.0
    result["perf"]["timings_ms"]["json_serialize"] = round(timings_ms["json_serialize"], 3)

    return result


def analyze_csv_text(csv_text: str) -> dict[str, Any]:
    return analyze_csv_stream(io.StringIO(csv_text))


def analyze_base64_csv(base64_csv: str) -> dict[str, Any]:
    csv_bytes = base64.b64decode(base64_csv)
    csv_text = csv_bytes.decode("utf-8-sig")
    return analyze_csv_text(csv_text)


def to_frontend_compat_payload(fast_result: dict[str, Any], scoring_mode: str = "hybrid") -> dict[str, Any]:
    """Build backward-compatible payload expected by existing frontend views."""
    bias_scores_upper = fast_result["biasScores"]
    bias_scores_lower = {
        "overtrading": bias_scores_upper["OVERTRADING"],
        "loss_aversion": bias_scores_upper["LOSS_AVERSION"],
        "revenge_trading": bias_scores_upper["REVENGE_TRADING"],
        "recency_bias": bias_scores_upper["RECENCY_BIAS"],
    }

    behavior_index = round(sum(bias_scores_lower.values()) / 4.0, 2)
    daily_series = fast_result["aggregates"]["dailyPnl"]
    monthly_series = fast_result["aggregates"]["monthlyPnl"]
    hourly_series = fast_result["aggregates"].get("hourlyPnl", [])

    flagged_trades_compat: list[dict[str, Any]] = []
    for ft in fast_result["flaggedTrades"]:
        bias_flags = ft.get("biasFlags", [])
        mapped_biases = [
            "overtrading" if b == "OVERTRADING"
            else "loss_aversion" if b == "LOSS_AVERSION"
            else "revenge_trading" if b == "REVENGE_TRADING"
            else "recency_bias"
            for b in bias_flags
        ]
        flagged_trades_compat.append(
            {
                "trade_id": ft.get("tradeId"),
                "timestamp": ft.get("timestamp"),
                "symbol": ft.get("asset"),
                "side": ft.get("side"),
                "quantity": ft.get("quantity"),
                "entry_price": ft.get("entry_price"),
                "exit_price": ft.get("exit_price"),
                "profit_loss": ft.get("pnl"),
                "balance": ft.get("balance"),
                "bias": mapped_biases[0] if mapped_biases else "overtrading",
                "bias_types": mapped_biases,
                "confidence": round(float(ft.get("severity", 0.0))),
                "flag_severity": "high" if float(ft.get("severity", 0.0)) >= 70 else "medium" if float(ft.get("severity", 0.0)) >= 40 else "low",
                "evidence": ft.get("explanationBullets", []),
            }
        )

    stats = fast_result.get("stats", {})
    total_pnl = float(stats.get("totalPnl", sum(point["pnl"] for point in daily_series)))
    winning_trades = int(stats.get("winCount", 0))
    losing_trades = int(stats.get("lossCount", 0))
    top_trigger_map = {
        "OVERTRADING": "Overtrading",
        "LOSS_AVERSION": "Loss Aversion",
        "REVENGE_TRADING": "Revenge Trading",
        "RECENCY_BIAS": "Recency Bias",
    }
    top_triggers = [top_trigger_map.get(item, item) for item in list(fast_result["topBiases"])]
    if hourly_series:
        trade_timeline = [
            {
                "timestamp": row["timestamp"],
                "pnl": row["pnl"],
                "trade_count": int(row.get("trade_count", 0)),
                "wins": int(row.get("wins", 0)),
                "losses": int(row.get("losses", 0)),
                "flat": int(row.get("flat", 0)),
            }
            for row in hourly_series
        ]
    else:
        trade_timeline = [{"timestamp": f'{row["date"]}T12:00:00', "pnl": row["pnl"], "trade_count": 1} for row in daily_series]

    compat_payload = {
        "behavior_index": behavior_index,
        "bias_scores": bias_scores_lower,
        "top_triggers": top_triggers,
        "danger_hours": fast_result.get("dangerHours", [[0] * 24 for _ in range(7)]),
        "daily_pnl": [{"date": row["date"], "pnl": row["pnl"]} for row in daily_series],
        "trade_timeline": trade_timeline,
        "flagged_trades": flagged_trades_compat,
        "explainability": {
            "overtrading": [],
            "loss_aversion": [],
            "revenge_trading": [],
            "recency_bias": [],
        },
        "summary": {
            "total_trades": fast_result["rowCount"],
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": round((winning_trades / max(1, fast_result["rowCount"])) * 100.0, 2),
            "total_pnl": round(float(total_pnl), 6),
            "avg_pnl_per_trade": round(float(total_pnl) / max(1, fast_result["rowCount"]), 6),
            "starting_balance": 0.0,
            "ending_balance": round(float(total_pnl), 6),
            "max_drawdown": 0.0,
            "volatility": 0.0,
            "date_range": (
                f'{daily_series[0]["date"]} to {daily_series[-1]["date"]}'
                if daily_series else "N/A"
            ),
        },
        "monthly_pnl": monthly_series,
        "ml_active": False,
        "ml_probabilities": {},
        "scoring_mode": scoring_mode,
        "perf": fast_result.get("perf", {"rows": fast_result["rowCount"], "timings_ms": {}}),
    }
    return compat_payload


def merge_fast_and_compat(fast_result: dict[str, Any], compat_payload: dict[str, Any]) -> dict[str, Any]:
    """Return output contract plus compatibility fields for existing clients."""
    merged = dict(fast_result)
    merged.update(compat_payload)
    return merged
