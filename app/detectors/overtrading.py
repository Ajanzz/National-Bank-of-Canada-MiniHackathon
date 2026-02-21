"""Overtrading bias detector – aligned with strategy discipline definition."""

from collections import defaultdict

from app.models.schemas import DetectionResult, NormalizedTrade
from app.utils.time_utils import minutes_between
from app.utils.stats_utils import mean, stdev


def detect_overtrading(trades: list[NormalizedTrade]) -> DetectionResult:
    """
    Detect overtrading bias using BEHAVIORAL RATES not raw counts, so results
    are consistent whether the dataset has 200 or 200,000 trades.

    Signals (all scored as rates 0-1, then weighted into final score):
    1. Position over-extension: size_usd > 20% of account balance (absolute threshold)
    2. Rapid direction switching: flip rate among consecutive trade pairs (BUY↔SELL < 30 min)
    3. Reactive trading after outlier outcomes: re-entry within 15 min of a 2-sigma PnL event
    4. Hourly burst density: hours where trade density is >= 3x the trader's own average rate
    """
    if not trades:
        return DetectionResult(score=0, severity="low", flagged_trade_ids=[], evidence_by_trade_id={}, stats={})

    flagged: set[str] = set()
    evidence: dict[str, str] = {}
    total = len(trades)

    def add_evidence(tid: str, msg: str) -> None:
        existing = evidence.get(tid, "")
        evidence[tid] = f"{existing}+{msg}" if existing else msg
        flagged.add(tid)

    # ── Signal 1: Position over-extension (absolute threshold) ────────────────
    # Trading > 20% of account balance on a single position is objectively risky
    # regardless of how many total trades are in the dataset.
    OVERSIZE_RATIO = 0.20
    oversize_count = 0
    for t in trades:
        if t.balance > 0:
            ratio = t.size_usd / t.balance
            if ratio > OVERSIZE_RATIO:
                oversize_count += 1
                add_evidence(t.trade_id, f"oversize_{round(ratio * 100, 0):.0f}pct_of_balance")
    oversize_rate = oversize_count / total if total > 0 else 0.0

    # ── Signal 2: Rapid position switching rate ────────────────────────────────
    # A high RATE of BUY↔SELL flips within a short window signals lack of conviction.
    # Scored as: flips / total transitions — independent of dataset size.
    FLIP_WINDOW_MIN = 30
    flip_count = 0
    transitions = 0
    for i in range(1, total):
        prev, curr = trades[i - 1], trades[i]
        transitions += 1
        if prev.side != curr.side:
            gap = minutes_between(prev.timestamp, curr.timestamp)
            if gap <= FLIP_WINDOW_MIN:
                flip_count += 1
                add_evidence(curr.trade_id, f"dir_flip_{int(gap)}m")
    flip_rate = flip_count / transitions if transitions > 0 else 0.0

    # ── Signal 3: Reactive trading after outlier outcomes ─────────────────────
    # Uses mean + 2*stdev (true statistical outliers) as the "large PnL" threshold
    # so it doesn't always fire on 25% of trades like a p75 would.
    abs_pnls = [t.abs_pnl for t in trades]
    avg_pnl = mean(abs_pnls)
    sd_pnl = stdev(abs_pnls)
    # Only flag if the outlier threshold itself is distinguishable (sd > 0)
    outlier_threshold = avg_pnl + 2 * sd_pnl if sd_pnl > 0 else float("inf")
    REACTION_WINDOW_MIN = 15
    reaction_count = 0
    for i in range(1, total):
        prev, curr = trades[i - 1], trades[i]
        if prev.abs_pnl >= outlier_threshold:
            gap = minutes_between(prev.timestamp, curr.timestamp)
            if gap <= REACTION_WINDOW_MIN:
                reaction_count += 1
                label = "react_after_big_win" if prev.is_win else "react_after_big_loss"
                add_evidence(curr.trade_id, f"{label}_{int(gap)}m")
    reaction_rate = reaction_count / total if total > 0 else 0.0

    # ── Signal 4: Hourly burst density (rate-based) ───────────────────────────
    # Compares each hour's trade count to the trader's own mean hourly rate.
    # Uses a 3x multiplier so a normally busy trader isn't penalised.
    # Flags the PROPORTION of hours that are bursts, not the raw burst count.
    by_hour: dict[tuple, list] = defaultdict(list)
    for t in trades:
        by_hour[(t.date, t.hour)].append(t)

    num_hours = len(by_hour)
    avg_hourly = total / num_hours if num_hours > 0 else total
    BURST_MULTIPLIER = 3.0
    BURST_ABS_MIN = 10  # must be at least 10 trades/hour to call it a burst
    burst_hours = 0
    for (date, hour), group in by_hour.items():
        count = len(group)
        if count >= BURST_MULTIPLIER * avg_hourly and count >= BURST_ABS_MIN:
            burst_hours += 1
            for t in group:
                add_evidence(t.trade_id, f"burst_hour({count}_trades)")
    burst_hour_rate = burst_hours / num_hours if num_hours > 0 else 0.0

    # ── Score (all rate-based, not count-based) ────────────────────────────────
    # Each component contributes at most its weight; rates are capped at 1.0.
    score = 0
    score += min(30, int(oversize_rate * 60))       # over-extension: up to 30 pts
    score += min(30, int(flip_rate * 60))            # direction-flip rate: up to 30 pts
    score += min(20, int(reaction_rate * 200))       # reaction rate: up to 20 pts (rarer events)
    score += min(20, int(burst_hour_rate * 80))      # burst hour proportion: up to 20 pts

    final_score = min(100, score)
    severity = "low" if final_score < 30 else "med" if final_score < 70 else "high"

    stats = {
        "total_trades": total,
        "flagged_count": len(flagged),
        "oversize_rate": round(oversize_rate, 3),
        "flip_rate": round(flip_rate, 3),
        "reaction_rate": round(reaction_rate, 3),
        "burst_hour_rate": round(burst_hour_rate, 3),
        "position_flips": flip_count,
        "burst_hours": burst_hours,
        "total_hours_active": num_hours,
        "avg_trades_per_hour": round(avg_hourly, 1),
        "outlier_pnl_threshold": round(outlier_threshold, 2),
        "flip_window_min": FLIP_WINDOW_MIN,
        "reaction_window_min": REACTION_WINDOW_MIN,
    }

    return DetectionResult(
        score=final_score,
        severity=severity,
        flagged_trade_ids=list(flagged),
        evidence_by_trade_id=evidence,
        stats=stats,
    )
    if not trades:
        return DetectionResult(score=0, severity="low", flagged_trade_ids=[], evidence_by_trade_id={}, stats={})

    flagged: set[str] = set()
    evidence: dict[str, str] = {}
    total = len(trades)

    def add_evidence(tid: str, msg: str) -> None:
        existing = evidence.get(tid, "")
        evidence[tid] = f"{existing}+{msg}" if existing else msg
        flagged.add(tid)

    # ── Signal 1: Median inter-trade gap (pace score) ─────────────────────────
    # The clearest overtrading signal: how fast is the trader actually firing?
    # Calm, disciplined traders wait minutes between trades. Overtrades do seconds.
    # >= 5 min median → 0 pts.  <= 30 sec median → 35 pts.  Linear between.
    gaps_sec: list[float] = []
    for i in range(1, total):
        g = minutes_between(trades[i - 1].timestamp, trades[i].timestamp) * 60.0
        if g >= 0:
            gaps_sec.append(g)

    median_gap_sec = 0.0
    if gaps_sec:
        s = sorted(gaps_sec)
        mid = len(s) // 2
        median_gap_sec = s[mid] if len(s) % 2 == 1 else (s[mid - 1] + s[mid]) / 2.0

    CALM_GAP_SEC = 300.0   # 5 minutes → score = 0
    FAST_GAP_SEC = 30.0    # 30 seconds → score = 35
    if median_gap_sec >= CALM_GAP_SEC:
        gap_score = 0
    elif median_gap_sec <= FAST_GAP_SEC:
        gap_score = 35
    else:
        gap_score = int((CALM_GAP_SEC - median_gap_sec) / (CALM_GAP_SEC - FAST_GAP_SEC) * 35)

    # Flag individual fast trades for the evidence map
    FAST_FLAG_SEC = 60.0
    for i in range(1, total):
        if gaps_sec and gaps_sec[i - 1] <= FAST_FLAG_SEC:
            add_evidence(trades[i].trade_id, f"fast_{int(gaps_sec[i - 1])}s_gap")

    # ── Signal 2: Same-asset rapid direction flip ──────────────────────────────
    # Flipping BUY→SELL or SELL→BUY on the SAME asset within 5 min is genuine
    # indecision/impulsivity. Normal BUY/SELL alternation across different assets
    # is not flagged here, which is why we require same-asset.
    SAME_ASSET_FLIP_MIN = 5.0
    same_asset_flips = 0
    transitions = max(total - 1, 1)
    for i in range(1, total):
        prev, curr = trades[i - 1], trades[i]
        if prev.asset == curr.asset and prev.side != curr.side:
            gap_min = minutes_between(prev.timestamp, curr.timestamp)
            if 0 <= gap_min <= SAME_ASSET_FLIP_MIN:
                same_asset_flips += 1
                add_evidence(curr.trade_id, f"same_asset_flip_{curr.asset}_{int(gap_min)}m")

    flip_rate = same_asset_flips / transitions
    flip_score = min(25, int(flip_rate * 250))

    # ── Signal 3: Reactive re-entry after outlier outcomes ────────────────────
    # Uses mean + 2σ so only genuine outlier PnL events trigger this (~2-3% of trades).
    abs_pnls = [t.abs_pnl for t in trades]
    avg_pnl = mean(abs_pnls)
    sd_pnl = stdev(abs_pnls)
    outlier_threshold = avg_pnl + 2 * sd_pnl if sd_pnl > 0 else float("inf")
    REACTION_WINDOW_MIN = 15.0
    reaction_count = 0
    for i in range(1, total):
        prev, curr = trades[i - 1], trades[i]
        if prev.abs_pnl >= outlier_threshold:
            gap = minutes_between(prev.timestamp, curr.timestamp)
            if 0 <= gap <= REACTION_WINDOW_MIN:
                reaction_count += 1
                label = "react_after_big_win" if prev.is_win else "react_after_big_loss"
                add_evidence(curr.trade_id, f"{label}_{int(gap)}m")
    reaction_rate = reaction_count / total if total > 0 else 0.0
    reaction_score = min(20, int(reaction_rate * 300))

    # ── Signal 4: Absolute burst hours ────────────────────────────────────────
    # >30 trades in any single clock-hour is objectively excessive regardless of
    # dataset size. Scored as the proportion of active hours that are burst hours.
    BURST_HOUR_ABS = 30
    by_hour: dict[tuple, list] = defaultdict(list)
    for t in trades:
        by_hour[(t.date, t.hour)].append(t)

    num_hours = len(by_hour)
    avg_hourly = total / num_hours if num_hours > 0 else total
    burst_hours = 0
    for (date, hour), group in by_hour.items():
        if len(group) >= BURST_HOUR_ABS:
            burst_hours += 1
            for t in group:
                add_evidence(t.trade_id, f"burst_hour({len(group)}_trades)")

    burst_hour_rate = burst_hours / num_hours if num_hours > 0 else 0.0
    burst_score = min(20, int(burst_hour_rate * 60))

    # ── Score ──────────────────────────────────────────────────────────────────
    final_score = min(100, gap_score + flip_score + reaction_score + burst_score)
    severity = "low" if final_score < 30 else "med" if final_score < 70 else "high"

    stats = {
        "total_trades": total,
        "flagged_count": len(flagged),
        "median_gap_sec": round(median_gap_sec, 1),
        "same_asset_flips": same_asset_flips,
        "flip_rate": round(flip_rate, 3),
        "reaction_rate": round(reaction_rate, 3),
        "burst_hours": burst_hours,
        "total_hours_active": num_hours,
        "avg_trades_per_hour": round(avg_hourly, 1),
        "outlier_pnl_threshold": round(outlier_threshold, 2),
        "gap_score": gap_score,
        "flip_score": flip_score,
        "reaction_score": reaction_score,
        "burst_score": burst_score,
    }

    return DetectionResult(
        score=final_score,
        severity=severity,
        flagged_trade_ids=list(flagged),
        evidence_by_trade_id=evidence,
        stats=stats,
    )
