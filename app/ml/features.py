"""
Feature engineering for XGBoost bias classification.

Extracts 18 per-window behavioral features from raw trade data.
Works with DataFrames using our CSV column format:
    timestamp, asset, side, quantity, entry_price, exit_price, profit_loss, balance

Based on the BiasLens QHacks 2026 feature pipeline.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from app.models.schemas import NormalizedTrade

# Ordered feature list — must stay in sync with extract_features()
FEATURE_NAMES: list[str] = [
    "trades_per_hour",
    "mean_time_between_trades_sec",
    "std_time_between_trades_sec",
    "burst_count_60s",
    "pnl_mean",
    "pnl_std",
    "pnl_total",
    "win_rate",
    "avg_quantity",
    "std_quantity",
    "avg_trade_value",
    "loss_hold_to_win_hold_ratio",
    "avg_size_after_loss_ratio",
    "reentry_after_loss_mean_sec",
    "consecutive_loss_streak_max",
    "side_switch_rate",
    "unique_symbols",
    "balance_drawdown_pct",
]


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Rename our internal column names to the canonical names used by features."""
    df = df.copy()
    rename_map = {
        "asset": "symbol",
        "profit_loss": "pnl",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def trades_to_dataframe(trades: list[NormalizedTrade]) -> pd.DataFrame:
    """Convert a list of NormalizedTrade objects to a pandas DataFrame."""
    if not trades:
        return pd.DataFrame()
    # Build column arrays directly — avoids per-row dict allocation
    df = pd.DataFrame({
        "timestamp":   pd.to_datetime([t.timestamp for t in trades], utc=True, errors="coerce"),
        "symbol":      [t.asset        for t in trades],
        "side":        [t.side         for t in trades],
        "quantity":    [t.quantity     for t in trades],
        "entry_price": [t.entry_price  for t in trades],
        "exit_price":  [t.exit_price   for t in trades],
        "pnl":         [t.profit_loss  for t in trades],
        "balance":     [t.balance      for t in trades],
    })
    return df.sort_values("timestamp").reset_index(drop=True)


def extract_features(df: pd.DataFrame) -> np.ndarray:
    """
    Extract an 18-feature vector from a DataFrame window of trades.

    Accepts DataFrames with either:
    - Raw CSV format: asset/symbol, profit_loss/pnl columns
    - Internal normalized format

    Returns a 1-D float numpy array matching FEATURE_NAMES order.
    """
    df = _normalize_df(df)

    n = len(df)
    if n < 2:
        return np.zeros(len(FEATURE_NAMES), dtype=float)

    # ── Time features ──────────────────────────────────────────────────
    deltas = df["timestamp"].diff().dt.total_seconds().dropna()
    total_hours = max(
        (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 3600,
        0.001,
    )
    trades_per_hour = n / total_hours
    mean_delta = float(deltas.mean()) if len(deltas) > 0 else 0.0
    std_delta = float(deltas.std(ddof=0)) if len(deltas) > 0 else 0.0

    # Burst: trades within 60s of each other
    burst_count = int((deltas <= 60).sum())

    # ── PnL features ───────────────────────────────────────────────────
    pnl = df["pnl"].astype(float)
    pnl_mean = float(pnl.mean())
    pnl_std = float(pnl.std(ddof=0))
    pnl_total = float(pnl.sum())
    win_rate = float((pnl > 0).sum() / n)

    # ── Size features ──────────────────────────────────────────────────
    qty = df["quantity"].astype(float)
    avg_qty = float(qty.mean())
    std_qty = float(qty.std(ddof=0))

    # Trade value = quantity × entry_price
    if "entry_price" in df.columns:
        trade_values = qty * df["entry_price"].astype(float)
    elif "price" in df.columns:
        trade_values = qty * df["price"].astype(float)
    else:
        trade_values = qty
    avg_trade_value = float(trade_values.mean())

    # ── Loss aversion: hold-time proxy ─────────────────────────────────
    wins_mask = pnl > 0
    losses_mask = pnl <= 0
    # Use inter-trade time as a proxy for hold duration
    hold_proxy = deltas.reindex(range(n), fill_value=float(deltas.median()) if len(deltas) > 0 else 60.0)
    win_hold = hold_proxy[wins_mask].mean() if wins_mask.sum() > 0 else 1.0
    loss_hold = hold_proxy[losses_mask].mean() if losses_mask.sum() > 0 else 1.0
    loss_hold_to_win_hold_ratio = float(loss_hold / max(win_hold, 0.001))

    # ── Revenge trading features ────────────────────────────────────────
    sizes_after_loss: list[float] = []
    reentry_times_after_loss: list[float] = []

    pnl_vals = pnl.values
    qty_vals = qty.values
    delta_vals = deltas.values  # length n-1

    for i in range(1, n):
        prev_pnl = float(pnl_vals[i - 1])
        curr_qty = float(qty_vals[i])
        prev_qty = float(qty_vals[i - 1])

        if prev_pnl < 0:
            sizes_after_loss.append(curr_qty / max(prev_qty, 0.001))
            if i - 1 < len(delta_vals):
                reentry_times_after_loss.append(float(delta_vals[i - 1]))

    avg_size_after_loss_ratio = float(np.mean(sizes_after_loss)) if sizes_after_loss else 1.0
    reentry_after_loss_mean = float(np.mean(reentry_times_after_loss)) if reentry_times_after_loss else 300.0

    # Max consecutive loss streak
    max_streak = 0
    current_streak = 0
    for p in pnl_vals:
        if p <= 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    # ── Diversity features ──────────────────────────────────────────────
    side_changes = int((df["side"].str.lower() != df["side"].str.lower().shift()).sum())
    side_switch_rate = float(side_changes / max(n - 1, 1))
    unique_symbols = int(df["symbol"].nunique()) if "symbol" in df.columns else 1

    # ── Balance drawdown ────────────────────────────────────────────────
    if "balance" in df.columns:
        bal = df["balance"].astype(float)
        # Shift balance so the minimum is always >= 1 before computing drawdown.
        # This handles datasets where balance goes negative (e.g. overtrading).
        bal_floor = bal.min()
        if bal_floor <= 0:
            bal = bal - bal_floor + 1.0  # shift so min == 1
        running_max = bal.cummax()
        # Avoid division by zero; running_max is always >= 1 after shift
        drawdown = ((running_max - bal) / running_max).max()
        balance_drawdown_pct = float(np.clip(drawdown, 0.0, 1.0)) * 100
    else:
        balance_drawdown_pct = 0.0

    return np.array(
        [
            trades_per_hour,
            mean_delta,
            std_delta,
            burst_count,
            pnl_mean,
            pnl_std,
            pnl_total,
            win_rate,
            avg_qty,
            std_qty,
            avg_trade_value,
            loss_hold_to_win_hold_ratio,
            avg_size_after_loss_ratio,
            reentry_after_loss_mean,
            max_streak,
            side_switch_rate,
            float(unique_symbols),
            balance_drawdown_pct,
        ],
        dtype=float,
    )


def extract_windowed_features(
    df: pd.DataFrame,
    window_size: int = 50,
    stride: int = 25,
) -> list[np.ndarray]:
    """
    Slide a window over the trade DataFrame and extract features per window.
    Precomputes all column arrays once as numpy arrays so each window only
    does index arithmetic — no per-window pandas overhead.
    """
    df = _normalize_df(df)
    n_total = len(df)
    if n_total < window_size:
        return [extract_features(df)]

    # ── Precompute arrays once ───────────────────────────────────────────
    ts_ns   = df["timestamp"].values.astype("int64")          # nanoseconds
    pnl_a   = df["pnl"].to_numpy(dtype=float)
    qty_a   = df["quantity"].to_numpy(dtype=float)
    ep_a    = df["entry_price"].to_numpy(dtype=float)
    bal_a   = df["balance"].to_numpy(dtype=float) if "balance" in df.columns else None
    side_a  = df["side"].str.lower().to_numpy()
    sym_a   = df["symbol"].to_numpy() if "symbol" in df.columns else None

    # Per-row time deltas in seconds (length n_total, first element = 0)
    delta_ns = np.empty(n_total, dtype=float)
    delta_ns[0] = 0.0
    delta_ns[1:] = (ts_ns[1:] - ts_ns[:-1]) / 1e9  # ns → s

    # Trade values
    tv_a = qty_a * ep_a

    features = []
    starts = range(0, n_total - window_size + 1, stride)
    for s in starts:
        e = s + window_size
        # Window slices (views, no copy)
        ts_w    = ts_ns[s:e]
        pnl_w   = pnl_a[s:e]
        qty_w   = qty_a[s:e]
        tv_w    = tv_a[s:e]
        side_w  = side_a[s:e]
        # Deltas within window: ts_w[i] - ts_w[i-1]  (length w, first=0)
        dw = np.empty(window_size, dtype=float)
        dw[0] = 0.0
        dw[1:] = (ts_w[1:] - ts_w[:-1]) / 1e9
        deltas_w = dw[1:]  # length w-1, matches pandas .diff().dropna()

        # ── Time features ──────────────────────────────────────────────
        total_sec = float(ts_w[-1] - ts_w[0]) / 1e9
        total_hours = max(total_sec / 3600, 0.001)
        tph = window_size / total_hours
        mean_d = float(deltas_w.mean()) if len(deltas_w) else 0.0
        std_d  = float(deltas_w.std())  if len(deltas_w) else 0.0
        burst  = int((deltas_w <= 60).sum())

        # ── PnL features ───────────────────────────────────────────────
        pnl_mean  = float(pnl_w.mean())
        pnl_std   = float(pnl_w.std())
        pnl_total = float(pnl_w.sum())
        win_rate  = float((pnl_w > 0).sum() / window_size)

        # ── Size features ──────────────────────────────────────────────
        avg_qty = float(qty_w.mean())
        std_qty = float(qty_w.std())
        avg_tv  = float(tv_w.mean())

        # ── Loss aversion: hold-time proxy ─────────────────────────────
        wins_mask   = pnl_w > 0
        losses_mask = ~wins_mask
        # Use deltas as hold proxy; index deltas: trade i corresponds to delta[i-1]
        hold_proxy = np.empty(window_size, dtype=float)
        med_d = float(np.median(deltas_w)) if len(deltas_w) else 60.0
        hold_proxy[0] = med_d
        hold_proxy[1:] = deltas_w
        win_hold  = float(hold_proxy[wins_mask].mean())   if wins_mask.sum()   > 0 else 1.0
        loss_hold = float(hold_proxy[losses_mask].mean()) if losses_mask.sum() > 0 else 1.0
        la_ratio  = loss_hold / max(win_hold, 0.001)

        # ── Revenge trading features ────────────────────────────────────
        loss_prev = pnl_w[:-1] < 0
        if loss_prev.sum() > 0:
            curr_qty_al  = qty_w[1:][loss_prev]
            prev_qty_al  = qty_w[:-1][loss_prev]
            sz_ratio_arr = curr_qty_al / np.maximum(prev_qty_al, 0.001)
            avg_sz_ratio = float(sz_ratio_arr.mean())
            re_deltas    = deltas_w[loss_prev[:len(deltas_w)]] if len(deltas_w) > 0 else np.array([300.0])
            reentry_mean = float(re_deltas.mean()) if len(re_deltas) > 0 else 300.0
        else:
            avg_sz_ratio = 1.0
            reentry_mean = 300.0

        # Max consecutive loss streak (vectorised via run-length encoding)
        losses_bool = (pnl_w <= 0).view(np.uint8)
        streak = 0
        max_str = 0
        for v in losses_bool:
            if v:
                streak += 1
                if streak > max_str:
                    max_str = streak
            else:
                streak = 0

        # ── Diversity features ──────────────────────────────────────────
        side_changes = int((side_w[1:] != side_w[:-1]).sum())
        side_switch  = float(side_changes / max(window_size - 1, 1))
        unique_syms  = float(len(set(sym_a[s:e]))) if sym_a is not None else 1.0

        # ── Balance drawdown ────────────────────────────────────────────
        if bal_a is not None:
            bw = bal_a[s:e].copy()
            bf = bw.min()
            if bf <= 0:
                bw = bw - bf + 1.0
            rmax = np.maximum.accumulate(bw)
            dd = float(np.clip(((rmax - bw) / rmax).max(), 0.0, 1.0)) * 100
        else:
            dd = 0.0

        features.append(np.array([
            tph, mean_d, std_d, burst,
            pnl_mean, pnl_std, pnl_total, win_rate,
            avg_qty, std_qty, avg_tv,
            la_ratio, avg_sz_ratio, reentry_mean, float(max_str),
            side_switch, unique_syms, dd,
        ], dtype=float))

    return features


def extract_features_from_trades(
    trades: list[NormalizedTrade],
    window_size: int = 50,
    stride: int = 25,
    max_windows: int = 100,
) -> list[np.ndarray]:
    """
    Convert NormalizedTrade list to DataFrame and extract windowed features.
    If fewer trades than window_size, extract a single feature vector from all trades.
    Uses an adaptive stride so the number of windows never exceeds max_windows,
    keeping ML inference fast on large datasets (e.g. 10k trades).
    """
    df = trades_to_dataframe(trades)
    if len(df) < 2:
        return [np.zeros(len(FEATURE_NAMES), dtype=float)]
    if len(df) < window_size:
        return [extract_features(df)]
    # Adaptive stride: ensure we don't generate more than max_windows windows
    n_possible = (len(df) - window_size) // stride + 1
    if n_possible > max_windows:
        stride = max(window_size, (len(df) - window_size) // (max_windows - 1))
    return extract_windowed_features(df, window_size=window_size, stride=stride)
