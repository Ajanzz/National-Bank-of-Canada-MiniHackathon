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
    rows = []
    for t in trades:
        rows.append({
            "timestamp": t.timestamp,
            "symbol": t.asset,
            "side": t.side,
            "quantity": t.quantity,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "pnl": t.profit_loss,
            "balance": t.balance,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


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
        running_max = bal.cummax()
        drawdown = ((running_max - bal) / running_max.replace(0, 1)).max()
        balance_drawdown_pct = float(drawdown) * 100
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
    Returns a list of 18-feature vectors (one per window).
    """
    df = _normalize_df(df)
    features = []
    for start in range(0, len(df) - window_size + 1, stride):
        window = df.iloc[start : start + window_size]
        features.append(extract_features(window))
    return features


def extract_features_from_trades(
    trades: list[NormalizedTrade],
    window_size: int = 50,
    stride: int = 25,
) -> list[np.ndarray]:
    """
    Convert NormalizedTrade list to DataFrame and extract windowed features.
    If fewer trades than window_size, extract a single feature vector from all trades.
    """
    df = trades_to_dataframe(trades)
    if len(df) < 2:
        return [np.zeros(len(FEATURE_NAMES), dtype=float)]
    if len(df) >= window_size:
        return extract_windowed_features(df, window_size=window_size, stride=stride)
    return [extract_features(df)]
