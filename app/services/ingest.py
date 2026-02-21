"""CSV ingestion and normalization service."""

from __future__ import annotations

import datetime
import hashlib
import math

import numpy as np
import pandas as pd
from typing import Optional

try:
    import xxhash as _xxhash
    def _hash16(s: str) -> str:
        return f"{_xxhash.xxh3_64_intdigest(s):016x}"
except ImportError:  # pragma: no cover
    def _hash16(s: str) -> str:
        return hashlib.md5(s.encode()).hexdigest()[:16]

from app.models.schemas import NormalizedTrade
from app.utils.csv_utils import (
    normalize_headers,
    validate_required_columns,
    coerce_side,
)
from app.utils.stats_utils import percentile


REQUIRED_COLUMNS = ["timestamp", "asset", "side", "quantity", "entry_price", "exit_price", "profit_loss"]
OPTIONAL_COLUMNS = ["balance"]

HEADER_ALIASES: dict[str, list[str]] = {
    "timestamp": [
        "timestamp",
        "time",
        "datetime",
        "date_time",
        "filled_time",
        "execution_time",
        "open_time",
        "close_time",
        "date",
    ],
    "asset": ["asset", "symbol", "ticker", "instrument", "product"],
    "side": ["side", "action", "buy_sell", "direction"],
    "quantity": ["quantity", "size", "qty", "shares", "units", "amount"],
    "entry_price": ["entry_price", "open_price", "avg_entry_price", "price_in", "buy_price"],
    "exit_price": ["exit_price", "close_price", "avg_exit_price", "price_out", "sell_price"],
    "profit_loss": ["profit_loss", "pnl", "profit", "realized_pnl", "realized_profit", "net_pnl", "pl"],
    "balance": ["balance", "account_balance", "running_balance", "equity"],
}

# Map frontend canonical field names to backend canonical names
FRONTEND_TO_BACKEND_MAPPING = {
    "symbol": "asset",
    "size": "quantity",
    "pnl": "profit_loss",
    "fees": None,
    "open_time": "timestamp",
    "close_time": "timestamp",
}

# Vectorised trade_id: xxh3_64 (falls back to MD5) of pipe-joined fields
def _vec_trade_id(ts: pd.Series, asset: pd.Series, side: pd.Series,
                  qty: pd.Series, ep: pd.Series, xp: pd.Series) -> list[str]:
    # Convert to numpy arrays for faster zip iteration (avoids pandas overhead)
    ts_a   = ts.astype(str).to_numpy()
    ast_a  = asset.to_numpy()
    sid_a  = side.to_numpy()
    qty_a  = qty.to_numpy()
    ep_a   = ep.to_numpy()
    xp_a   = xp.to_numpy()
    return [
        _hash16(f"{t}|{a}|{s}|{q}|{e}|{x}")
        for t, a, s, q, e, x in zip(ts_a, ast_a, sid_a, qty_a, ep_a, xp_a)
    ]


def _normalize_header_key(name: str) -> str:
    """Normalize header key for robust matching."""
    return str(name).lower().strip().replace(" ", "_").replace("-", "_")


def ingest_csv(df: pd.DataFrame, mapping: Optional[dict[str, str]] = None) -> tuple[list[NormalizedTrade], list[str]]:
    """
    Ingest CSV DataFrame and return normalized trades.
    Fully vectorized with no Python row loop.
    """
    errors: list[str] = []

    # Normalize headers
    df = normalize_headers(df)
    df.columns = [_normalize_header_key(col) for col in df.columns]

    # Build col_map
    col_map: dict[str, str] = {}

    if mapping:
        for frontend_field, csv_column in mapping.items():
            if not csv_column:
                continue
            backend_field = FRONTEND_TO_BACKEND_MAPPING.get(frontend_field, frontend_field)
            if backend_field and backend_field in REQUIRED_COLUMNS:
                normalized_csv_col = _normalize_header_key(str(csv_column))
                if normalized_csv_col in df.columns:
                    col_map[backend_field] = normalized_csv_col
                else:
                    errors.append(f"CSV column not found for mapping: {csv_column}")

    # Auto-detect any required column not yet resolved
    for req_col in REQUIRED_COLUMNS:
        if req_col not in col_map:
            for alias in HEADER_ALIASES.get(req_col, [req_col]):
                alias_key = _normalize_header_key(alias)
                if alias_key in df.columns:
                    col_map[req_col] = alias_key
                    break

    if not validate_required_columns(df, REQUIRED_COLUMNS, col_map):
        missing = set(REQUIRED_COLUMNS) - set(col_map.keys())
        errors.append(f"Missing required columns: {missing}")
        return [], errors

    # Pull mapped columns into a working frame
    raw_asset = df[col_map["asset"]]
    wdf = pd.DataFrame({
        "timestamp_raw": df[col_map["timestamp"]],
        "asset":         raw_asset.astype(str).str.strip(),
        "_asset_null":   raw_asset.isna(),
        "side_raw":      df[col_map["side"]],
        "quantity":      pd.to_numeric(df[col_map["quantity"]], errors="coerce"),
        "entry_price":   pd.to_numeric(df[col_map["entry_price"]], errors="coerce"),
        "exit_price":    pd.to_numeric(df[col_map["exit_price"]], errors="coerce"),
        "profit_loss":   pd.to_numeric(df[col_map["profit_loss"]], errors="coerce"),
    })

    # Auto-detect balance column
    balance_col = col_map.get("balance")
    if not balance_col:
        for alias in HEADER_ALIASES["balance"]:
            alias_key = _normalize_header_key(alias)
            if alias_key in df.columns:
                balance_col = alias_key
                break

    if balance_col:
        wdf["balance"] = pd.to_numeric(df[balance_col], errors="coerce")
    else:
        # Running sum from $10,000
        wdf["balance"] = 10_000.0 + wdf["profit_loss"].fillna(0).cumsum()

    # Parse timestamps vectorially
    wdf["timestamp"] = pd.to_datetime(wdf["timestamp_raw"], format="mixed", errors="coerce")

    # Sort by timestamp
    wdf = wdf.sort_values("timestamp").reset_index(drop=True)

    # Infer missing side from price direction / pnl
    side_raw = wdf["side_raw"].astype(str).str.strip().str.upper()
    # Map known aliases
    side_mapped = side_raw.replace({"LONG": "BUY", "SHORT": "SELL", "NAN": "", "NONE": "", "NAT": ""})

    # Where side is blank/unknown, infer from price direction
    unknown_mask = ~side_mapped.isin(["BUY", "SELL"])
    if unknown_mask.any():
        price_diff = wdf["exit_price"] - wdf["entry_price"]
        inferred = pd.Series("BUY", index=wdf.index)
        inferred[price_diff < 0] = "SELL"
        # Fallback to pnl sign when prices are equal
        equal_price = price_diff == 0
        inferred[equal_price & (wdf["profit_loss"] < 0)] = "SELL"
        side_mapped[unknown_mask] = inferred[unknown_mask]

    wdf["side"] = side_mapped

    # Drop rows with unparseable critical values
    required_numeric = ["quantity", "entry_price", "exit_price", "profit_loss", "balance"]
    bad_numeric = wdf[required_numeric].isna().any(axis=1)
    bad_timestamp = wdf["timestamp"].isna()
    bad_asset = wdf["_asset_null"] | wdf["asset"].isin(["", "nan", "none", "NaN", "None"])
    bad_mask = bad_numeric | bad_timestamp | bad_asset

    if bad_mask.any():
        for orig_idx in wdf[bad_mask].index:
            errors.append(f"Row {orig_idx + 2}: could not parse required values - skipped")
        wdf = wdf[~bad_mask].reset_index(drop=True)

    if wdf.empty:
        errors.append("No valid trades after parsing.")
        return [], errors

    # Derived columns (all vectorized)
    wdf["date"]     = wdf["timestamp"].dt.strftime("%Y-%m-%d")
    wdf["hour"]     = wdf["timestamp"].dt.hour
    wdf["weekday"]  = wdf["timestamp"].dt.weekday
    wdf["is_win"]   = wdf["profit_loss"] > 0
    wdf["abs_pnl"]  = wdf["profit_loss"].abs()
    wdf["size_usd"] = wdf["quantity"] * wdf["entry_price"]

    safe_size = wdf["size_usd"].replace(0, np.nan)
    rp = wdf["profit_loss"] / safe_size
    rp = rp.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    wdf["return_pct"] = rp

    # Stable trade_id returns a list[str] directly (no .astype(str) needed)
    trade_ids = _vec_trade_id(
        wdf["timestamp"], wdf["asset"], wdf["side"],
        wdf["quantity"], wdf["entry_price"], wdf["exit_price"],
    )

    # Build NormalizedTrade list directly from column arrays
    # Avoids the slow to_dict("records") + per-row dict access pattern.
    # Convert pandas Timestamps to Python datetimes in one vectorized call
    timestamps  = wdf["timestamp"].dt.to_pydatetime().tolist()
    assets      = wdf["asset"].tolist()
    sides       = wdf["side"].tolist()
    quantities  = wdf["quantity"].tolist()
    entry_prices= wdf["entry_price"].tolist()
    exit_prices = wdf["exit_price"].tolist()
    pnls        = wdf["profit_loss"].tolist()
    balances    = wdf["balance"].tolist()
    dates       = wdf["date"].tolist()
    hours       = wdf["hour"].tolist()
    weekdays    = wdf["weekday"].tolist()
    is_wins     = wdf["is_win"].tolist()
    abs_pnls    = wdf["abs_pnl"].tolist()
    sizes_usd   = wdf["size_usd"].tolist()
    ret_pcts    = wdf["return_pct"].tolist()

    normalized_trades = [
        NormalizedTrade(
            trade_id    = trade_ids[i],
            timestamp   = timestamps[i],
            asset       = assets[i],
            side        = sides[i],
            quantity    = quantities[i],
            entry_price = entry_prices[i],
            exit_price  = exit_prices[i],
            profit_loss = pnls[i],
            balance     = balances[i],
            date        = dates[i],
            hour        = hours[i],
            weekday     = weekdays[i],
            is_win      = is_wins[i],
            abs_pnl     = abs_pnls[i],
            size_usd    = sizes_usd[i],
            return_pct  = ret_pcts[i],
        )
        for i in range(len(wdf))
    ]

    return normalized_trades, errors
