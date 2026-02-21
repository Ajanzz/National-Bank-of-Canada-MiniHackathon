"""CSV ingestion and normalization service."""

from __future__ import annotations

import pandas as pd
from datetime import datetime
from typing import Optional

from app.models.schemas import NormalizedTrade
from app.utils.csv_utils import (
    normalize_headers,
    validate_required_columns,
    coerce_numeric,
    coerce_side,
    coerce_timestamp,
)
from app.utils.time_utils import (
    timestamp_to_hour,
    timestamp_to_weekday,
    timestamp_to_date_string,
)
from app.utils.stats_utils import compute_trade_id


REQUIRED_COLUMNS = ["timestamp", "asset", "side", "quantity", "entry_price", "exit_price", "profit_loss"]
OPTIONAL_COLUMNS = ["balance"]

# Map frontend canonical field names to backend canonical names
FRONTEND_TO_BACKEND_MAPPING = {
    "symbol": "asset",
    "size": "quantity",
    "pnl": "profit_loss",
    "fees": None,  # Optional, not required
    "open_time": "timestamp",
    "close_time": "timestamp",
    # Everything else maps to itself
}


def ingest_csv(df: pd.DataFrame, mapping: Optional[dict[str, str]] = None) -> tuple[list[NormalizedTrade], list[str]]:
    """
    Ingest CSV DataFrame and return normalized trades.
    
    Args:
        df: DataFrame from parse_csv
        mapping: Optional mapping from canonical field names to DataFrame column names
                 e.g., {"timestamp": "Timestamp", "asset": "Symbol", "side": "Direction", ...}
    
    Returns:
        (normalized_trades, errors)
    """
    errors = []
    
    # Normalize headers
    df = normalize_headers(df)
    
    # Build column mapping either from provided mapping or by name matching
    col_map = {}
    
    if mapping:
        # mapping comes from frontend: frontend_field_name -> csv_column_name
        # We need to convert to: backend_field_name -> csv_column
        for frontend_field, csv_column in mapping.items():
            # Translate frontend field name to backend field name
            backend_field = FRONTEND_TO_BACKEND_MAPPING.get(frontend_field, frontend_field)
            
            if backend_field and backend_field in REQUIRED_COLUMNS:
                # Find the column in df that matches the CSV column name (case-insensitive)
                normalized_csv_col = csv_column.lower().strip()
                found = False
                for df_col in df.columns:
                    if df_col.lower() == normalized_csv_col:
                        col_map[backend_field] = df_col
                        found = True
                        break
                if not found:
                    errors.append(f"CSV column not found for mapping: {csv_column}")
    else:
        # Fallback to case-insensitive column name matching
        for req_col in REQUIRED_COLUMNS:
            for df_col in df.columns:
                if df_col.lower() == req_col.lower():
                    col_map[req_col] = df_col
                    break
    
    # Validate columns
    if not validate_required_columns(df, REQUIRED_COLUMNS, col_map):
        missing = set(REQUIRED_COLUMNS) - set(col_map.keys())
        errors.append(f"Missing required columns: {missing}")
        return [], errors
    
    # Sort by timestamp
    try:
        df["_timestamp_parsed"] = df[col_map["timestamp"]].apply(lambda x: coerce_timestamp(x))
        df = df.sort_values("_timestamp_parsed").reset_index(drop=True)
    except Exception as e:
        errors.append(f"Failed to parse timestamps: {str(e)}")
        return [], errors
    
    # Auto-detect balance column if not in mapping
    balance_col = col_map.get("balance")
    if not balance_col:
        # Look for balance column in CSV
        for df_col in df.columns:
            if df_col.lower() in ("balance", "account_balance", "equity"):
                balance_col = df_col
                break
    
    # Normalize trades
    normalized_trades = []
    running_balance = None
    
    for idx, row in df.iterrows():
        try:
            timestamp = coerce_timestamp(row[col_map["timestamp"]])
            asset = str(row[col_map["asset"]]).strip()
            side = coerce_side(row[col_map["side"]])
            quantity = coerce_numeric(row[col_map["quantity"]])
            entry_price = coerce_numeric(row[col_map["entry_price"]])
            exit_price = coerce_numeric(row[col_map["exit_price"]])
            profit_loss = coerce_numeric(row[col_map["profit_loss"]])
            
            # Get or compute balance
            if balance_col:
                balance = coerce_numeric(row[balance_col])
            else:
                # Compute running balance from cumulative PnL
                if running_balance is None:
                    running_balance = 10000.0  # Start with $10k
                running_balance += profit_loss
                balance = running_balance
            
            # Validate coerced values
            if any(v is None for v in [quantity, entry_price, exit_price, profit_loss, balance]):
                errors.append(f"Row {idx + 1}: Failed to coerce numeric values")
                continue
            
            # Compute fields
            trade_id = compute_trade_id(timestamp.isoformat(), asset, side, quantity, entry_price, exit_price)
            date = timestamp_to_date_string(timestamp)
            hour = timestamp_to_hour(timestamp)
            weekday = timestamp_to_weekday(timestamp)
            is_win = profit_loss > 0
            abs_pnl = abs(profit_loss)
            size_usd = quantity * entry_price
            return_pct = (profit_loss / size_usd) if size_usd > 0 else 0.0
            
            trade = NormalizedTrade(
                trade_id=trade_id,
                timestamp=timestamp,
                asset=asset,
                side=side,
                quantity=quantity,
                entry_price=entry_price,
                exit_price=exit_price,
                profit_loss=profit_loss,
                balance=balance,
                date=date,
                hour=hour,
                weekday=weekday,
                is_win=is_win,
                abs_pnl=abs_pnl,
                size_usd=size_usd,
                return_pct=return_pct,
            )
            normalized_trades.append(trade)
        except Exception as e:
            errors.append(f"Row {idx + 1}: {str(e)}")
    
    return normalized_trades, errors
