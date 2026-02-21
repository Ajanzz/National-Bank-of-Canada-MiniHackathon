"""CSV utility functions."""

from __future__ import annotations

import io
import base64
from datetime import datetime
from typing import Optional

import pandas as pd


def parse_csv(csv_content: str) -> pd.DataFrame:
    """Parse CSV content into DataFrame."""
    df = pd.read_csv(io.StringIO(csv_content))
    return df


def parse_base64_csv(base64_csv: str) -> pd.DataFrame:
    """Parse base64-encoded CSV."""
    csv_bytes = base64.b64decode(base64_csv)
    csv_content = csv_bytes.decode('utf-8')
    return parse_csv(csv_content)


def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column headers: lowercase, strip whitespace."""
    df.columns = df.columns.str.lower().str.strip()
    return df


def validate_required_columns(df: pd.DataFrame, required: list[str], col_map: Optional[dict[str, str]] = None) -> bool:
    """Check if all required columns exist (case-insensitive or via col_map)."""
    if col_map:
        # Check that all required columns are in the mapping
        required_lower = set(c.lower() for c in required)
        mapped_keys = set(k.lower() for k in col_map.keys())
        return required_lower.issubset(mapped_keys)
    else:
        # Original logic: check case-insensitive column names
        df_cols = set(df.columns.str.lower())
        required_lower = set(c.lower() for c in required)
        return required_lower.issubset(df_cols)


def coerce_numeric(value):
    """Coerce value to float, return None if fails."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def coerce_side(side_str) -> str:
    """Coerce side to BUY or SELL."""
    if side_str is None or (isinstance(side_str, float) and side_str != side_str):
        raise ValueError("Missing side value")
    s = str(side_str).upper().strip()
    if s in ("BUY", "LONG"):
        return "BUY"
    elif s in ("SELL", "SHORT"):
        return "SELL"
    else:
        raise ValueError(f"Invalid side: {side_str}")


def coerce_timestamp(ts_str: str, lazy: bool = True) -> datetime:
    """
    Coerce timestamp string to datetime.
    Supports ISO8601 and "YYYY-MM-DD HH:MM:SS".
    If lazy=True, handles naive timestamps by converting to UTC-aware.
    """
    ts_str = str(ts_str).strip()
    
    # Try common formats
    formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(ts_str, fmt)
            if lazy and dt.tzinfo is None:
                # Treat as UTC
                dt = dt.replace(tzinfo=None)
            return dt
        except ValueError:
            continue
    
    raise ValueError(f"Could not parse timestamp: {ts_str}")
