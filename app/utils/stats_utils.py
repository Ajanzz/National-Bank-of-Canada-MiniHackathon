"""Statistics utility functions."""

import hashlib
from typing import Any


def compute_trade_id(timestamp: str, asset: str, side: str, quantity: float, entry_price: float, exit_price: float) -> str:
    """
    Compute stable trade_id as hash of: timestamp+asset+side+quantity+entry_price+exit_price.
    """
    key = f"{timestamp}|{asset}|{side}|{quantity}|{entry_price}|{exit_price}"
    return hashlib.md5(key.encode()).hexdigest()[:16]


def percentile(values: list[float], p: float) -> float:
    """Calculate percentile of a list of values."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = (len(sorted_vals) - 1) * (p / 100.0)
    lower = int(idx)
    upper = lower + 1
    weight = idx - lower
    if upper >= len(sorted_vals):
        return sorted_vals[lower]
    return sorted_vals[lower] * (1 - weight) + sorted_vals[upper] * weight


def median(values: list[float]) -> float:
    """Calculate median."""
    return percentile(values, 50)


def mean(values: list[float]) -> float:
    """Calculate mean."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def stdev(values: list[float]) -> float:
    """Calculate standard deviation."""
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    variance = sum((x - avg) ** 2 for x in values) / len(values)
    return variance ** 0.5
