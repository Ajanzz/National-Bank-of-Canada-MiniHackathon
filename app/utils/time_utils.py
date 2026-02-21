"""Time utility functions."""

from datetime import datetime, timedelta


def timestamp_to_hour(dt: datetime) -> int:
    """Extract hour from datetime (0-23)."""
    return dt.hour


def timestamp_to_weekday(dt: datetime) -> int:
    """Extract weekday from datetime (0=Monday, 6=Sunday)."""
    return dt.weekday()


def timestamp_to_date_string(dt: datetime) -> str:
    """Convert datetime to YYYY-MM-DD string."""
    return dt.strftime("%Y-%m-%d")


def minutes_between(dt1: datetime, dt2: datetime) -> float:
    """Calculate minutes between two datetimes."""
    delta = abs((dt2 - dt1).total_seconds())
    return delta / 60.0


def seconds_between(dt1: datetime, dt2: datetime) -> float:
    """Calculate seconds between two datetimes."""
    delta = abs((dt2 - dt1).total_seconds())
    return delta
