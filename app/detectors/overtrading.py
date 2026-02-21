"""Overtrading bias detector."""

from collections import defaultdict
from datetime import timedelta

import pandas as pd

from app.models.schemas import DetectionResult, NormalizedTrade
from app.utils.time_utils import minutes_between
from app.utils.stats_utils import percentile


def detect_overtrading(trades: list[NormalizedTrade]) -> DetectionResult:
    """
    Detect overtrading bias:
    - Daily trade count thresholds
    - Rolling 10-minute burst count
    - Score uses weighted components
    """
    if not trades:
        return DetectionResult(score=0, severity="low", flagged_trade_ids=[], evidence_by_trade_id={}, stats={})
    
    flagged = set()
    evidence = {}
    
    # Group by date
    by_date = defaultdict(list)
    for trade in trades:
        by_date[trade.date].append(trade)
    
    # Daily trade count analysis
    daily_counts = []
    daily_overtrading = {}
    for date_str, day_trades in by_date.items():
        count = len(day_trades)
        daily_counts.append(count)
        if count >= 20:  # Threshold for high daily activity (stricter: was 10)
            daily_overtrading[date_str] = count
    
    daily_threshold = percentile(daily_counts, 75) if daily_counts else 0
    
    # Rolling 10-minute burst analysis
    burst_flagged = set()
    for i, trade in enumerate(trades):
        # Find all trades within 10 min window
        burst_trades = []
        for j in range(max(0, i - 10), min(len(trades), i + 10)):
            if minutes_between(trades[j].timestamp, trade.timestamp) <= 10:
                burst_trades.append(trades[j])
        
        if len(burst_trades) >= 8:  # Stricter: was 5
            burst_flagged.add(trade.trade_id)
    
    # Hold time analysis (percentile-based)
    holds_in_minutes = []
    for i, trade in enumerate(trades):
        if i + 1 < len(trades):
            hold_min = minutes_between(trade.timestamp, trades[i + 1].timestamp)
            holds_in_minutes.append(hold_min)
    
    hold_p10 = percentile(holds_in_minutes, 10) if holds_in_minutes else 0  # Stricter: was p25
    
    # Flag trades with short hold times in high-frequency windows
    for i, trade in enumerate(trades):
        score_components = []
        
        # Check if in daily overtrading window
        if trade.date in daily_overtrading and daily_overtrading[trade.date] >= 20:
            score_components.append(("daily_high_activity", 35))
        
        # Check if in burst
        if trade.trade_id in burst_flagged:
            score_components.append(("burst_window", 40))
        
        # Check hold time vs percentile (stricter)
        if i + 1 < len(trades):
            hold_min = minutes_between(trade.timestamp, trades[i + 1].timestamp)
            if hold_min < hold_p10 and hold_p10 > 0:
                score_components.append(("short_hold", 35))
        
        if score_components:
            flagged.add(trade.trade_id)
            evidence[trade.trade_id] = " + ".join(str(c[0]) for c in score_components)
    
    # Compute score
    flagged_count = len(flagged)
    total_trades = len(trades)
    base_score = int((flagged_count / total_trades) * 100) if total_trades > 0 else 0
    
    # Adjust for intensity (reduced bonus: was up to 30)
    max_daily = max(daily_counts) if daily_counts else 0
    intensity_bonus = min(15, (max_daily - 10) * 1) if max_daily > 10 else 0  # Stricter
    
    final_score = min(100, base_score + intensity_bonus)
    
    severity = "low" if final_score < 30 else "med" if final_score < 70 else "high"
    
    stats = {
        "flagged_count": flagged_count,
        "total_trades": total_trades,
        "max_daily_trades": int(max_daily),
        "avg_daily_trades": round(sum(daily_counts) / len(daily_counts), 2) if daily_counts else 0,
        "hold_time_p10_minutes": round(hold_p10, 2),
    }
    
    return DetectionResult(
        score=final_score,
        severity=severity,
        flagged_trade_ids=list(flagged),
        evidence_by_trade_id=evidence,
        stats=stats,
    )
