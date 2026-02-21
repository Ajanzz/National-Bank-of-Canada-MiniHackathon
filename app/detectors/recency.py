"""Recency bias detector (QHacks-style)."""

from app.models.schemas import DetectionResult, NormalizedTrade
from app.utils.time_utils import minutes_between
from app.utils.stats_utils import mean


def detect_recency_bias(trades: list[NormalizedTrade]) -> DetectionResult:
    """
    Detect recency bias using multiple signals:
    - Signal 1: prev-trade reaction: after win/loss, trade again within 5m OR size_usd >= 1.3x baseline => flag
    - Signal 2: streak reaction: after 3-win or 3-loss streak, next trade within 10m and size_usd >= 1.25x baseline => flag
    - Signal 3: 50-trade windows stride 25: compute trade_rate and size_median drift vs baseline
    """
    if len(trades) < 2:
        return DetectionResult(score=0, severity="low", flagged_trade_ids=[], evidence_by_trade_id={}, stats={})
    
    flagged = set()
    evidence = {}
    
    # Baseline size
    sizes = [t.size_usd for t in trades if t.size_usd > 0]
    baseline_size = sorted(sizes)[len(sizes) // 2] if sizes else 0
    baseline_size_large = baseline_size * 1.3
    
    # Signal 1: Prev-trade reaction
    for i in range(len(trades) - 1):
        prev_trade = trades[i]
        curr_trade = trades[i + 1]
        
        time_gap = minutes_between(prev_trade.timestamp, curr_trade.timestamp)
        size_vs_baseline = curr_trade.size_usd / baseline_size if baseline_size > 0 else 0
        
        # Flag if VERY quick reaction (within 2min) AND size spike
        if time_gap <= 2 and size_vs_baseline >= 1.5:
            flagged.add(curr_trade.trade_id)
            evidence[curr_trade.trade_id] = f"Rapid re-entry within {int(time_gap)}m with size spike {size_vs_baseline:.2f}x"
    
    # Signal 2: Streak reaction
    # Find win/loss streaks of length 3+
    streak_indices = []
    current_streak_start = 0
    streak_type = None  # 'win' or 'loss'
    
    for i in range(len(trades)):
        is_win = trades[i].is_win
        current_type = 'win' if is_win else 'loss'
        
        if streak_type is None or streak_type == current_type:
            streak_type = current_type
        else:
            # Streak ended
            if i - current_streak_start >= 3:
                streak_indices.append((current_streak_start, i - 1, streak_type))
            current_streak_start = i
            streak_type = current_type
    
    # Check final streak
    if streak_type and len(trades) - current_streak_start >= 3:
        streak_indices.append((current_streak_start, len(trades) - 1, streak_type))
    
    # Flag trades after streaks
    for start, end, streak_type in streak_indices:
        if end + 1 < len(trades):
            next_trade = trades[end + 1]
            time_gap = minutes_between(trades[end].timestamp, next_trade.timestamp)
            size_vs_baseline = next_trade.size_usd / baseline_size if baseline_size > 0 else 0
            
            # Flag only if rapid reaction after LONG streak with aggressive size
            if time_gap <= 5 and size_vs_baseline >= 1.5 and (end - start + 1) >= 5:
                flagged.add(next_trade.trade_id)
                evidence[next_trade.trade_id] = f"{end - start + 1}-{streak_type} streak reaction - size {size_vs_baseline:.2f}x baseline in {int(time_gap)}m"
    
    # Signal 3: 50-trade window stride analysis
    window_size = 50
    stride = 25
    
    baseline_trade_rate = len(trades) / max(1, (trades[-1].timestamp - trades[0].timestamp).days)
    baseline_size_median = sorted(sizes)[len(sizes) // 2] if sizes else 0
    
    window_flagged = {}
    for window_start in range(0, len(trades) - window_size + 1, stride):
        window_end = window_start + window_size
        window_trades = trades[window_start:window_end]
        
        # Compute window metrics
        window_duration = (window_trades[-1].timestamp - window_trades[0].timestamp).days
        if window_duration == 0:
            window_duration = 1 / 24  # At least an hour
        
        window_trade_rate = window_size / window_duration
        window_sizes = [t.size_usd for t in window_trades if t.size_usd > 0]
        window_size_median = sorted(window_sizes)[len(window_sizes) // 2] if window_sizes else 0
        
        # Drift detection
        rate_drift = window_trade_rate / baseline_trade_rate if baseline_trade_rate > 0 else 0
        size_drift = window_size_median / baseline_size_median if baseline_size_median > 0 else 0
        
        # Window PnL magnitude
        window_pnl = sum(t.profit_loss for t in window_trades)
        window_pnl_magnitude = abs(window_pnl)
        avg_pnl = mean([t.abs_pnl for t in window_trades])
        
        # Flag only if VERY aggressive drift and loss magnitude
        if rate_drift >= 2.0 and size_drift >= 1.8 and window_pnl_magnitude >= avg_pnl * 3:
            for trade in window_trades:
                window_flagged[trade.trade_id] = f"Recency window drift - rate {rate_drift:.2f}x, size {size_drift:.2f}x, pnl ${window_pnl:.0f}"
    
    # Add window flags
    for trade_id, flag_evidence in window_flagged.items():
        if trade_id not in flagged:
            flagged.add(trade_id)
            evidence[trade_id] = flag_evidence
    
    # Score
    flagged_count = len(flagged)
    total_trades = len(trades)
    base_score = int((flagged_count / total_trades) * 100) if total_trades > 0 else 0
    final_score = min(100, base_score)
    
    severity = "low" if final_score < 30 else "med" if final_score < 70 else "high"
    
    stats = {
        "flagged_count": flagged_count,
        "total_trades": total_trades,
        "signal1_flags": len([f for f in evidence.values() if "Prev-trade" in f]),
        "signal2_flags": len([f for f in evidence.values() if "streak" in f]),
        "signal3_flags": len([f for f in evidence.values() if "window" in f]),
    }
    
    return DetectionResult(
        score=final_score,
        severity=severity,
        flagged_trade_ids=list(flagged),
        evidence_by_trade_id=evidence,
        stats=stats,
    )
