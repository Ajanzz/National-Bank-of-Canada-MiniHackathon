"""Loss aversion / disposition bias detector."""

import pandas as pd

from app.models.schemas import DetectionResult, NormalizedTrade
from app.utils.stats_utils import mean, percentile


def detect_loss_aversion(trades: list[NormalizedTrade]) -> DetectionResult:
    """
    Detect loss aversion / disposition bias:
    - Compare avg win return_pct vs avg loss return_pct magnitude
    - Frequency of small wins and large losses
    - If duration not available, document limitation
    """
    if not trades:
        return DetectionResult(score=0, severity="low", flagged_trade_ids=[], evidence_by_trade_id={}, stats={})
    
    flagged = set()
    evidence = {}
    
    wins = [t for t in trades if t.is_win]
    losses = [t for t in trades if not t.is_win]
    
    if not wins or not losses:
        return DetectionResult(score=0, severity="low", flagged_trade_ids=[], evidence_by_trade_id={}, stats={})
    
    # Win/loss analysis
    avg_win_return = mean([t.return_pct for t in wins])
    avg_loss_return = mean([t.return_pct for t in losses])  # negative
    
    # Magnitude analysis
    avg_loss_magnitude = mean([abs(t.return_pct) for t in losses])
    avg_win_magnitude = mean([abs(t.return_pct) for t in wins])
    
    win_count = len(wins)
    loss_count = len(losses)
    total = len(trades)
    
    # Flag trades: identify large losses and small wins
    small_win_threshold = percentile([t.profit_loss for t in wins], 25) if wins else 0
    large_loss_threshold = percentile([abs(t.profit_loss) for t in losses], 75) if losses else 0
    
    for trade in trades:
        if trade.is_win and trade.profit_loss <= small_win_threshold:
            flagged.add(trade.trade_id)
            evidence[trade.trade_id] = "Small win (bottom 25% magnitude)"
        elif not trade.is_win and abs(trade.profit_loss) >= large_loss_threshold:
            flagged.add(trade.trade_id)
            evidence[trade.trade_id] = "Large loss (top 25% magnitude)"
    
    # Score calculation
    # High loss magnitude + high loss frequency + small wins = high bias
    loss_ratio = loss_count / total
    magnitude_ratio = avg_loss_magnitude / (avg_win_magnitude + 0.001)
    win_size_ratio = small_win_threshold / (large_loss_threshold + 0.001)
    
    # Components
    score_val = 0
    score_val += int(loss_ratio * 30)  # Loss frequency component
    score_val += min(30, int(magnitude_ratio * 20))  # Magnitude ratio component
    score_val += min(30, int((1 - win_size_ratio) * 20))  # Win size vs loss size component
    score_val += int((len(flagged) / total) * 10)  # Flagged trade proportion
    
    final_score = min(100, score_val)
    severity = "low" if final_score < 30 else "med" if final_score < 70 else "high"
    
    stats = {
        "win_count": win_count,
        "loss_count": loss_count,
        "win_rate": round(win_count / total, 3),
        "avg_win_return_pct": round(avg_win_return * 100, 2),
        "avg_loss_return_pct": round(avg_loss_return * 100, 2),
        "avg_win_magnitude_pct": round(avg_win_magnitude * 100, 2),
        "avg_loss_magnitude_pct": round(avg_loss_magnitude * 100, 2),
        "duration_available": "false",
        "duration_not_available_note": "Hold times not inferred; pnl magnitude + frequency used instead",
    }
    
    return DetectionResult(
        score=final_score,
        severity=severity,
        flagged_trade_ids=list(flagged),
        evidence_by_trade_id=evidence,
        stats=stats,
    )
