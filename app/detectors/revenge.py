"""Revenge trading bias detector."""

from app.models.schemas import DetectionResult, NormalizedTrade
from app.utils.time_utils import minutes_between
from app.utils.stats_utils import mean


def detect_revenge_trading(trades: list[NormalizedTrade]) -> DetectionResult:
    """
    Detect revenge trading bias:
    - If loss then next trade within 15m and size_usd >= 1.25x prior size_usd => flag
    - Loss streak length>=2 then next trade size escalates >=1.1x avg streak size => flag
    """
    if len(trades) < 2:
        return DetectionResult(score=0, severity="low", flagged_trade_ids=[], evidence_by_trade_id={}, stats={})
    
    flagged = set()
    evidence = {}
    
    # Baseline size (median)
    sizes = [t.size_usd for t in trades if t.size_usd > 0]
    baseline_size = sorted(sizes)[len(sizes) // 2] if sizes else 0
    
    # Detect loss streaks
    loss_streaks = []
    current_streak = []
    for trade in trades:
        if not trade.is_win:
            current_streak.append(trade)
        else:
            if len(current_streak) >= 2:
                loss_streaks.append(current_streak)
            current_streak = []
    if len(current_streak) >= 2:
        loss_streaks.append(current_streak)
    
    # Check for revenge trading patterns
    for i, trade in enumerate(trades[:-1]):
        if not trade.is_win:  # After a loss
            next_trade = trades[i + 1]
            time_gap = minutes_between(trade.timestamp, next_trade.timestamp)
            size_escalation = next_trade.size_usd / trade.size_usd if trade.size_usd > 0 else 0
            
            # Pattern 1: Re-entry within 15m with size escalation
            if time_gap <= 15 and size_escalation >= 1.25:
                flagged.add(next_trade.trade_id)
                evidence[next_trade.trade_id] = f"Re-entry {int(time_gap)}s after loss - size {size_escalation:.2f}x prior"
    
    # Check loss streak + size escalation
    for streak in loss_streaks:
        streak_idx_in_trades = trades.index(streak[-1]) if hasattr(trades, 'index') else None
        if streak_idx_in_trades is None:
            # Find by trade_id
            for j, t in enumerate(trades):
                if t.trade_id == streak[-1].trade_id:
                    streak_idx_in_trades = j
                    break
        
        if streak_idx_in_trades is not None and streak_idx_in_trades + 1 < len(trades):
            next_trade = trades[streak_idx_in_trades + 1]
            avg_streak_size = mean([t.size_usd for t in streak])
            size_escalation = next_trade.size_usd / avg_streak_size if avg_streak_size > 0 else 0
            
            if size_escalation >= 1.1:
                flagged.add(next_trade.trade_id)
                evidence[next_trade.trade_id] = f"{len(streak)}-loss streak then {size_escalation:.2f}x size escalation"
    
    # Score
    flagged_count = len(flagged)
    total_trades = len(trades)
    base_score = int((flagged_count / total_trades) * 100) if total_trades > 0 else 0
    
    # Boost for streak patterns
    streak_boost = min(20, len(loss_streaks) * 5)
    final_score = min(100, base_score + streak_boost)
    
    severity = "low" if final_score < 30 else "med" if final_score < 70 else "high"
    
    stats = {
        "flagged_count": flagged_count,
        "total_trades": total_trades,
        "loss_streak_count": len(loss_streaks),
        "baseline_size_usd": round(baseline_size, 2),
    }
    
    return DetectionResult(
        score=final_score,
        severity=severity,
        flagged_trade_ids=list(flagged),
        evidence_by_trade_id=evidence,
        stats=stats,
    )
