"""Revenge trading bias detector – aligned with definition."""

from app.models.schemas import DetectionResult, NormalizedTrade
from app.utils.time_utils import minutes_between
from app.utils.stats_utils import mean, percentile


def detect_revenge_trading(trades: list[NormalizedTrade]) -> DetectionResult:
    """
    Detect revenge trading – impulsive attempts to "win back" losses:
    1. Opening a larger trade immediately after a single loss (within 15 min, >= 1.25x size)
    2. Increased risk-taking after a negative P/L streak (>= 2 losses, next trade >= 1.1x size)
    3. Aggressive size escalation after a large loss (top 25% by abs_pnl, next trade >= 1.3x size)
    """
    if len(trades) < 2:
        return DetectionResult(score=0, severity="low", flagged_trade_ids=[], evidence_by_trade_id={}, stats={})

    flagged: set[str] = set()
    evidence: dict[str, str] = {}
    total = len(trades)

    def add_evidence(tid: str, msg: str) -> None:
        existing = evidence.get(tid, "")
        evidence[tid] = f"{existing}+{msg}" if existing else msg
        flagged.add(tid)

    # Precompute index map for O(1) streak lookups
    trade_idx: dict[str, int] = {t.trade_id: i for i, t in enumerate(trades)}

    # Large-loss threshold: top 25% of losses by abs_pnl
    loss_abs_pnls = [t.abs_pnl for t in trades if not t.is_win]
    large_loss_threshold = percentile(loss_abs_pnls, 75) if loss_abs_pnls else float("inf")

    # ── Signal 1: Immediate re-entry after a single loss ──────────────────────
    # Trader immediately opens a bigger position trying to recoup the loss
    TIMING_WINDOW_MIN = 15
    for i, trade in enumerate(trades):
        if not trade.is_win and i + 1 < total:
            next_t = trades[i + 1]
            gap = minutes_between(trade.timestamp, next_t.timestamp)
            size_mult = next_t.size_usd / trade.size_usd if trade.size_usd > 0 else 0
            if gap <= TIMING_WINDOW_MIN and size_mult >= 1.25:
                add_evidence(next_t.trade_id, f"fast_reentry_{int(gap)}m_{size_mult:.1f}x_size")

    # ── Signal 2: Increased risk-taking following a negative P/L streak ───────
    # After absorbing 2+ consecutive losses, trader escalates position size
    STREAK_MIN = 2
    SIZE_ESCALATION_STREAK = 1.1
    current_streak: list[NormalizedTrade] = []
    loss_streaks: list[list[NormalizedTrade]] = []

    for trade in trades:
        if not trade.is_win:
            current_streak.append(trade)
        else:
            if len(current_streak) >= STREAK_MIN:
                loss_streaks.append(current_streak[:])
            current_streak = []
    if len(current_streak) >= STREAK_MIN:
        loss_streaks.append(current_streak)

    for streak in loss_streaks:
        last_idx = trade_idx.get(streak[-1].trade_id)
        if last_idx is not None and last_idx + 1 < total:
            next_t = trades[last_idx + 1]
            avg_streak_size = mean([t.size_usd for t in streak])
            streak_total_loss = sum(t.abs_pnl for t in streak)
            size_mult = next_t.size_usd / avg_streak_size if avg_streak_size > 0 else 0
            if size_mult >= SIZE_ESCALATION_STREAK:
                add_evidence(
                    next_t.trade_id,
                    f"{len(streak)}-loss_streak_{size_mult:.1f}x_size(lost_{round(streak_total_loss, 2)})",
                )

    # ── Signal 3: Aggressive escalation after a large loss ────────────────────
    # After a single outsized loss, trader goes even bigger on the very next trade
    SIZE_ESCALATION_BIG_LOSS = 1.3
    for i, trade in enumerate(trades):
        if not trade.is_win and trade.abs_pnl >= large_loss_threshold and i + 1 < total:
            next_t = trades[i + 1]
            size_mult = next_t.size_usd / trade.size_usd if trade.size_usd > 0 else 0
            if size_mult >= SIZE_ESCALATION_BIG_LOSS:
                add_evidence(next_t.trade_id, f"post_big_loss_{size_mult:.1f}x_size")

    # ── Score ──────────────────────────────────────────────────────────────────
    flagged_count = len(flagged)
    base_score = int((flagged_count / total) * 80) if total > 0 else 0
    streak_bonus = min(20, len(loss_streaks) * 4)
    final_score = min(100, base_score + streak_bonus)
    severity = "low" if final_score < 30 else "med" if final_score < 70 else "high"

    stats = {
        "total_trades": total,
        "flagged_count": flagged_count,
        "loss_streak_count": len(loss_streaks),
        "large_loss_threshold": round(large_loss_threshold, 2),
        "timing_window_min": TIMING_WINDOW_MIN,
        "streak_min_length": STREAK_MIN,
        "size_escalation_after_streak": SIZE_ESCALATION_STREAK,
        "size_escalation_after_big_loss": SIZE_ESCALATION_BIG_LOSS,
    }

    return DetectionResult(
        score=final_score,
        severity=severity,
        flagged_trade_ids=list(flagged),
        evidence_by_trade_id=evidence,
        stats=stats,
    )
