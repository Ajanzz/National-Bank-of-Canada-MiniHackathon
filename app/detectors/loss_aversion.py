"""Loss aversion / disposition bias detector – aligned with definition."""

from app.models.schemas import DetectionResult, NormalizedTrade
from app.utils.stats_utils import mean, percentile


def detect_loss_aversion(trades: list[NormalizedTrade]) -> DetectionResult:
    """
    Detect loss aversion / disposition bias:
    1. Letting losing trades run too long (large loss magnitudes vs account balance)
    2. Closing winning trades too early (small win magnitudes relative to losses)
    3. Unbalanced risk/reward ratio (avg loss PnL > avg win PnL)
    4. Higher average loss size than average win size
    """
    if not trades:
        return DetectionResult(score=0, severity="low", flagged_trade_ids=[], evidence_by_trade_id={}, stats={})

    wins = [t for t in trades if t.is_win]
    losses = [t for t in trades if not t.is_win]

    if not wins or not losses:
        return DetectionResult(score=0, severity="low", flagged_trade_ids=[], evidence_by_trade_id={}, stats={})

    flagged: set[str] = set()
    evidence: dict[str, str] = {}
    total = len(trades)
    win_count = len(wins)
    loss_count = len(losses)

    # Core metrics
    avg_win_pnl = mean([t.abs_pnl for t in wins])
    avg_loss_pnl = mean([t.abs_pnl for t in losses])
    avg_win_return = mean([abs(t.return_pct) for t in wins])
    avg_loss_return = mean([abs(t.return_pct) for t in losses])

    # Risk/reward ratio: avg dollars lost per losing trade vs avg dollars gained per winning trade
    rr_ratio = avg_loss_pnl / avg_win_pnl if avg_win_pnl > 0 else 1.0

    # ── Signal 1: Letting losses run – losses in the worst 33% by magnitude ───
    # Traders with loss aversion hold losers hoping they recover, leading to large losses
    large_loss_threshold = percentile([t.abs_pnl for t in losses], 67)
    for t in losses:
        if t.abs_pnl >= large_loss_threshold:
            flagged.add(t.trade_id)
            evidence[t.trade_id] = f"large_loss(pnl={round(t.abs_pnl, 2)}, rr_ratio={round(rr_ratio, 2)})"

    # ── Signal 2: Closing wins too early – wins in the bottom 33% by magnitude ─
    # Traders exit winning trades prematurely out of fear of reversal
    small_win_threshold = percentile([t.abs_pnl for t in wins], 33)
    for t in wins:
        if t.abs_pnl <= small_win_threshold:
            flagged.add(t.trade_id)
            evidence[t.trade_id] = f"small_win(pnl={round(t.abs_pnl, 2)})"

    # ── Score ──────────────────────────────────────────────────────────────────
    score = 0

    # Component 1: avg loss PnL > avg win PnL (unbalanced risk/reward) – up to 35 pts
    if avg_loss_pnl > avg_win_pnl:
        loss_win_ratio = min(2.5, avg_loss_pnl / avg_win_pnl)
        score += int((loss_win_ratio - 1.0) / 1.5 * 35)

    # Component 2: avg loss return_pct > avg win return_pct – up to 25 pts
    if avg_loss_return > avg_win_return:
        return_ratio = min(2.5, avg_loss_return / avg_win_return)
        score += int((return_ratio - 1.0) / 1.5 * 25)

    # Component 3: proportion of large losses in dataset – up to 25 pts
    large_loss_frac = sum(1 for t in losses if t.abs_pnl >= large_loss_threshold) / total
    score += int(large_loss_frac * 25)

    # Component 4: proportion of small wins in dataset – up to 15 pts
    small_win_frac = sum(1 for t in wins if t.abs_pnl <= small_win_threshold) / total
    score += int(small_win_frac * 15)

    final_score = min(100, score)
    severity = "low" if final_score < 30 else "med" if final_score < 70 else "high"

    stats = {
        "win_count": win_count,
        "loss_count": loss_count,
        "win_rate": round(win_count / total, 3),
        "avg_win_pnl": round(avg_win_pnl, 2),
        "avg_loss_pnl": round(avg_loss_pnl, 2),
        "avg_win_return_pct": round(avg_win_return * 100, 2),
        "avg_loss_return_pct": round(avg_loss_return * 100, 2),
        "risk_reward_ratio": round(rr_ratio, 2),
        "large_loss_threshold": round(large_loss_threshold, 2),
        "small_win_threshold": round(small_win_threshold, 2),
        "flagged_count": len(flagged),
    }

    return DetectionResult(
        score=final_score,
        severity=severity,
        flagged_trade_ids=list(flagged),
        evidence_by_trade_id=evidence,
        stats=stats,
    )
