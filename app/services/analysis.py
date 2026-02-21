"""Core analysis service orchestrating all detectors."""

from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np

from app.core.logging import logger
from app.models.schemas import (
    AnalysisResult,
    BiasCard,
    EquityCurvePoint,
    EventData,
    FlaggedTradeDetail,
    NormalizedTrade,
)
from app.detectors.overtrading import detect_overtrading
from app.detectors.loss_aversion import detect_loss_aversion
from app.detectors.revenge import detect_revenge_trading
from app.detectors.recency import detect_recency_bias
from app.events.provider import StubEventsProvider
from app.events.shock_generator import generate_market_shocks_from_trades
from app.utils.stats_utils import mean, percentile
from app.ml.features import extract_features_from_trades
from app.ml.model import get_classifier

# Mapping from internal BiasCard bias_type to the ML class label
_BIAS_TO_ML_CLASS: dict[str, str] = {
    "Overtrading": "overtrading",
    "Loss Aversion": "loss_aversion",
    "Revenge Trading": "revenge_trading",
    # "Recency Bias" has no ML equivalent — stays rules-only
}

_ML_WINDOW_SIZE = 50
_ML_STRIDE = 25


def _sanitize(obj):
    """Recursively replace nan/inf floats with 0 so JSON serialization never fails."""
    if isinstance(obj, float):
        return 0.0 if not np.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def compute_danger_hours_heatmap(trades: list[NormalizedTrade]) -> list[list[int]]:
    """
    Compute 7x24 heatmap (weekday x hour).
    Values are trade counts per bin.
    """
    # Initialize 7x24 matrix
    heatmap = [[0] * 24 for _ in range(7)]
    
    # Populate with trade counts
    for trade in trades:
        heatmap[trade.weekday][trade.hour] += 1
    
    return heatmap


def analyze_trades(
    trades: list[NormalizedTrade],
    scoring_mode: str = "hybrid",
) -> AnalysisResult:
    """
    Run complete analysis on normalized trades.

    Args:
        trades: Normalized trade list.
        scoring_mode: "hybrid"   — 75 % rules + 25 % XGBoost ML (default)
                      "rules"    — 100 % rule-based detectors only

    Returns AnalysisResult with:
    - summary: totals, win_rate, pnl, etc.
    - bias_cards: list of detector results (blended scores in hybrid mode)
    - flagged_trades: trades with flags and evidence
    - heatmap: 7x24 danger hours matrix
    - equity_curve: balance over time
    - optional_events: events from provider + shocks
    - ml_active: whether the XGBoost model contributed to scores
    - ml_probabilities: raw per-class ML probabilities (empty if rules-only)
    """
    if not trades:
        return AnalysisResult(
            summary={},
            bias_cards=[],
            flagged_trades=[],
            heatmap=[[0] * 24 for _ in range(7)],
            equity_curve=[],
            optional_events=[],
            ml_active=False,
            ml_probabilities={},
        )
    
    # Basic statistics — vectorized with numpy
    total_trades = len(trades)
    pnls = np.fromiter((t.profit_loss for t in trades), dtype=np.float64, count=total_trades)
    wins_mask = pnls > 0
    n_wins = int(wins_mask.sum())
    n_losses = total_trades - n_wins

    win_rate = n_wins / total_trades if total_trades > 0 else 0.0
    total_pnl = float(pnls.sum())
    avg_pnl = float(pnls.mean()) if total_trades > 0 else 0.0

    if trades:
        first_balance = trades[0].balance - trades[0].profit_loss
        last_balance = trades[-1].balance
    else:
        first_balance = 0.0
        last_balance = 0.0

    # Drawdown: maximum peak-to-trough decline (vectorized)
    balances = np.fromiter((t.balance for t in trades), dtype=np.float64, count=total_trades)
    running_max = np.maximum.accumulate(balances)
    max_drawdown = float((running_max - balances).max()) if total_trades > 0 else 0.0

    # Volatility: std dev of PnL
    volatility = float(pnls.std()) if total_trades > 1 else 0.0
    
    # Date range
    if trades:
        date_start = trades[0].timestamp.strftime("%Y-%m-%d")
        date_end = trades[-1].timestamp.strftime("%Y-%m-%d")
        date_range = f"{date_start} to {date_end}"
    else:
        date_range = ""
    
    summary = _sanitize({
        "total_trades": total_trades,
        "winning_trades": n_wins,
        "losing_trades": n_losses,
        "win_rate": round(win_rate * 100, 2),
        "total_pnl": round(total_pnl, 2),
        "avg_pnl_per_trade": round(avg_pnl, 2),
        "starting_balance": round(first_balance, 2),
        "ending_balance": round(last_balance, 2),
        "max_drawdown": round(max_drawdown, 2),
        "volatility": round(volatility, 2),
        "date_range": date_range,
    })
    
    # Run detectors
    overtrading_result = detect_overtrading(trades)
    loss_aversion_result = detect_loss_aversion(trades)
    revenge_result = detect_revenge_trading(trades)
    recency_result = detect_recency_bias(trades)

    # ── ML inference (hybrid mode) ──────────────────────────────────────
    ml_active = False
    ml_probabilities: dict[str, float] = {}

    if scoring_mode == "hybrid":
        try:
            ml_model = get_classifier()
            if ml_model.is_loaded:
                window_features = extract_features_from_trades(
                    trades,
                    window_size=_ML_WINDOW_SIZE,
                    stride=_ML_STRIDE,
                )
                # Batch predict: one predict_proba call for all windows
                window_preds = ml_model.predict_batch(window_features)
                # Average probabilities across all windows
                ml_classes = ml_model.classes
                ml_probabilities = {
                    cls: float(np.mean([p.get(cls, 0.0) for p in window_preds]))
                    for cls in ml_classes
                }
                ml_active = True
                logger.info(
                    f"ML inference active ({len(window_preds)} windows): {ml_probabilities}"
                )
            else:
                logger.info("No trained ML model found — using rules-only scoring.")
        except Exception as exc:
            logger.warning(f"ML inference failed, falling back to rules-only: {exc}")

    def _blend_score(rule_score: int, ml_class: str | None) -> int:
        """Return 75 % rule + 25 % ML, or pure rule if ML not available."""
        if not ml_active or ml_class is None:
            return rule_score
        ml_score = ml_probabilities.get(ml_class, 0.0) * 100
        return int(round(0.75 * rule_score + 0.25 * ml_score))

    def _severity(score: int) -> str:
        if score >= 70:
            return "high"
        if score >= 40:
            return "med"
        return "low"

    # ── Create bias cards (with blended scores in hybrid mode) ──────────
    bias_cards = [
        BiasCard(
            bias_type="Overtrading",
            score=_blend_score(overtrading_result.score, "overtrading"),
            severity=_severity(_blend_score(overtrading_result.score, "overtrading")),
            flagged_trade_count=len(overtrading_result.flagged_trade_ids),
            key_stats=_sanitize({
                **overtrading_result.stats,
                **({
                    "ml_score": round(ml_probabilities.get("overtrading", 0.0) * 100, 1),
                    "ml_active": True,
                } if ml_active else {}),
            }),
        ),
        BiasCard(
            bias_type="Loss Aversion",
            score=_blend_score(loss_aversion_result.score, "loss_aversion"),
            severity=_severity(_blend_score(loss_aversion_result.score, "loss_aversion")),
            flagged_trade_count=len(loss_aversion_result.flagged_trade_ids),
            key_stats=_sanitize({
                **loss_aversion_result.stats,
                **({
                    "ml_score": round(ml_probabilities.get("loss_aversion", 0.0) * 100, 1),
                    "ml_active": True,
                } if ml_active else {}),
            }),
        ),
        BiasCard(
            bias_type="Revenge Trading",
            score=_blend_score(revenge_result.score, "revenge_trading"),
            severity=_severity(_blend_score(revenge_result.score, "revenge_trading")),
            flagged_trade_count=len(revenge_result.flagged_trade_ids),
            key_stats=_sanitize({
                **revenge_result.stats,
                **({
                    "ml_score": round(ml_probabilities.get("revenge_trading", 0.0) * 100, 1),
                    "ml_active": True,
                } if ml_active else {}),
            }),
        ),
        BiasCard(
            # Recency bias has no ML equivalent — rules-only regardless of mode
            bias_type="Recency Bias",
            score=recency_result.score,
            severity=recency_result.severity,
            flagged_trade_count=len(recency_result.flagged_trade_ids),
            key_stats=_sanitize(recency_result.stats),
        ),
    ]
    
    # Pre-convert flagged_trade_ids lists → sets for O(1) membership tests
    ot_flagged = set(overtrading_result.flagged_trade_ids)
    la_flagged = set(loss_aversion_result.flagged_trade_ids)
    rv_flagged = set(revenge_result.flagged_trade_ids)
    rc_flagged = set(recency_result.flagged_trade_ids)

    # Aggregate all flagged trades by trade_id
    all_flags = defaultdict(list)
    all_evidence = defaultdict(dict)
    
    for detector_name, result in [
        ("Overtrading", overtrading_result),
        ("Loss Aversion", loss_aversion_result),
        ("Revenge Trading", revenge_result),
        ("Recency Bias", recency_result),
    ]:
        for trade_id in result.flagged_trade_ids:
            all_flags[trade_id].append(detector_name)
            all_evidence[trade_id][detector_name] = result.evidence_by_trade_id.get(trade_id, "")
    
    # Create flagged trade details
    trade_map = {t.trade_id: t for t in trades}
    flagged_trades = []
    
    for trade_id, biases in all_flags.items():
        if trade_id in trade_map:
            trade = trade_map[trade_id]
            # Compute confidence as average of detector scores (O(1) set lookups)
            detector_scores = []
            if trade_id in ot_flagged:
                detector_scores.append(overtrading_result.score)
            if trade_id in la_flagged:
                detector_scores.append(loss_aversion_result.score)
            if trade_id in rv_flagged:
                detector_scores.append(revenge_result.score)
            if trade_id in rc_flagged:
                detector_scores.append(recency_result.score)
            
            confidence = int(mean(detector_scores)) if detector_scores else 50
            
            # Severity: high if revenge-flagged and revenge card is high
            severity = "low"
            if trade_id in rv_flagged:
                for bias_card in bias_cards:
                    if bias_card.bias_type == "Revenge Trading" and bias_card.severity == "high":
                        severity = "high"
                        break
            
            evidence_str = "; ".join(
                f"{bias}: {all_evidence[trade_id].get(bias, '')}"
                for bias in biases
            )
            
            flagged_trades.append(
                FlaggedTradeDetail(
                    trade_id=trade_id,
                    timestamp=trade.timestamp,
                    asset=trade.asset,
                    side=trade.side,
                    quantity=trade.quantity,
                    entry_price=trade.entry_price,
                    exit_price=trade.exit_price,
                    profit_loss=trade.profit_loss,
                    balance=trade.balance,
                    bias_type=", ".join(biases),
                    flag_severity=severity,
                    confidence=confidence,
                    evidence=evidence_str,
                )
            )
    
    # Heatmap
    heatmap = compute_danger_hours_heatmap(trades)
    
    # Equity curve — downsample to at most 500 points for large datasets
    _MAX_EQUITY_POINTS = 500
    equity_trades = trades
    if len(trades) > _MAX_EQUITY_POINTS:
        step = len(trades) // _MAX_EQUITY_POINTS
        equity_trades = trades[::step]
        # Always include the final trade
        if equity_trades[-1] is not trades[-1]:
            equity_trades = list(equity_trades) + [trades[-1]]
    equity_curve = [
        EquityCurvePoint(timestamp=trade.timestamp, balance=trade.balance)
        for trade in equity_trades
    ]
    
    # Events
    events_provider = StubEventsProvider()
    provider_events = events_provider.get_events(trades[0].timestamp, trades[-1].timestamp)
    market_shocks = generate_market_shocks_from_trades(trades)
    
    all_events = provider_events + market_shocks
    optional_events = [
        {
            "timestamp": e.timestamp.isoformat(),
            "event_type": e.event_type,
            "label": e.label,
            "symbols": e.symbols,
            "pnl_magnitude": e.pnl_magnitude,
        }
        for e in all_events
    ]
    
    # Compute behavior_index as average of bias scores (0-100)
    if bias_cards:
        scores = [card.score for card in bias_cards]
        behavior_index = sum(scores) / len(bias_cards)
        logger.info(f"Behavior index calculation: scores={scores}, sum={sum(scores)}, count={len(bias_cards)}, result={behavior_index}")
    else:
        behavior_index = 0.0
    
    return AnalysisResult(
        behavior_index=behavior_index,
        summary=summary,
        bias_cards=bias_cards,
        flagged_trades=flagged_trades,
        heatmap=heatmap,
        equity_curve=equity_curve,
        optional_events=optional_events,
        ml_active=ml_active,
        ml_probabilities=_sanitize({k: round(v, 4) for k, v in ml_probabilities.items()}),
    )