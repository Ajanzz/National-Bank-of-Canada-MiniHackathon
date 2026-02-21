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
from app.ml.model import BiasMLClassifier

# Mapping from internal BiasCard bias_type to the ML class label
_BIAS_TO_ML_CLASS: dict[str, str] = {
    "Overtrading": "overtrading",
    "Loss Aversion": "loss_aversion",
    "Revenge Trading": "revenge_trading",
    # "Recency Bias" has no ML equivalent — stays rules-only
}

_ML_WINDOW_SIZE = 50
_ML_STRIDE = 25


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
    
    # Basic statistics
    total_trades = len(trades)
    wins = [t for t in trades if t.is_win]
    losses = [t for t in trades if not t.is_win]
    
    win_rate = len(wins) / total_trades if total_trades > 0 else 0
    total_pnl = sum(t.profit_loss for t in trades)
    avg_pnl = mean([t.profit_loss for t in trades])
    pnls = [t.profit_loss for t in trades]
    
    if trades:
        first_balance = trades[0].balance - trades[0].profit_loss
        last_balance = trades[-1].balance
    else:
        first_balance = 0
        last_balance = 0
    
    # Calculate drawdown: maximum peak-to-trough decline
    max_balance = 0
    max_drawdown = 0
    for trade in trades:
        if trade.balance > max_balance:
            max_balance = trade.balance
        drawdown = max_balance - trade.balance
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    # Calculate volatility: standard deviation of PnL
    if len(pnls) > 1:
        mean_pnl = mean(pnls)
        variance = sum((p - mean_pnl) ** 2 for p in pnls) / len(pnls)
        volatility = variance ** 0.5
    else:
        volatility = 0.0
    
    # Date range
    if trades:
        date_start = trades[0].timestamp.strftime("%Y-%m-%d")
        date_end = trades[-1].timestamp.strftime("%Y-%m-%d")
        date_range = f"{date_start} to {date_end}"
    else:
        date_range = ""
    
    summary = {
        "total_trades": total_trades,
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": round(win_rate * 100, 2),
        "total_pnl": round(total_pnl, 2),
        "avg_pnl_per_trade": round(avg_pnl, 2),
        "starting_balance": round(first_balance, 2),
        "ending_balance": round(last_balance, 2),
        "max_drawdown": round(max_drawdown, 2),
        "volatility": round(volatility, 2),
        "date_range": date_range,
    }
    
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
            ml_model = BiasMLClassifier()
            if ml_model.is_loaded:
                window_features = extract_features_from_trades(
                    trades,
                    window_size=_ML_WINDOW_SIZE,
                    stride=_ML_STRIDE,
                )
                window_preds = [ml_model.predict_bias_probabilities(fv) for fv in window_features]
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
            key_stats={
                **overtrading_result.stats,
                **({
                    "ml_score": round(ml_probabilities.get("overtrading", 0.0) * 100, 1),
                    "ml_active": True,
                } if ml_active else {}),
            },
        ),
        BiasCard(
            bias_type="Loss Aversion",
            score=_blend_score(loss_aversion_result.score, "loss_aversion"),
            severity=_severity(_blend_score(loss_aversion_result.score, "loss_aversion")),
            flagged_trade_count=len(loss_aversion_result.flagged_trade_ids),
            key_stats={
                **loss_aversion_result.stats,
                **({
                    "ml_score": round(ml_probabilities.get("loss_aversion", 0.0) * 100, 1),
                    "ml_active": True,
                } if ml_active else {}),
            },
        ),
        BiasCard(
            bias_type="Revenge Trading",
            score=_blend_score(revenge_result.score, "revenge_trading"),
            severity=_severity(_blend_score(revenge_result.score, "revenge_trading")),
            flagged_trade_count=len(revenge_result.flagged_trade_ids),
            key_stats={
                **revenge_result.stats,
                **({
                    "ml_score": round(ml_probabilities.get("revenge_trading", 0.0) * 100, 1),
                    "ml_active": True,
                } if ml_active else {}),
            },
        ),
        BiasCard(
            # Recency bias has no ML equivalent — rules-only regardless of mode
            bias_type="Recency Bias",
            score=recency_result.score,
            severity=recency_result.severity,
            flagged_trade_count=len(recency_result.flagged_trade_ids),
            key_stats=recency_result.stats,
        ),
    ]
    
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
            # Compute confidence as average of detector scores
            detector_scores = []
            if trade_id in overtrading_result.flagged_trade_ids:
                detector_scores.append(overtrading_result.score)
            if trade_id in loss_aversion_result.flagged_trade_ids:
                detector_scores.append(loss_aversion_result.score)
            if trade_id in revenge_result.flagged_trade_ids:
                detector_scores.append(revenge_result.score)
            if trade_id in recency_result.flagged_trade_ids:
                detector_scores.append(recency_result.score)
            
            confidence = int(mean(detector_scores)) if detector_scores else 50
            
            # Severity: if any is high, result is high
            severity = "low"
            if any(b in all_flags[trade_id] for b in ["Revenge Trading"]):
                if any(bias_card.severity == "high" for bias_card in bias_cards if bias_card.bias_type == "Revenge Trading"):
                    severity = "high"
            
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
    
    # Equity curve
    equity_curve = [
        EquityCurvePoint(timestamp=trade.timestamp, balance=trade.balance)
        for trade in trades
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
        ml_probabilities={k: round(v, 4) for k, v in ml_probabilities.items()},
    )
