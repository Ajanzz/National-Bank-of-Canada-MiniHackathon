"""FastAPI routes for the backend API."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal, Optional

from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field, ValidationError

from app.core.logging import logger
from app.models.schemas import (
    AnalyzeRequest,
    AnalysisResult,
    EventsResponse,
    HealthResponse,
    ImportRequest,
    NewsResponse,
    NewsItem,
    SimulateRequest,
    SimulationResult,
    Trade,
    PracticeQuestionsRequest,
    PracticeQuestionsResponse,
)
from app.services.analysis import analyze_trades, _sanitize as _sanitize_response
from app.services.ingest import ingest_csv
from app.services.simulator import simulate_with_rules
from app.events.shock_generator import generate_market_shocks_from_trades
from app.events.provider import StubEventsProvider
from app.utils.csv_utils import parse_base64_csv, parse_csv

# In-memory storage for imported datasets
IMPORTED_DATASETS = {}
# In-memory emotional check-in storage keyed by session_id -> bias_type
EMOTIONAL_CHECKIN_STORE: dict[str, dict[str, dict[str, Any]]] = {}

FRONTEND_BIAS_TO_CANONICAL = {
    "overtrading": "OVERTRADING",
    "loss_aversion": "LOSS_AVERSION",
    "revenge_trading": "REVENGE_TRADING",
    "recency_bias": "RECENCY_BIAS",
}


def _normalize_scoring_mode(raw_mode: Optional[str]) -> str:
    """Normalize scoring mode from frontend aliases to backend-supported values."""
    scoring_mode = (raw_mode or "hybrid").lower()
    if scoring_mode == "rule_only":
        scoring_mode = "rules"
    if scoring_mode not in ("hybrid", "rules"):
        scoring_mode = "hybrid"
    return scoring_mode


def _build_frontend_analysis_response(normalized_trades: list[Any], result: Any, scoring_mode: str) -> dict[str, Any]:
    """Transform internal analysis output into frontend-compatible payload."""
    bias_name_map = {
        "Overtrading": "overtrading",
        "Loss Aversion": "loss_aversion",
        "Revenge Trading": "revenge_trading",
        "Recency Bias": "recency_bias",
    }

    sorted_trades = sorted(normalized_trades, key=lambda t: t.timestamp)
    trade_timeline = [
        {
            "timestamp": trade.timestamp.isoformat(),
            "pnl": trade.profit_loss,
        }
        for trade in sorted_trades
    ]

    daily_pnl_totals: dict[str, float] = {}
    for trade in sorted_trades:
        date_key = trade.timestamp.date().isoformat()
        daily_pnl_totals[date_key] = daily_pnl_totals.get(date_key, 0.0) + trade.profit_loss

    daily_pnl = [
        {"date": date_key, "pnl": round(pnl, 2)}
        for date_key, pnl in sorted(daily_pnl_totals.items())
    ]

    frontend_response: dict[str, Any] = {
        "behavior_index": result.behavior_index,
        "bias_scores": {},
        "top_triggers": [],
        "danger_hours": result.heatmap,
        "daily_pnl": daily_pnl,
        "trade_timeline": trade_timeline,
        "flagged_trades": [],
        "explainability": {},
        "summary": result.summary,
        "ml_active": result.ml_active,
        "ml_probabilities": result.ml_probabilities,
        "scoring_mode": scoring_mode,
    }

    for card in result.bias_cards:
        frontend_key = bias_name_map.get(card.bias_type, card.bias_type.lower().replace(" ", "_"))
        frontend_response["bias_scores"][frontend_key] = card.score
        if card.score > 50:
            frontend_response["top_triggers"].append(card.bias_type)

    for ft in result.flagged_trades:
        bias_list = [b.strip() for b in ft.bias_type.split(",")]
        bias_keys = [bias_name_map.get(b, b.lower().replace(" ", "_")) for b in bias_list]

        frontend_response["flagged_trades"].append({
            "trade_id": ft.trade_id,
            "timestamp": ft.timestamp.isoformat() if ft.timestamp else None,
            "symbol": ft.asset,
            "side": ft.side,
            "quantity": ft.quantity,
            "entry_price": ft.entry_price,
            "exit_price": ft.exit_price,
            "profit_loss": ft.profit_loss,
            "balance": ft.balance,
            "bias": bias_keys[0] if bias_keys else "unknown",
            "bias_types": bias_keys,
            "confidence": ft.confidence,
            "flag_severity": ft.flag_severity,
            "evidence": ft.evidence.split("; ") if ft.evidence and ft.evidence.strip() else [],
        })

    return _sanitize_response(frontend_response)


def _get_or_compute_dataset_analysis(dataset_id: str, scoring_mode: str) -> tuple[dict[str, Any], bool]:
    """Return cached analysis payload for dataset+mode; compute once on cache miss."""
    dataset_entry = IMPORTED_DATASETS.get(dataset_id)
    if dataset_entry is None:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    analysis_cache = dataset_entry.setdefault("analysis_cache", {})
    cached = analysis_cache.get(scoring_mode)
    if cached is not None:
        return cached, True

    trades = dataset_entry.get("trades", [])
    if not trades and "frontend_analysis_payload" in dataset_entry:
        # Backward compatibility for older in-memory entries.
        payload = dataset_entry["frontend_analysis_payload"]
        analysis_cache[scoring_mode] = payload
        return payload, False

    result = analyze_trades(trades, scoring_mode=scoring_mode)
    payload = _build_frontend_analysis_response(trades, result, scoring_mode)
    analysis_cache[scoring_mode] = payload
    return payload, False

class ImportDatasetRequest(BaseModel):
    """Frontend CSV import request with mapping."""
    raw_csv: Optional[str] = None
    rows: Optional[list[dict[str, Any]]] = None
    mapping: Optional[dict[str, Any]] = None
    timezone: str = "UTC"
    session_template: str = "equities_rth"
    
    class Config:
        # Allow extra fields and coerce values
        extra = "allow"

class ImportDatasetResponse(BaseModel):
    """Response after importing CSV."""
    dataset_id: str
    stats: dict

router = APIRouter(prefix="/api/v1", tags=["trading"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="ok", version="1.0.0")


def _ingest_and_store_dataset(
    raw_csv: str,
    mapping: Optional[dict[str, Any]],
    timezone: str,
    session_template: str,
    dataset_id_override: Optional[str] = None,
) -> ImportDatasetResponse:
    """Parse CSV, normalize trades, and precompute legacy analysis payloads."""
    dataframe = parse_csv(raw_csv)
    normalized_trades, ingest_errors = ingest_csv(dataframe, mapping=mapping)
    if not normalized_trades:
        detail = "; ".join(ingest_errors[:8]) if ingest_errors else "No valid trades parsed from CSV."
        raise HTTPException(status_code=400, detail=detail)

    dataset_id = dataset_id_override or f"ds-{uuid.uuid4().hex[:8]}"
    # Precompute both modes so switching modes does not trigger long recomputation.
    hybrid_result = analyze_trades(normalized_trades, scoring_mode="hybrid")
    hybrid_payload = _build_frontend_analysis_response(normalized_trades, hybrid_result, "hybrid")
    rules_result = analyze_trades(normalized_trades, scoring_mode="rules")
    rules_payload = _build_frontend_analysis_response(normalized_trades, rules_result, "rules")

    date_range = None
    if normalized_trades:
        date_range = {
            "start": normalized_trades[0].date,
            "end": normalized_trades[-1].date,
        }

    IMPORTED_DATASETS[dataset_id] = {
        "trades": normalized_trades,
        "timezone": timezone,
        "session_template": session_template,
        "analysis_cache": {
            "hybrid": hybrid_payload,
            "rules": rules_payload,
        },
    }

    stats = {
        "rows": len(normalized_trades),
        "detected_date_range": date_range,
    }
    if ingest_errors:
        stats["ingest_warnings"] = ingest_errors[:20]

    logger.info(
        "CSV import successful (legacy): dataset_id=%s rows=%s warnings=%s",
        dataset_id,
        len(normalized_trades),
        len(ingest_errors),
    )
    return ImportDatasetResponse(dataset_id=dataset_id, stats=stats)


@router.post("/import/csv", response_model=ImportDatasetResponse)
async def import_csv(request: ImportDatasetRequest = Body(...)):
    """
    Import CSV data with column mapping.
    
    Accepts raw CSV text and mapping to normalize columns.
    Returns a dataset_id for subsequent /analyze calls.
    """
    try:
        logger.info(f"Received import request: raw_csv len={len(request.raw_csv) if request.raw_csv else 0}, mapping={request.mapping}")
        
        if not request.raw_csv:
            raise HTTPException(status_code=400, detail="raw_csv is required")
        return _ingest_and_store_dataset(
            raw_csv=request.raw_csv,
            mapping=request.mapping,
            timezone=request.timezone,
            session_template=request.session_template,
        )
    
    except HTTPException:
        raise
    except ValidationError as ve:
        logger.error(f"Validation error: {ve.json()}")
        raise HTTPException(status_code=422, detail=f"Validation error: {str(ve)}")
    except Exception as e:
        logger.error(f"CSV import error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/import/csv-file", response_model=ImportDatasetResponse)
async def import_csv_file(
    file: UploadFile = File(...),
    timezone: str = Form("UTC"),
    session_template: str = Form("equities_rth"),
):
    """Import CSV via multipart file upload using legacy ingestion pipeline."""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="CSV file is required")
        if not file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="File must be a .csv")

        file.file.seek(0)
        raw_bytes = file.file.read()
        raw_csv = raw_bytes.decode("utf-8-sig")

        imported = _ingest_and_store_dataset(
            raw_csv=raw_csv,
            mapping=None,
            timezone=timezone,
            session_template=session_template,
        )
        logger.info(
            "Received CSV file import (legacy): filename=%s rows=%s",
            file.filename,
            imported.stats.get("rows"),
        )
        return imported
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("CSV file import error: %s", str(exc), exc_info=True)
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        await file.close()


@router.post("/analyze")
async def analyze(request: AnalyzeRequest):
    """
    Analyze trades from dataset_id, JSON list, or base64-encoded CSV.
    
    Supports either:
    - dataset_id (from a prior /import/csv call)
    - JSON list of trades
    - base64-encoded CSV content
    """
    try:
        logger.info(f"Analyze request: dataset_id={request.dataset_id}, scoring_mode={request.scoring_mode}, has_trades={bool(request.trades)}, has_base64_csv={bool(request.base64_csv)}")

        if request.dataset_id:
            if request.dataset_id not in IMPORTED_DATASETS:
                logger.error(f"Dataset not found: {request.dataset_id}, available: {list(IMPORTED_DATASETS.keys())}")
                raise HTTPException(status_code=404, detail=f"Dataset not found: {request.dataset_id}")

            scoring_mode = _normalize_scoring_mode(request.scoring_mode)
            cached_payload, cache_hit = _get_or_compute_dataset_analysis(request.dataset_id, scoring_mode)
            logger.info(
                "Analysis response for legacy dataset=%s mode=%s cache_hit=%s",
                request.dataset_id,
                scoring_mode,
                cache_hit,
            )
            return cached_payload

        if request.base64_csv:
            dataframe = parse_base64_csv(request.base64_csv)
            normalized_trades, ingest_errors = ingest_csv(dataframe, mapping=None)
            if not normalized_trades:
                detail = "; ".join(ingest_errors[:8]) if ingest_errors else "No valid trades parsed from base64 CSV."
                raise HTTPException(status_code=400, detail=detail)

            scoring_mode = _normalize_scoring_mode(request.scoring_mode)
            result = analyze_trades(normalized_trades, scoring_mode=scoring_mode)
            return _build_frontend_analysis_response(normalized_trades, result, scoring_mode)

        normalized_trades = []
        if request.trades:
            # Convert Trade objects to NormalizedTrade
            from app.utils.time_utils import (
                timestamp_to_date_string,
                timestamp_to_hour,
                timestamp_to_weekday,
            )
            from app.utils.stats_utils import compute_trade_id
            
            for trade in request.trades:
                trade_id = compute_trade_id(
                    trade.timestamp.isoformat(),
                    trade.asset,
                    trade.side,
                    trade.quantity,
                    trade.entry_price,
                    trade.exit_price,
                )
                
                is_win = trade.profit_loss > 0
                size_usd = trade.quantity * trade.entry_price
                return_pct = (trade.profit_loss / size_usd) if size_usd > 0 else 0
                
                from app.models.schemas import NormalizedTrade
                
                nt = NormalizedTrade(
                    trade_id=trade_id,
                    timestamp=trade.timestamp,
                    asset=trade.asset,
                    side=trade.side,
                    quantity=trade.quantity,
                    entry_price=trade.entry_price,
                    exit_price=trade.exit_price,
                    profit_loss=trade.profit_loss,
                    balance=trade.balance,
                    date=timestamp_to_date_string(trade.timestamp),
                    hour=timestamp_to_hour(trade.timestamp),
                    weekday=timestamp_to_weekday(trade.timestamp),
                    is_win=is_win,
                    abs_pnl=abs(trade.profit_loss),
                    size_usd=size_usd,
                    return_pct=return_pct,
                )
                normalized_trades.append(nt)
        else:
            raise HTTPException(
                status_code=400,
                detail="Must provide either 'dataset_id', 'trades', or 'base64_csv'"
            )

        scoring_mode = _normalize_scoring_mode(request.scoring_mode)

        result = analyze_trades(normalized_trades, scoring_mode=scoring_mode)
        frontend_response = _build_frontend_analysis_response(normalized_trades, result, scoring_mode)
        logger.info(f"Analysis complete: {len(normalized_trades)} trades, behavior_index={result.behavior_index}")
        return frontend_response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/simulate", response_model=SimulationResult)
async def simulate(request: SimulateRequest):
    """
    Run what-if simulation with trading rules.
    
    Rules:
    - cooldown_after_loss_minutes: Minutes to wait after loss
    - daily_trade_cap: Max trades per day
    - max_position_size_multiplier: Max size as multiple of baseline
    - stop_after_consecutive_losses: Stop after N losses in a row
    """
    try:
        # Prefer server-side dataset trades to avoid large client payloads.
        if request.dataset_id:
            if request.dataset_id not in IMPORTED_DATASETS:
                raise HTTPException(status_code=404, detail=f"Dataset not found: {request.dataset_id}")
            normalized_trades = IMPORTED_DATASETS[request.dataset_id]["trades"]
            if not normalized_trades:
                raise HTTPException(
                    status_code=400,
                    detail="Simulation requires row-level trades. Re-import with legacy pipeline if simulation is needed.",
                )
        else:
            if not request.trades:
                raise HTTPException(status_code=400, detail="Must provide either dataset_id or trades")

            # Convert trades to NormalizedTrade
            from app.utils.time_utils import (
                timestamp_to_date_string,
                timestamp_to_hour,
                timestamp_to_weekday,
            )
            from app.utils.stats_utils import compute_trade_id
            from app.models.schemas import NormalizedTrade

            normalized_trades = []
            for trade in request.trades:
                trade_id = compute_trade_id(
                    trade.timestamp.isoformat(),
                    trade.asset,
                    trade.side,
                    trade.quantity,
                    trade.entry_price,
                    trade.exit_price,
                )

                is_win = trade.profit_loss > 0
                size_usd = trade.quantity * trade.entry_price
                return_pct = (trade.profit_loss / size_usd) if size_usd > 0 else 0

                nt = NormalizedTrade(
                    trade_id=trade_id,
                    timestamp=trade.timestamp,
                    asset=trade.asset,
                    side=trade.side,
                    quantity=trade.quantity,
                    entry_price=trade.entry_price,
                    exit_price=trade.exit_price,
                    profit_loss=trade.profit_loss,
                    balance=trade.balance,
                    date=timestamp_to_date_string(trade.timestamp),
                    hour=timestamp_to_hour(trade.timestamp),
                    weekday=timestamp_to_weekday(trade.timestamp),
                    is_win=is_win,
                    abs_pnl=abs(trade.profit_loss),
                    size_usd=size_usd,
                    return_pct=return_pct,
                )
                normalized_trades.append(nt)
        
        # Run simulation
        result = simulate_with_rules(normalized_trades, request.rules)
        logger.info(f"Simulation complete: {result.blocked_trades.__len__()} trades blocked")
        return result
    
    except Exception as e:
        logger.error(f"Simulation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/events", response_model=EventsResponse)
async def get_events(start: str, end: str):
    """
    Get events for date range (stub provider + market shocks).
    
    Query params:
    - start: YYYY-MM-DD
    - end: YYYY-MM-DD
    """
    try:
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        
        # Use stub provider
        provider = StubEventsProvider()
        events = provider.get_events(start_dt, end_dt)
        
        logger.info(f"Events retrieved for {start} to {end}: {len(events)} events")
        
        from app.models.schemas import EventsResponse, EventData
        return EventsResponse(events=events)
    
    except Exception as e:
        logger.error(f"Events error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/news", response_model=NewsResponse)
async def get_news(start: str, end: str):
    """
    Get news for date range (stub, no external API).
    
    Schema defined but no external integration.
    Returns empty list. Ready for future integration.
    """
    try:
        # Stub implementation
        news_items = []
        
        logger.info(f"News requested for {start} to {end}: returning empty (stub)")
        
        return NewsResponse(news=news_items)
    
    except Exception as e:
        logger.error(f"News error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/practice/questions", response_model=PracticeQuestionsResponse)
async def generate_practice_questions(request: PracticeQuestionsRequest):
    """
    Generate adaptive practice questions for the trader's top 2 biases.
    
    Returns 4 questions (2 per bias) with scenarios, multiple choice, 
    correct answers, and explanations.
    """
    try:
        logger.info("Practice questions request: dataset_id=%s scoring_mode=%s", request.dataset_id, request.scoring_mode)
        
        if not request.dataset_id:
            raise HTTPException(status_code=400, detail="dataset_id is required")

        scoring_mode = _normalize_scoring_mode(request.scoring_mode)
        cached_analysis, cache_hit = _get_or_compute_dataset_analysis(request.dataset_id, scoring_mode)
        trades = IMPORTED_DATASETS[request.dataset_id]["trades"]
        logger.info(
            "Loaded analysis for practice questions dataset=%s mode=%s cache_hit=%s trades=%s",
            request.dataset_id,
            scoring_mode,
            cache_hit,
            len(trades),
        )

        raw_bias_scores = cached_analysis.get("bias_scores", {})
        if not isinstance(raw_bias_scores, dict):
            raw_bias_scores = {}

        sortable_biases: list[tuple[str, int]] = []
        for bias_key, score_value in raw_bias_scores.items():
            if not isinstance(bias_key, str):
                continue
            try:
                normalized_score = int(round(float(score_value)))
            except (TypeError, ValueError):
                normalized_score = 0
            sortable_biases.append((bias_key, max(0, min(100, normalized_score))))

        sorted_biases = sorted(sortable_biases, key=lambda row: row[1], reverse=True)[:2]
        if not sorted_biases:
            raise HTTPException(status_code=400, detail="No bias scores available for this dataset.")

        logger.info("Top 2 biases identified for practice: %s", sorted_biases)
        
        # Generate questions using AI
        from app.services.practice_questions import generate_practice_questions
        
        logger.info("Calling generate_practice_questions...")
        summary = cached_analysis.get("summary", {})
        if not isinstance(summary, dict):
            summary = {}
        questions = await generate_practice_questions(
            top_two_biases=sorted_biases,
            trader_stats={
                "total_trades": len(trades),
                "trades_summary": summary,
            }
        )
        logger.info("Received %s questions from AI service", len(questions))
        
        # Build response
        from app.models.schemas import PracticeQuestionsResponse, PracticeQuestion
        
        practice_qs = [
            PracticeQuestion(
                question_id=q["question_id"],
                bias=q["bias"],
                scenario=q["scenario"],
                choices=q["choices"],
                correct_index=q["correct_index"],
                explanation=q["explanation"],
            )
            for q in questions
        ]
        
        response = PracticeQuestionsResponse(
            questions=practice_qs,
            biases_covered=[
                {"bias": b[0], "score": int(b[1])}
                for b in sorted_biases
            ]
        )
        
        logger.info("Returning %s practice questions to frontend", len(questions))
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Practice questions error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


BiasType = Literal["REVENGE_TRADING", "LOSS_AVERSION", "RECENCY_BIAS", "OVERTRADING"]
YesNoAnswer = Literal["YES", "NO"]

class EmotionalCheckInAnswer(BaseModel):
    """Single yes/no/skip answer for a fixed emotional check-in question."""

    question_id: str
    question: str
    answer: Optional[YesNoAnswer] = None


class EmotionalCheckInBiasPayload(BaseModel):
    """Responses for one bias type."""

    bias_type: BiasType
    score: Optional[int] = Field(default=None, ge=0, le=100)
    responses: list[EmotionalCheckInAnswer] = Field(default_factory=list, max_length=3)


class TopBiasScore(BaseModel):
    """A single bias score entry used to determine top-two biases."""

    bias_type: BiasType
    score: int = Field(ge=0, le=100)


class BiasContextRequest(BaseModel):
    """Request for bias context + emotional check-in personalization."""

    dataset_id: str
    session_id: Optional[str] = None
    top_two_biases: list[TopBiasScore] = Field(default_factory=list)
    emotional_check_in: list[EmotionalCheckInBiasPayload] = Field(default_factory=list)
    date_range: Optional[dict[str, str]] = None  # {"start": "2024-01-01", "end": "2024-01-31"}


def _resolve_date_range_from_trades(trades: list[Any]) -> Optional[dict[str, str]]:
    trade_dates: list[str] = []
    for trade in trades:
        if hasattr(trade, "entry_date"):
            date_value = getattr(trade, "entry_date", None) or getattr(trade, "date", None)
        else:
            date_value = trade.get("entry_date") if isinstance(trade, dict) else None
            if not date_value and isinstance(trade, dict):
                date_value = trade.get("date")

        if date_value:
            trade_dates.append(str(date_value))

    if not trade_dates:
        return None

    trade_dates.sort()
    return {"start": trade_dates[0], "end": trade_dates[-1]}


def _resolve_top_two_biases_from_dataset(dataset_id: str, scoring_mode: str = "hybrid") -> list[dict[str, Any]]:
    """Resolve top 2 biases from cached frontend analysis payload for a dataset."""
    normalized_mode = _normalize_scoring_mode(scoring_mode)
    cached_analysis, cache_hit = _get_or_compute_dataset_analysis(dataset_id, normalized_mode)
    logger.info(
        "Resolved top-two bias candidates from cache dataset=%s mode=%s cache_hit=%s",
        dataset_id,
        normalized_mode,
        cache_hit,
    )

    raw_bias_scores = cached_analysis.get("bias_scores", {})
    if not isinstance(raw_bias_scores, dict):
        return []

    bias_rows: list[dict[str, Any]] = []
    for frontend_bias, raw_score in raw_bias_scores.items():
        if not isinstance(frontend_bias, str):
            continue
        bias_type = FRONTEND_BIAS_TO_CANONICAL.get(frontend_bias)
        if not bias_type:
            continue
        try:
            score = int(round(float(raw_score)))
        except (TypeError, ValueError):
            score = 0
        bias_rows.append({"bias_type": bias_type, "score": max(0, min(100, score))})

    bias_rows.sort(key=lambda row: row["score"], reverse=True)
    return bias_rows[:2]


@router.post("/bias-context-explainer")
async def get_bias_context_explainer(request: BiasContextRequest):
    """
    Generate context explanation for trader's top 2 biases.

    Includes emotional check-in responses (YES/NO/Skip) for fixed bias-specific
    questions and returns a personalized, empathetic, process-focused summary.
    """
    try:
        session_id = request.session_id or request.dataset_id
        logger.info(
            "Bias context request: dataset_id=%s session_id=%s checkin_biases=%s",
            request.dataset_id,
            session_id,
            len(request.emotional_check_in),
        )

        if not request.dataset_id or request.dataset_id not in IMPORTED_DATASETS:
            raise HTTPException(status_code=404, detail=f"Dataset not found: {request.dataset_id}")

        # Get the trades to determine date range
        trades = IMPORTED_DATASETS[request.dataset_id]["trades"]

        if not request.date_range:
            request.date_range = _resolve_date_range_from_trades(trades)

        from app.services.bias_context import (
            EMOTIONAL_CHECKIN_QUESTIONS,
            get_bias_context_explanations,
            normalize_emotional_checkin_input,
            normalize_top_two_biases_input,
        )

        requested_top_two_biases = normalize_top_two_biases_input(
            [item.model_dump() for item in request.top_two_biases]
        )
        top_two_biases = requested_top_two_biases
        if not top_two_biases:
            top_two_biases = normalize_top_two_biases_input(
                _resolve_top_two_biases_from_dataset(request.dataset_id, scoring_mode="hybrid")
            )
        if not top_two_biases:
            raise HTTPException(status_code=400, detail="Unable to determine top two biases for this dataset.")

        normalized_checkin_by_bias = normalize_emotional_checkin_input(
            [item.model_dump() for item in request.emotional_check_in]
        )
        top_bias_types = {item["bias_type"] for item in top_two_biases}
        normalized_emotional_checkin: list[dict[str, Any]] = []
        for bias in top_two_biases:
            bias_type = bias["bias_type"]
            responses = normalized_checkin_by_bias.get(bias_type)
            if responses is None:
                responses = [
                    {
                        "question_id": f"{bias_type}_Q{idx + 1}",
                        "question": EMOTIONAL_CHECKIN_QUESTIONS[bias_type][idx],
                        "answer": None,
                    }
                    for idx in range(3)
                ]
            normalized_emotional_checkin.append(
                {
                    "bias_type": bias_type,
                    "score": bias.get("score"),
                    "responses": responses,
                }
            )

        # Persist emotional check-in payload in-memory (optional backend persistence).
        if normalized_emotional_checkin:
            by_bias = EMOTIONAL_CHECKIN_STORE.setdefault(session_id, {})
            for bias_payload in normalized_emotional_checkin:
                if bias_payload["bias_type"] not in top_bias_types:
                    continue
                by_bias[bias_payload["bias_type"]] = {
                    "score": bias_payload.get("score"),
                    "responses": bias_payload.get("responses", []),
                    "updated_at": datetime.utcnow().isoformat() + "Z",
                }

        logger.info(f"Date range: {request.date_range}")

        context_payload = await get_bias_context_explanations(
            top_two_biases=top_two_biases,
            emotional_check_in=normalized_emotional_checkin,
            date_range=request.date_range,
            num_events=5,
        )

        logger.info(
            "Generated context payload: contexts=%s personalized=%s",
            len(context_payload.get("bias_contexts", [])),
            len(context_payload.get("personalized_sections", [])),
        )

        return {
            "session_id": session_id,
            "bias_contexts": context_payload.get("bias_contexts", []),
            "personalized_sections": context_payload.get("personalized_sections", []),
            "saved_emotional_check_in": EMOTIONAL_CHECKIN_STORE.get(session_id, {}),
            "date_range": request.date_range,
            "global_note": context_payload.get(
                "global_note",
                "Educational context only. Emotional explanations are hypotheses, not diagnoses.",
            ),
            "methodology": (
                "Recent market/economic events and emotional check-in responses are combined to build "
                "educational bias context. Personalized explanations are hypotheses, not diagnoses or "
                "financial advice."
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bias context error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

class CoachMessage(BaseModel):
    """Message in coach conversation."""
    role: str  # "user" or "assistant"
    content: str


class CoachChatRequest(BaseModel):
    """Request for coach chat endpoint."""
    dataset_id: str
    chat_history: list[CoachMessage]
    reflection_notes: Optional[list[str]] = None


@router.post("/coach")
async def coach_chat(request: CoachChatRequest):
    """
    Educational finance coach chatbot endpoint.
    Provides explanations of financial concepts, risk management, and trading strategies.
    Does NOT provide personalized financial advice or buy/sell recommendations.
    
    Stores conversation history in-session only (no database persistence).
    Uses recent conversation to understand context for follow-up questions.
    """
    try:
        logger.info("Coach chat request: dataset_id=%s messages=%s", request.dataset_id, len(request.chat_history))
        
        if not request.dataset_id or request.dataset_id not in IMPORTED_DATASETS:
            logger.error(f"Dataset not found: {request.dataset_id}")
            raise HTTPException(status_code=404, detail=f"Dataset not found: {request.dataset_id}")
        
        # Get coach response using LangChain
        from app.services.coach_chat import get_coach_response
        
        response = await get_coach_response(
            chat_history=request.chat_history,
            reflection_notes=request.reflection_notes or [],
        )
        
        logger.info("Coach response generated")
        
        return {
            "assistant_response": response,
            "recommended_next_actions": [],
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Coach chat error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

