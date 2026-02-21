"""FastAPI routes for the backend API."""

from __future__ import annotations

import base64
import io
import uuid
from datetime import datetime
from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter, Body, File, HTTPException, UploadFile
from pydantic import BaseModel, ValidationError

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
from app.services.ingest import ingest_csv
from app.services.analysis import analyze_trades
from app.services.simulator import simulate_with_rules
from app.utils.csv_utils import normalize_headers, parse_base64_csv, parse_csv
from app.events.shock_generator import generate_market_shocks_from_trades
from app.events.provider import StubEventsProvider

# In-memory storage for imported datasets
IMPORTED_DATASETS = {}

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
        
        # Parse CSV content
        df = parse_csv(request.raw_csv)
        logger.info(f"Parsed CSV: {df.shape[0]} rows, {df.shape[1]} columns")
        logger.info(f"CSV columns: {list(df.columns)}")
        
        # Ingest and normalize with mapping
        normalized_trades, errors = ingest_csv(df, mapping=request.mapping)
        logger.info(f"Ingestion result: {len(normalized_trades)} trades, errors={errors}")
        
        if errors:
            logger.warning(f"CSV ingestion warnings: {errors}")
        
        if not normalized_trades:
            raise HTTPException(
                status_code=400,
                detail=f"No valid trades parsed from CSV. Errors: {errors}"
            )
        
        # Generate dataset_id and store trades
        dataset_id = f"ds-{uuid.uuid4().hex[:8]}"
        IMPORTED_DATASETS[dataset_id] = {
            "trades": normalized_trades,
            "timezone": request.timezone,
            "session_template": request.session_template,
        }
        
        # Prepare stats
        if normalized_trades:
            dates = [t.timestamp.date() for t in normalized_trades]
            stats = {
                "rows": len(normalized_trades),
                "detected_date_range": {
                    "start": min(dates).isoformat(),
                    "end": max(dates).isoformat(),
                } if dates else None,
            }
        else:
            stats = {"rows": 0}
        
        logger.info(f"CSV import successful: dataset_id={dataset_id}, trades={len(normalized_trades)}")
        return ImportDatasetResponse(dataset_id=dataset_id, stats=stats)
    
    except HTTPException:
        raise
    except ValidationError as ve:
        logger.error(f"Validation error: {ve.json()}")
        raise HTTPException(status_code=422, detail=f"Validation error: {str(ve)}")
    except Exception as e:
        logger.error(f"CSV import error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


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
        
        normalized_trades = []
        
        # Parse based on input
        if request.dataset_id:
            # Retrieve from stored datasets
            if request.dataset_id not in IMPORTED_DATASETS:
                logger.error(f"Dataset not found: {request.dataset_id}, available: {list(IMPORTED_DATASETS.keys())}")
                raise HTTPException(
                    status_code=404,
                    detail=f"Dataset not found: {request.dataset_id}"
                )
            normalized_trades = IMPORTED_DATASETS[request.dataset_id]["trades"]
            logger.info(f"Retrieved dataset {request.dataset_id}: {len(normalized_trades)} trades")
        
        elif request.base64_csv:
            df = parse_base64_csv(request.base64_csv)
            normalized_trades, errors = ingest_csv(df)
            
            if errors:
                logger.warning(f"CSV analysis warnings: {errors}")
            
            if not normalized_trades:
                raise HTTPException(
                    status_code=400,
                    detail=f"No valid trades in CSV. Errors: {errors}"
                )
        
        elif request.trades:
            # Convert Trade objects to NormalizedTrade
            from app.services.ingest import REQUIRED_COLUMNS
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
        
        # Normalize scoring_mode: the frontend may send "hybrid", "rules", "rule_only", or None
        scoring_mode = (request.scoring_mode or "hybrid").lower()
        if scoring_mode == "rule_only":
            scoring_mode = "rules"
        if scoring_mode not in ("hybrid", "rules"):
            scoring_mode = "hybrid"

        # Analyze
        result = analyze_trades(normalized_trades, scoring_mode=scoring_mode)

        # Map internal bias names to frontend bias keys
        bias_name_map = {
            "Overtrading": "overtrading",
            "Loss Aversion": "loss_aversion",
            "Revenge Trading": "revenge_trading",
            "Recency Bias": "recency_bias",
        }
        
        # Transform to frontend-expected structure
        frontend_response = {
            "behavior_index": result.behavior_index,
            "bias_scores": {},
            "top_triggers": [],
            "danger_hours": result.heatmap,
            "flagged_trades": [],
            "explainability": {},
            "summary": result.summary,
            "ml_active": result.ml_active,
            "ml_probabilities": result.ml_probabilities,
            "scoring_mode": scoring_mode,
        }
        
        # Populate bias scores
        for card in result.bias_cards:
            frontend_key = bias_name_map.get(card.bias_type, card.bias_type.lower().replace(" ", "_"))
            frontend_response["bias_scores"][frontend_key] = card.score
            if card.score > 50:
                frontend_response["top_triggers"].append(card.bias_type)
        
        # Populate flagged trades
        for ft in result.flagged_trades:
            # Convert bias_type (e.g., "Overtrading, Loss Aversion") to array of keys
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
        logger.info(f"🎯 Practice questions request: dataset_id={request.dataset_id}, scoring_mode={request.scoring_mode}")
        
        # Get or analyze dataset
        if not request.dataset_id:
            raise HTTPException(status_code=400, detail="dataset_id is required")
        
        if request.dataset_id not in IMPORTED_DATASETS:
            raise HTTPException(status_code=404, detail=f"Dataset not found: {request.dataset_id}")
        
        # Retrieve stored trades
        trades = IMPORTED_DATASETS[request.dataset_id]["trades"]
        logger.info(f"Analyzing {len(trades)} trades from dataset")
        
        # Analyze to get bias scores
        analysis_result = analyze_trades(trades)
        logger.info(f"Analysis complete. Top bias: {analysis_result.bias_cards[0].bias_type if analysis_result.bias_cards else 'N/A'}")
        
        # Extract bias scores from analysis result
        bias_scores = {}
        for card in analysis_result.bias_cards:
            # Map internal names to frontend keys
            bias_name_map = {
                "Overtrading": "overtrading",
                "Loss Aversion": "loss_aversion",
                "Revenge Trading": "revenge_trading",
                "Recency Bias": "recency_bias",
            }
            frontend_key = bias_name_map.get(card.bias_type, card.bias_type.lower().replace(" ", "_"))
            bias_scores[frontend_key] = card.score
        
        # Get top 2 biases
        sorted_biases = sorted(bias_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        
        logger.info(f"📊 Top 2 biases identified: {sorted_biases}")
        
        # Generate questions using AI
        from app.services.practice_questions import generate_practice_questions
        
        logger.info("⏳ Calling generate_practice_questions...")
        questions = await generate_practice_questions(
            top_two_biases=sorted_biases,
            trader_stats={
                "total_trades": len(trades),
                "trades_summary": analysis_result.summary,
            }
        )
        logger.info(f"✓ Received {len(questions)} questions from AI service")
        
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
        
        logger.info(f"✅ Returning {len(questions)} practice questions to frontend")
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Practice questions error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class BiasContextRequest(BaseModel):
    """Request for bias context explainer with market events."""
    dataset_id: str
    top_two_biases: list[dict[str, Any]]  # [{"bias_type": "overtrading", "score": 85}, ...]
    date_range: Optional[dict[str, str]] = None  # {"start": "2024-01-01", "end": "2024-01-31"}


@router.post("/bias-context-explainer")
async def get_bias_context_explainer(request: BiasContextRequest):
    """
    Generate context explanation for trader's top 2 biases using market events.
    
    Returns market/economic events from the trading period, explains how they 
    may have contributed to the detected biases, and provides practical takeaways.
    """
    try:
        logger.info(f"📊 Bias context request: dataset_id={request.dataset_id}")
        
        if not request.dataset_id or request.dataset_id not in IMPORTED_DATASETS:
            raise HTTPException(status_code=404, detail=f"Dataset not found: {request.dataset_id}")
        
        # Get the trades to determine date range
        trades = IMPORTED_DATASETS[request.dataset_id]["trades"]
        
        if not request.date_range:
            # Calculate from trades - handle both dict and Pydantic model objects
            trade_dates = []
            for t in trades:
                # If it's a Pydantic model, access as attribute; if dict, use get()
                if hasattr(t, 'entry_date'):
                    date = getattr(t, 'entry_date', None) or getattr(t, 'date', None)
                else:
                    date = t.get("entry_date") if isinstance(t, dict) else None
                    if not date:
                        date = t.get("date") if isinstance(t, dict) else None
                
                if date:
                    trade_dates.append(date)
            
            if trade_dates:
                trade_dates.sort()
                request.date_range = {
                    "start": str(trade_dates[0]),
                    "end": str(trade_dates[-1]),
                }
        
        logger.info(f"Date range: {request.date_range}")
        
        # Get context explanations from AI service
        from app.services.bias_context import get_bias_context_explanations
        
        context_explanations = await get_bias_context_explanations(
            top_two_biases=request.top_two_biases,
            date_range=request.date_range,
            num_events=5,
        )
        
        logger.info(f"✅ Generated bias context explanations for {len(context_explanations)} biases")
        
        return {
            "bias_contexts": context_explanations,
            "date_range": request.date_range,
            "methodology": "Market events and economic data are correlated with detected bias patterns. This provides context, not causation.",
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
        logger.info(f"📚 Coach chat request: dataset_id={request.dataset_id}, messages={len(request.chat_history)}")
        
        if not request.dataset_id or request.dataset_id not in IMPORTED_DATASETS:
            logger.error(f"Dataset not found: {request.dataset_id}")
            raise HTTPException(status_code=404, detail=f"Dataset not found: {request.dataset_id}")
        
        # Get coach response using LangChain
        from app.services.coach_chat import get_coach_response
        
        response = await get_coach_response(
            chat_history=request.chat_history,
            reflection_notes=request.reflection_notes or [],
        )
        
        logger.info(f"✅ Coach response generated")
        
        return {
            "assistant_response": response,
            "recommended_next_actions": [],
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Coach chat error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
