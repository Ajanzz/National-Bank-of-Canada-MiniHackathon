"""Pydantic schemas for request/response models."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Trade(BaseModel):
    """Base trade model."""

    timestamp: datetime
    asset: str
    side: str  # BUY or SELL
    quantity: float
    entry_price: float
    exit_price: float
    profit_loss: float
    balance: float


class NormalizedTrade(BaseModel):
    """Normalized trade with computed fields."""

    trade_id: str
    timestamp: datetime
    asset: str
    side: str  # BUY or SELL
    quantity: float
    entry_price: float
    exit_price: float
    profit_loss: float
    balance: float
    date: str
    hour: int
    weekday: int
    is_win: bool
    abs_pnl: float
    size_usd: float
    return_pct: float


class DetectionResult(BaseModel):
    """Result from a bias detector."""

    score: int = Field(ge=0, le=100)
    severity: str  # "low", "med", "high"
    flagged_trade_ids: list[str]
    evidence_by_trade_id: dict[str, str]
    stats: dict[str, float | int | str]


class FlaggedTradeDetail(BaseModel):
    """Flagged trade with evidence."""

    trade_id: str
    timestamp: datetime
    asset: str
    side: str
    quantity: float
    entry_price: float
    exit_price: float
    profit_loss: float
    balance: float
    bias_type: str
    flag_severity: str
    confidence: int
    evidence: str


class BiasCard(BaseModel):
    """Bias detection result for frontend."""

    bias_type: str
    score: int
    severity: str
    flagged_trade_count: int
    key_stats: dict[str, float | int | str]


class EquityCurvePoint(BaseModel):
    """Point in equity curve."""

    timestamp: datetime
    balance: float


class AnalysisResult(BaseModel):
    """Complete analysis result."""

    behavior_index: float = Field(description="Overall bias risk score (0-100)")
    summary: dict = Field(description="Totals, win_rate, pnl, etc.")
    bias_cards: list[BiasCard]
    flagged_trades: list[FlaggedTradeDetail]
    heatmap: list[list[int]] = Field(description="7x24 matrix (weekday x hour)")
    equity_curve: list[EquityCurvePoint]
    optional_events: list[dict] = Field(default_factory=list, description="Events from provider + shocks")
    ml_active: bool = Field(default=False, description="Whether XGBoost model contributed to scores")
    ml_probabilities: dict[str, float] = Field(
        default_factory=dict,
        description="Raw per-class ML probabilities from XGBoost (empty if rules-only)",
    )


class ImportRequest(BaseModel):
    """CSV import request."""

    csv_content: Optional[str] = None
    base64_csv: Optional[str] = None


class AnalyzeRequest(BaseModel):
    """Analyze request."""

    dataset_id: Optional[str] = None
    trades: Optional[list[Trade]] = None
    base64_csv: Optional[str] = None
    scoring_mode: Optional[str] = "hybrid"


class SimulationRule(BaseModel):
    """Simulation rule."""

    cooldown_after_loss_minutes: Optional[int] = None
    daily_trade_cap: Optional[int] = None
    max_position_size_multiplier: Optional[float] = None
    stop_after_consecutive_losses: Optional[int] = None


class SimulateRequest(BaseModel):
    """Simulate request."""

    dataset_id: Optional[str] = None
    trades: Optional[list[Trade]] = None
    rules: SimulationRule


class SimulatedTrade(BaseModel):
    """Simulated trade result."""

    trade_id: str
    timestamp: datetime
    asset: str
    profit_loss: float
    balance_before: float
    balance_after: float
    allowed: bool
    reason: Optional[str] = None


class SimulationResult(BaseModel):
    """Simulation result."""

    baseline_pnl: float
    baseline_trade_count: int
    simulated_pnl: float
    simulated_trade_count: int
    pnl_difference: float
    blocked_trades: list[SimulatedTrade]
    simulation_log: list[dict]


class EventData(BaseModel):
    """Event data point."""

    timestamp: datetime
    event_type: str  # "news", "market_shock"
    label: str
    symbols: Optional[list[str]] = None
    pnl_magnitude: Optional[float] = None


class EventsResponse(BaseModel):
    """Events response."""

    events: list[EventData]


class NewsItem(BaseModel):
    """News item."""

    timestamp: datetime
    headline: str
    source: str
    symbols: Optional[list[str]] = None


class NewsResponse(BaseModel):
    """News response."""

    news: list[NewsItem]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str


class PracticeQuestion(BaseModel):
    """A practice question for trading bias training."""

    question_id: str
    bias: str
    scenario: str
    choices: list[str]
    correct_index: int
    explanation: str


class PracticeQuestionsRequest(BaseModel):
    """Request for generating practice questions."""

    dataset_id: Optional[str] = None
    scoring_mode: Optional[str] = "hybrid"


class PracticeQuestionsResponse(BaseModel):
    """Response with generated practice questions."""

    questions: list[PracticeQuestion]
    biases_covered: list[dict]  # [{"bias": "overtrading", "score": 75}, ...]
