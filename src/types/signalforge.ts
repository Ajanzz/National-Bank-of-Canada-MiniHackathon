export type ThemeMode = "dark";

export type ScoringMode = "rule_only" | "hybrid";

export type BiasKey = "overtrading" | "loss_aversion" | "revenge_trading" | "recency_bias";

export type CanonicalTradeField =
  | "trade_id"
  | "timestamp"
  | "open_time"
  | "close_time"
  | "symbol"
  | "side"
  | "size"
  | "entry_price"
  | "exit_price"
  | "pnl"
  | "fees"
  | "balance";

export type ImportDatasetRequest = {
  raw_csv?: string;
  rows?: Record<string, string>[];
  mapping?: Partial<Record<CanonicalTradeField, string>>;
  timezone: string;
  session_template: string;
};

export type ImportDatasetResponse = {
  dataset_id: string;
  stats: {
    rows: number;
    detected_date_range?: {
      start: string;
      end: string;
    };
  };
};

export type DriverContribution = {
  feature: string;
  contribution: number;
  direction: "positive" | "negative";
  detail?: string;
};

export type FlaggedTrade = {
  trade_id?: string;
  timestamp?: string;
  symbol?: string;
  bias: BiasKey;
  bias_types?: BiasKey[];
  confidence: number;
  evidence: string[];
  side?: string;
  quantity?: number;
  entry_price?: number;
  exit_price?: number;
  profit_loss?: number;
  balance?: number;
  flag_severity?: string;
};

export type DailyPnlPoint = {
  date: string;
  pnl: number;
};

export type TradeTimelinePoint = {
  timestamp: string;
  pnl: number;
  trade_count?: number;
  wins?: number;
  losses?: number;
  flat?: number;
};

export type AnalysisOutput = {
  behavior_index: number;
  bias_scores: Record<BiasKey, number>;
  top_triggers: string[];
  danger_hours: number[][];
  daily_pnl: DailyPnlPoint[];
  trade_timeline: TradeTimelinePoint[];
  flagged_trades: FlaggedTrade[];
  explainability: Record<BiasKey, DriverContribution[]>;
  summary?: {
    total_trades: number;
    winning_trades: number;
    losing_trades: number;
    win_rate: number;
    total_pnl: number;
    avg_pnl_per_trade: number;
    starting_balance: number;
    ending_balance: number;
    max_drawdown: number;
    volatility: number;
    date_range: string;
  };
};

export type AnalyzeRequest = {
  dataset_id: string;
  scoring_mode: ScoringMode;
};

export type ExplainRequest = {
  dataset_id: string;
  bias_type: BiasKey;
  trade_id?: string;
};

export type ExplainResponse = {
  drivers: DriverContribution[];
  evidence_bullets: string[];
  percentile_comparisons: string[];
};

export type SimulateRequest = {
  dataset_id?: string;
  trades?: Array<{
    timestamp: string;
    asset: string;
    side: string;
    quantity: number;
    entry_price: number;
    exit_price: number;
    profit_loss: number;
    balance: number;
  }>;
  rules: {
    cooldown_after_loss_minutes?: number;
    daily_trade_cap?: number;
    max_position_size_multiplier?: number;
    stop_after_consecutive_losses?: number;
  };
};

export type SimulatedTrade = {
  trade_id: string;
  timestamp: string;
  asset: string;
  profit_loss: number;
  balance_before: number;
  balance_after: number;
  allowed: boolean;
  reason?: string;
};

export type SimulationLog = {
  rule: string;
  trades_skipped: number;
  pnl_impact: number;
};

export type SimulationResult = {
  baseline_pnl: number;
  baseline_trade_count: number;
  simulated_pnl: number;
  simulated_trade_count: number;
  pnl_difference: number;
  blocked_trades: SimulatedTrade[];
  simulation_log: SimulationLog[];
};

export type NewsItem = {
  id: string;
  timestamp: string;
  headline: string;
  category: string;
  symbols?: string[];
  tags?: string[];
};

export type NewsResponse = {
  headlines: NewsItem[];
  event_windows: Array<{
    start: string;
    end: string;
    label: string;
  }>;
};

export type CoachMessage = {
  role: "user" | "assistant";
  content: string;
};

export type CoachRequest = {
  dataset_id: string;
  chat_history: CoachMessage[];
  reflection_notes: string[];
};

export type CoachResponse = {
  assistant_response: string;
  recommended_next_actions: string[];
};

export type PracticeQuestion = {
  question_id: string;
  bias: BiasKey;
  scenario: string;
  choices: string[];
  correct_index: number;
  explanation: string;
};

export type PracticeQuestionsRequest = {
  dataset_id: string;
  scoring_mode?: ScoringMode;
};

export type PracticeQuestionsResponse = {
  questions: PracticeQuestion[];
  biases_covered: Array<{
    bias: BiasKey;
    score: number;
  }>;
};

export type BiasType = "REVENGE_TRADING" | "LOSS_AVERSION" | "RECENCY_BIAS" | "OVERTRADING";

export type EmotionalCheckInAnswer = "YES" | "NO" | null;

export type EmotionalCheckInQuestionResponse = {
  question_id: string;
  question: string;
  answer: EmotionalCheckInAnswer;
};

export type EmotionalCheckInBiasPayload = {
  bias_type: BiasType;
  score: number;
  responses: EmotionalCheckInQuestionResponse[];
};

export type BiasContextExplainerRequest = {
  dataset_id: string;
  session_id?: string;
  top_two_biases: Array<{ bias_type: BiasType; score: number }>;
  emotional_check_in: EmotionalCheckInBiasPayload[];
  date_range?: { start: string; end: string };
};

export type BiasContextEntry = {
  bias_type: BiasType;
  bias_name: string;
  score?: number;
  market_events: Array<{ date: string; headline: string }>;
  connection_explanation: string;
  practical_takeaway: string;
};

export type PersonalizedBiasEntry = {
  bias_type: BiasType;
  bias_name: string;
  headline: string;
  supportive_explanation: string;
  hypothetical_contributors: string[];
  gentle_process_habits: string[];
  checkin_summary: {
    yes_count: number;
    no_count: number;
    skipped_count: number;
  };
  compassionate_note: string;
};

export type BiasContextExplainerResponse = {
  session_id: string;
  bias_contexts: BiasContextEntry[];
  personalized_sections: PersonalizedBiasEntry[];
  saved_emotional_check_in: Record<
    string,
    {
      score?: number | null;
      updated_at?: string;
      responses: EmotionalCheckInQuestionResponse[];
    }
  >;
  date_range?: { start: string; end: string };
  global_note: string;
  methodology: string;
};
