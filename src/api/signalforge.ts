import type {
  AnalyzeRequest,
  AnalysisOutput,
  CoachRequest,
  CoachResponse,
  ExplainRequest,
  ExplainResponse,
  ImportDatasetRequest,
  ImportDatasetResponse,
  NewsResponse,
  PracticeQuestionsRequest,
  PracticeQuestionsResponse,
  SimulateRequest,
  SimulationResult,
} from "../types/signalforge";

const REQUIRED_BIAS_KEYS = ["overtrading", "loss_aversion", "revenge_trading", "recency_bias"] as const;
const WEEKDAY_COUNT = 7;
const HOUR_COUNT = 24;

const rawApiBaseUrl =
  (import.meta as ImportMeta & {
    env?: {
      VITE_SIGNALFORGE_API_BASE_URL?: string;
    };
  }).env?.VITE_SIGNALFORGE_API_BASE_URL ?? "http://localhost:7000/api/v1";

const API_BASE_URL = (rawApiBaseUrl ?? "http://localhost:7000/api/v1").replace(/\/$/, "");

function buildUrl(path: string) {
  if (!API_BASE_URL) return path;
  return `${API_BASE_URL}${path}`;
}

async function requestJson<T>(path: string, init: RequestInit): Promise<T> {
  const res = await fetch(buildUrl(path), init);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Request failed: ${res.status}`);
  }
  return (await res.json()) as T;
}

function toFiniteNumber(value: unknown, fallback: number = 0) {
  if (typeof value !== "number" || !Number.isFinite(value)) return fallback;
  return value;
}

function normalizeDangerHours(value: unknown): number[][] {
  const rows = Array.isArray(value) ? value : [];
  return Array.from({ length: WEEKDAY_COUNT }, (_, weekday) =>
    Array.from({ length: HOUR_COUNT }, (_, hour) => {
      const cell = (rows[weekday] as unknown[] | undefined)?.[hour];
      return Math.max(0, Math.round(toFiniteNumber(cell)));
    })
  );
}

function normalizeBiasScores(value: unknown): AnalysisOutput["bias_scores"] {
  const source = value && typeof value === "object" ? (value as Record<string, unknown>) : {};
  const normalized = {} as AnalysisOutput["bias_scores"];
  for (const key of REQUIRED_BIAS_KEYS) {
    normalized[key] = Math.max(0, Math.min(100, Math.round(toFiniteNumber(source[key]))));
  }
  return normalized;
}

function normalizeDailyPnl(value: unknown): AnalysisOutput["daily_pnl"] {
  if (!Array.isArray(value)) return [];

  return value
    .map((item) => {
      const row = item && typeof item === "object" ? (item as Record<string, unknown>) : null;
      const date = typeof row?.date === "string" ? row.date : "";
      if (!date) return null;
      return {
        date,
        pnl: toFiniteNumber(row?.pnl),
      };
    })
    .filter((row): row is AnalysisOutput["daily_pnl"][number] => row !== null)
    .sort((a, b) => a.date.localeCompare(b.date));
}

function normalizeTradeTimeline(value: unknown): AnalysisOutput["trade_timeline"] {
  if (!Array.isArray(value)) return [];

  return value
    .map((item) => {
      const row = item && typeof item === "object" ? (item as Record<string, unknown>) : null;
      const timestamp = typeof row?.timestamp === "string" ? row.timestamp : "";
      if (!timestamp) return null;
      return {
        timestamp,
        pnl: toFiniteNumber(row?.pnl),
      };
    })
    .filter((row): row is AnalysisOutput["trade_timeline"][number] => row !== null)
    .sort((a, b) => a.timestamp.localeCompare(b.timestamp));
}

function normalizeAnalysisOutput(value: unknown): AnalysisOutput {
  const source = value && typeof value === "object" ? (value as Record<string, unknown>) : {};

  const topTriggers = Array.isArray(source.top_triggers)
    ? source.top_triggers.filter((item): item is string => typeof item === "string")
    : [];

  const flaggedTrades = Array.isArray(source.flagged_trades)
    ? source.flagged_trades.filter((item): item is AnalysisOutput["flagged_trades"][number] => Boolean(item))
    : [];

  const explainability =
    source.explainability && typeof source.explainability === "object"
      ? (source.explainability as AnalysisOutput["explainability"])
      : ({} as AnalysisOutput["explainability"]);

  const result: AnalysisOutput = {
    behavior_index: toFiniteNumber(source.behavior_index),
    bias_scores: normalizeBiasScores(source.bias_scores),
    top_triggers: topTriggers,
    danger_hours: normalizeDangerHours(source.danger_hours),
    daily_pnl: normalizeDailyPnl(source.daily_pnl),
    trade_timeline: normalizeTradeTimeline(source.trade_timeline),
    flagged_trades: flaggedTrades,
    explainability,
  };

  if (source.summary && typeof source.summary === "object") {
    result.summary = source.summary as AnalysisOutput["summary"];
  }

  return result;
}

export function importDataset(payload: ImportDatasetRequest) {
  return requestJson<ImportDatasetResponse>("/import/csv", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function analyzeDataset(payload: AnalyzeRequest) {
  const raw = await requestJson<unknown>("/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return normalizeAnalysisOutput(raw);
}

export function explainDataset(payload: ExplainRequest) {
  return requestJson<ExplainResponse>("/explain", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function simulateDataset(payload: SimulateRequest) {
  return requestJson<SimulationResult>("/simulate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function getNews(params: { from: string; to: string; symbols?: string[] }) {
  const qp = new URLSearchParams({ start: params.from, end: params.to });
  if (params.symbols?.length) qp.set("symbols", params.symbols.join(","));
  return requestJson<NewsResponse>(`/news?${qp.toString()}`, {
    method: "GET",
  });
}

export function coachChat(payload: CoachRequest) {
  return requestJson<CoachResponse>("/coach", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function generatePracticeQuestions(payload: PracticeQuestionsRequest) {
  return requestJson<PracticeQuestionsResponse>("/practice/questions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function getBiasContextExplainer(payload: {
  dataset_id: string;
  top_two_biases: Array<{ bias_type: string; score: number }>;
  date_range?: { start: string; end: string };
}) {
  return requestJson<any>("/bias-context-explainer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}
