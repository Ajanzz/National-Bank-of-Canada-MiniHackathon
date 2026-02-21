import type {
  AnalyzeRequest,
  AnalysisOutput,
  BiasContextExplainerRequest,
  BiasContextExplainerResponse,
  CoachRequest,
  CoachResponse,
  ExplainRequest,
  ExplainResponse,
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
const FAST_TO_FRONTEND_BIAS: Record<string, (typeof REQUIRED_BIAS_KEYS)[number]> = {
  OVERTRADING: "overtrading",
  LOSS_AVERSION: "loss_aversion",
  REVENGE_TRADING: "revenge_trading",
  RECENCY_BIAS: "recency_bias",
};

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
  const fastScores =
    source.biasScores && typeof source.biasScores === "object"
      ? (source.biasScores as Record<string, unknown>)
      : {};
  const normalized = {} as AnalysisOutput["bias_scores"];
  for (const key of REQUIRED_BIAS_KEYS) {
    let candidate = source[key];
    if (candidate == null) {
      const fastKey = Object.entries(FAST_TO_FRONTEND_BIAS).find(([, mapped]) => mapped === key)?.[0];
      if (fastKey) candidate = fastScores[fastKey];
    }
    normalized[key] = Math.max(0, Math.min(100, Math.round(toFiniteNumber(candidate))));
  }
  return normalized;
}

function normalizeDailyPnl(value: unknown): AnalysisOutput["daily_pnl"] {
  const directRows = Array.isArray(value) ? value : [];
  const source = value && typeof value === "object" ? (value as Record<string, unknown>) : {};
  const aggregateRows = Array.isArray(source.aggregates)
    ? []
    : Array.isArray((source.aggregates as Record<string, unknown> | undefined)?.dailyPnl)
      ? (((source.aggregates as Record<string, unknown>).dailyPnl as unknown[]) ?? [])
      : [];
  const rows = directRows.length > 0 ? directRows : aggregateRows;
  if (!Array.isArray(rows)) return [];

  return rows
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
  const source = value && typeof value === "object" ? (value as Record<string, unknown>) : {};
  const directRows = Array.isArray(value) ? value : [];
  const aggregateRows = Array.isArray((source.aggregates as Record<string, unknown> | undefined)?.hourlyPnl)
    ? (((source.aggregates as Record<string, unknown>).hourlyPnl as unknown[]) ?? [])
    : [];
  const rows = directRows.length > 0 ? directRows : aggregateRows;
  if (!Array.isArray(rows)) return [];

  return rows
    .map((item) => {
      const row = item && typeof item === "object" ? (item as Record<string, unknown>) : null;
      const timestamp = typeof row?.timestamp === "string" ? row.timestamp : "";
      if (!timestamp) return null;
      return {
        timestamp,
        pnl: toFiniteNumber(row?.pnl),
        trade_count:
          typeof row?.trade_count === "number" && Number.isFinite(row.trade_count)
            ? Math.max(0, Math.round(row.trade_count))
            : undefined,
        wins:
          typeof row?.wins === "number" && Number.isFinite(row.wins)
            ? Math.max(0, Math.round(row.wins))
            : undefined,
        losses:
          typeof row?.losses === "number" && Number.isFinite(row.losses)
            ? Math.max(0, Math.round(row.losses))
            : undefined,
        flat:
          typeof row?.flat === "number" && Number.isFinite(row.flat)
            ? Math.max(0, Math.round(row.flat))
            : undefined,
      };
    })
    .filter((row): row is AnalysisOutput["trade_timeline"][number] => row !== null)
    .sort((a, b) => a.timestamp.localeCompare(b.timestamp));
}

function normalizeFlaggedTrades(value: unknown): AnalysisOutput["flagged_trades"] {
  if (!Array.isArray(value)) return [];
  const biasMap: Record<string, AnalysisOutput["flagged_trades"][number]["bias"]> = {
    OVERTRADING: "overtrading",
    LOSS_AVERSION: "loss_aversion",
    REVENGE_TRADING: "revenge_trading",
    RECENCY_BIAS: "recency_bias",
  };

  return value
    .map((item) => {
      if (!item || typeof item !== "object") return null;
      const row = item as Record<string, unknown>;
      if (typeof row.bias === "string") {
        return row as AnalysisOutput["flagged_trades"][number];
      }

      const fastBiasFlags = Array.isArray(row.biasFlags)
        ? row.biasFlags
            .filter((bias): bias is string => typeof bias === "string")
            .map((bias) => biasMap[bias] ?? "overtrading")
        : [];

      return {
        trade_id: typeof row.tradeId === "string" ? row.tradeId : undefined,
        timestamp: typeof row.timestamp === "string" ? row.timestamp : undefined,
        symbol: typeof row.asset === "string" ? row.asset : undefined,
        side: typeof row.side === "string" ? row.side : undefined,
        quantity: toFiniteNumber(row.quantity),
        entry_price: toFiniteNumber(row.entry_price),
        exit_price: toFiniteNumber(row.exit_price),
        profit_loss: toFiniteNumber(row.pnl),
        balance: toFiniteNumber(row.balance),
        bias: fastBiasFlags[0] ?? "overtrading",
        bias_types: fastBiasFlags,
        confidence: toFiniteNumber(row.severity),
        flag_severity: toFiniteNumber(row.severity) >= 70 ? "high" : toFiniteNumber(row.severity) >= 40 ? "medium" : "low",
        evidence: Array.isArray(row.explanationBullets)
          ? row.explanationBullets.filter((bullet): bullet is string => typeof bullet === "string")
          : [],
      };
    })
    .filter((row): row is AnalysisOutput["flagged_trades"][number] => row !== null);
}

function normalizeAnalysisOutput(value: unknown): AnalysisOutput {
  const source = value && typeof value === "object" ? (value as Record<string, unknown>) : {};

  const topTriggers = Array.isArray(source.top_triggers)
    ? source.top_triggers.filter((item): item is string => typeof item === "string")
    : Array.isArray(source.topBiases)
      ? source.topBiases
          .filter((item): item is string => typeof item === "string")
          .map((bias) => {
            if (bias === "OVERTRADING") return "Overtrading";
            if (bias === "LOSS_AVERSION") return "Loss Aversion";
            if (bias === "REVENGE_TRADING") return "Revenge Trading";
            if (bias === "RECENCY_BIAS") return "Recency Bias";
            return bias;
          })
      : [];

  const flaggedTrades = Array.isArray(source.flagged_trades)
    ? normalizeFlaggedTrades(source.flagged_trades)
    : normalizeFlaggedTrades(source.flaggedTrades);

  const explainability =
    source.explainability && typeof source.explainability === "object"
      ? (source.explainability as AnalysisOutput["explainability"])
      : ({} as AnalysisOutput["explainability"]);

  const result: AnalysisOutput = {
    behavior_index: toFiniteNumber(source.behavior_index, toFiniteNumber(source.behaviorIndex)),
    bias_scores: normalizeBiasScores(source.bias_scores ?? source),
    top_triggers: topTriggers,
    danger_hours: normalizeDangerHours(source.danger_hours ?? source.dangerHours),
    daily_pnl: normalizeDailyPnl(source.daily_pnl ?? source),
    trade_timeline: normalizeTradeTimeline(source.trade_timeline ?? source.tradeTimeline ?? source),
    flagged_trades: flaggedTrades,
    explainability,
  };

  if (source.summary && typeof source.summary === "object") {
    result.summary = source.summary as AnalysisOutput["summary"];
  }

  return result;
}

async function importDatasetLegacyCsv(
  file: File,
  options?: { timezone?: string; session_template?: string }
) {
  const rawCsv = await file.text();
  return requestJson<ImportDatasetResponse>("/import/csv", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      raw_csv: rawCsv,
      timezone: options?.timezone ?? "UTC",
      session_template: options?.session_template ?? "equities_rth",
    }),
  });
}

export async function importDatasetFile(
  file: File,
  options?: { timezone?: string; session_template?: string }
) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("timezone", options?.timezone ?? "UTC");
  formData.append("session_template", options?.session_template ?? "equities_rth");

  const res = await fetch(buildUrl("/import/csv-file"), {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const text = await res.text();
    if (res.status === 404 || res.status === 405) {
      // Backward-compatible fallback for backends that only expose /import/csv.
      return importDatasetLegacyCsv(file, options);
    }
    throw new Error(text || `Request failed: ${res.status}`);
  }
  return (await res.json()) as ImportDatasetResponse;
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

export function getBiasContextExplainer(payload: BiasContextExplainerRequest) {
  return requestJson<BiasContextExplainerResponse>("/bias-context-explainer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}
