import React, { useCallback, useEffect, useState, useRef, useMemo } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { LeftOutlined, RightOutlined, SearchOutlined } from "@ant-design/icons";
import * as TabsPrimitive from "@radix-ui/react-tabs";
import {
  Bar as RechartsBar,
  CartesianGrid as RechartsCartesianGrid,
  ComposedChart as RechartsComposedChart,
  Line as RechartsLine,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis as RechartsXAxis,
  YAxis as RechartsYAxis,
} from "recharts";
import {
  eachDayOfInterval,
  endOfYear,
  endOfMonth,
  format as formatDate,
  parseISO,
  startOfDay,
  startOfMonth,
  startOfYear,
} from "date-fns";
import {
  Alert as AntAlert,
  Button as AntButton,
  Card as AntCard,
  Carousel as AntCarousel,
  Input as AntInput,
  message as antMessage,
  Segmented as AntSegmented,
  Select as AntSelect,
  Slider as AntSlider,
  Table as AntTable,
  Tag as AntTag,
  Tooltip as AntTooltip,
  Upload as AntUpload,
} from "antd";
import "./styles.css";
import { analyzeDataset, importDatasetFile, simulateDataset, generatePracticeQuestions, getBiasContextExplainer, coachChat } from "./api/signalforge";
import { PnlCalendarControl, type PnlCalendarMode, type PnlCalendarValue } from "./components/PnlCalendarControl";
import type {
  AnalysisOutput,
  BiasKey,
  BiasContextExplainerResponse,
  BiasType as CanonicalBiasType,
  EmotionalCheckInAnswer,
  EmotionalCheckInBiasPayload,
  FlaggedTrade,
  PracticeQuestion,
  ScoringMode,
} from "./types/signalforge";

/* --------------------------- Types --------------------------- */

type Workspace =
  | "command"
  | "trades"
  | "simulator"
  | "replay"
  | "pulse"
  | "coach";

type DisplayBiasType = "Overtrading" | "Loss Aversion" | "Revenge Trading" | "Recency Bias";

type Trade = {
  id: string;
  time: string;
  symbol: string;
  side: "Buy" | "Sell";
  size: number;
  durationMin: number;
  pnl: number;
  balance?: number;
  flags: DisplayBiasType[];
  confidence: number; // 0..100
  evidence: string;
};

type YahooHeadline = {
  id: string;
  title: string;
  link: string;
  publishedAt: string;
};

type CsvImportPayload = {
  file: File;
};

const ICON_PATHS = {
  command: "M12 2l7 4v8c0 5-3 8-7 10-4-2-7-5-7-10V6l7-4z",
  trades: "M4 19V5h16v14H4zm2-2h12V7H6v10z",
  heatmap: "M4 6h6v6H4V6zm10 0h6v6h-6V6zM4 14h6v6H4v-6zm10 0h6v6h-6v-6z",
  simulator: "M6 20h12a2 2 0 002-2V9H4v9a2 2 0 002 2zm-2-13h16l-2-3H6L4 7z",
  replay: "M6 4h12v16H6V4zm3 4l6 4-6 4V8z",
  pulse: "M4 13h3l2-6 4 12 2-6h3",
  coach: "M4 5h16v10H7l-3 3V5zm3 3h10v2H7V8zm0 4h7v2H7v-2z",
  close: "M18.3 5.7 12 12l6.3 6.3-1.4 1.4L10.6 13.4 4.3 19.7 2.9 18.3 9.2 12 2.9 5.7 4.3 4.3l6.3 6.3 6.3-6.3z",
  upload: "M12 3l4 4h-3v7h-2V7H8l4-4zm-7 14h14v2H5v-2z",
} as const;

type IconName = keyof typeof ICON_PATHS;

const SAMPLE_TRADES: Trade[] = [
  {
    id: "T-19321",
    time: "10:41",
    symbol: "TSLA",
    side: "Buy",
    size: 2.0,
    durationMin: 6,
    pnl: 228,
    flags: ["Revenge Trading"],
    confidence: 86,
    evidence: "Re-entry 90s after loss - size 2.1x median - 2-loss streak",
  },
  {
    id: "T-19322",
    time: "11:03",
    symbol: "AAPL",
    side: "Buy",
    size: 1.2,
    durationMin: 2,
    pnl: -78,
    flags: ["Overtrading"],
    confidence: 72,
    evidence: "7 trades in 10m window - hold time bottom 8%",
  },
  {
    id: "T-19323",
    time: "13:58",
    symbol: "NVDA",
    side: "Sell",
    size: 0.8,
    durationMin: 54,
    pnl: -312,
    flags: ["Loss Aversion"],
    confidence: 79,
    evidence: "Loss held top 92% - win hold median 8m vs loss 51m",
  },
  {
    id: "T-19324",
    time: "14:22",
    symbol: "SPY",
    side: "Buy",
    size: 1.0,
    durationMin: 9,
    pnl: 64,
    flags: [],
    confidence: 0,
    evidence: "Within baseline behavior range",
  },
];

const YAHOO_FINANCE_SEARCH_API_URL = "https://query1.finance.yahoo.com/v1/finance/search";

/* --------------------------- Utils --------------------------- */

function cn(...xs: Array<string | false | null | undefined>) {
  return xs.filter(Boolean).join(" ");
}

function getErrorMessage(error: unknown, fallback: string) {
  if (error instanceof Error && error.message) return error.message;
  return fallback;
}

function cleanText(value: string) {
  return value.replace(/\s+/g, " ").trim();
}

function parseYahooFinanceSearchResponse(payloadText: string): YahooHeadline[] {
  try {
    const payload = JSON.parse(payloadText) as {
      news?: Array<{
        uuid?: string;
        title?: string;
        link?: string;
        providerPublishTime?: number;
      }>;
    };

    const newsItems = Array.isArray(payload.news) ? payload.news : [];
    const sortedNewsItems = [...newsItems].sort((a, b) => (b.providerPublishTime ?? 0) - (a.providerPublishTime ?? 0));

    return sortedNewsItems
      .slice(0, 8)
      .map((item, index) => {
        const title = cleanText(item.title ?? "");
        const link = cleanText(item.link ?? "");
        const publishedAt = item.providerPublishTime
          ? new Date(item.providerPublishTime * 1000).toISOString()
          : "";
        if (!title || !link) return null;

        return {
          id: item.uuid || `${index}-${link}`,
          title,
          link,
          publishedAt,
        };
      })
      .filter((headline): headline is YahooHeadline => headline !== null);
  } catch {
    return [];
  }
}

function formatHeadlineAge(publishedAt: string) {
  if (!publishedAt) return "Today";
  const date = new Date(publishedAt);
  if (Number.isNaN(date.getTime())) return "Today";

  const minutes = Math.max(1, Math.floor((Date.now() - date.getTime()) / 60000));
  if (minutes < 60) return `${minutes}m ago`;

  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;

  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

/* --------------------------- UI Primitives --------------------------- */

function Icon({ name }: { name: IconName }) {
  const path = ICON_PATHS[name];

  return (
    <svg className="sf-icon" viewBox="0 0 24 24" aria-hidden="true">
      <path d={path} fill="currentColor" />
    </svg>
  );
}

function Pill({ children, tone }: { children: React.ReactNode; tone?: "ok" | "warn" | "danger" | "muted" }) {
  return <AntTag className={cn("pill", tone && `pill--${tone}`)}>{children}</AntTag>;
}

function Button({
  children,
  variant = "primary",
  onClick,
  type = "button",
  disabled,
  className,
}: {
  children: React.ReactNode;
  variant?: "primary" | "ghost" | "subtle";
  onClick?: () => void;
  type?: "button" | "submit";
  disabled?: boolean;
  className?: string;
}) {
  const antType = variant === "primary" ? "primary" : variant === "subtle" ? "text" : "default";

  return (
    <AntButton
      className={cn("btn", `btn--${variant}`, className)}
      onClick={onClick}
      htmlType={type}
      disabled={disabled}
      type={antType}
    >
      {children}
    </AntButton>
  );
}

function Card({
  title,
  right,
  children,
  className,
}: {
  title: string;
  right?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <AntCard
      className={cn("card", className)}
      title={<h3 className="card__title">{title}</h3>}
      extra={<div className="card__right">{right}</div>}
      variant="outlined"
    >
      <div className="card__body">{children}</div>
    </AntCard>
  );
}

function Dock({ active, onPick }: { active: Workspace; onPick: (w: Workspace) => void }) {
  const items: Array<{ id: Workspace; label: string; icon: IconName }> = [
    { id: "command", label: "Command", icon: "command" },
    { id: "trades", label: "Trades", icon: "trades" },
    { id: "simulator", label: "Simulation", icon: "simulator" },
    { id: "replay", label: "Practice", icon: "replay" },
    { id: "pulse", label: "Bias Context", icon: "pulse" },
    { id: "coach", label: "Coach", icon: "coach" },
  ];

  return (
    <aside className="dock" aria-label="Workspace dock">
      {items.map((it) => (
        <AntButton
          key={it.id}
          className={cn("dock__btn", active === it.id && "dock__btn--active")}
          onClick={() => onPick(it.id)}
          title={it.label}
          type="text"
          icon={<Icon name={it.icon} />}
        >
          <span className="dock__label">{it.label}</span>
        </AntButton>
      ))}
    </aside>
  );
}

function TerminalBar({
  scoringMode,
  onScoringModeChange,
  apiBusy,
  onImport,
}: {
  scoringMode: ScoringMode;
  onScoringModeChange: (mode: ScoringMode) => void;
  apiBusy: boolean;
  onImport: () => void;
}) {
  const scoringSelectStyle = {
    ["--ant-select-background-color" as string]: "rgba(5,8,14,.9)",
    ["--ant-select-border-color" as string]: "rgba(255,255,255,.16)",
    ["--ant-select-color" as string]: "rgba(232,240,255,.94)",
  } as React.CSSProperties;

  return (
    <header className="topbar">
      <div className="brand">
        <div className="brand__mark" aria-hidden="true" />
        <div className="brand__txt">
          <div className="brand__name">Insid-ur Trading</div>
        </div>
      </div>

      <div className="topbar__right">
        <AntSelect
          className="topbar__select"
          classNames={{ popup: { root: "appSelectDropdown" } }}
          style={scoringSelectStyle}
          value={scoringMode}
          onChange={(value) => onScoringModeChange(value as ScoringMode)}
          aria-label="Scoring mode"
          disabled={apiBusy}
          options={[
            { value: "hybrid", label: "Hybrid" },
            { value: "rule_only", label: "Rule-only" },
          ]}
        />
        <Button onClick={onImport} disabled={apiBusy}>
          <span className="btn__icon">
            <Icon name="upload" />
          </span>
          Import CSV
        </Button>
        {apiBusy ? <Pill tone="muted">Syncing...</Pill> : null}
      </div>
    </header>
  );
}
/* --------------------------- CSV Upload Modal --------------------------- */

function CsvUploadModal({
  open,
  onClose,
  onImported,
}: {
  open: boolean;
  onClose: () => void;
  onImported: (payload: CsvImportPayload) => void;
}) {
  const [dragOver, setDragOver] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [err, setErr] = useState<string>("");
  const canImport = Boolean(selectedFile);

  const reset = () => {
    setDragOver(false);
    setSelectedFile(null);
    setErr("");
  };

  const close = () => {
    reset();
    onClose();
  };

  const onFilePick = (f: File | null) => {
    if (!f) return;
    if (!f.name.toLowerCase().endsWith(".csv")) {
      setErr("Please upload a .csv file.");
      return;
    }
    setErr("");
    setSelectedFile(f);
  };

  const onDrop = async (ev: React.DragEvent) => {
    ev.preventDefault();
    setDragOver(false);
    const f = ev.dataTransfer.files?.[0];
    onFilePick(f ?? null);
  };

  const onImport = () => {
    if (!selectedFile) return;
    onImported({ file: selectedFile });
    close();
  };

  if (!open) return null;
  const fileSizeMb = selectedFile ? (selectedFile.size / (1024 * 1024)).toFixed(2) : null;

  return (
    <div className="modalOverlay" role="dialog" aria-modal="true" aria-label="CSV import modal" onMouseDown={close}>
      <div className="modal" onMouseDown={(e) => e.stopPropagation()}>
        <div className="modalHdr">
          <div className="modalHdr__left">
            <div className="modalTitle">Import Trade Log (CSV)</div>
            <div className="modalSub muted">
              Upload your CSV. All parsing and analytics run on the backend.
            </div>
          </div>
          <AntButton className="modalClose" onClick={close} aria-label="Close" type="text" icon={<Icon name="close" />} />
        </div>

        <div className="modalBody">
          {/* Dropzone */}
          <div
            className={cn("dropzone", dragOver && "dropzone--over", selectedFile && "dropzone--loaded")}
            onDragEnter={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={(e) => {
              e.preventDefault();
              setDragOver(false);
            }}
            onDrop={onDrop}
          >
            <div className="dropzone__icon">
              <Icon name="upload" />
            </div>
            <div className="dropzone__text">
              <div className="dropzone__title">
                {selectedFile ? (
                  <>
                    Loaded <span className="mono">{selectedFile.name}</span>
                  </>
                ) : (
                  "Drag & drop your CSV here"
                )}
              </div>
              <div className="muted">
                {selectedFile && fileSizeMb ? `${fileSizeMb} MB selected` : "or choose a file from your computer"}
              </div>
            </div>

            <AntUpload
              accept=".csv,text/csv"
              showUploadList={false}
              beforeUpload={(file) => {
                onFilePick(file);
                return false;
              }}
            >
              <AntButton className="fileBtn">Browse CSV</AntButton>
            </AntUpload>
          </div>

          {err ? <AntAlert className="alert alert--danger" message={`Import error: ${err}`} type="error" showIcon /> : null}
        </div>

        <div className="modalFtr">
          <div className="modalFtr__left muted">
            {selectedFile ? (
              <>
                Ready: <b>Yes</b> - File: <b>{selectedFile.name}</b>
              </>
            ) : (
              "Upload a CSV to continue."
            )}
          </div>
          <div className="modalFtr__right">
            <Button variant="ghost" onClick={close} className="modalCancelBtn">
              Cancel
            </Button>
            <Button onClick={onImport} disabled={!canImport} className="modalImportBtn">
              Import
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

/* --------------------------- Workspace Views --------------------------- */

const DANGER_DAY_COUNT = 7;
const DANGER_HOUR_COUNT = 24;
const DANGER_HOURS = Array.from({ length: DANGER_HOUR_COUNT }, (_, hour) => hour.toString().padStart(2, "0"));
const MARKET_OPEN_HOUR = 9;
const MARKET_CLOSE_HOUR = 17;
const DANGER_MARKET_HOUR_INDEXES = Array.from({ length: DANGER_HOUR_COUNT }, (_, hour) => hour).filter(
  (hour) => hour >= MARKET_OPEN_HOUR && hour <= MARKET_CLOSE_HOUR
);
const DANGER_MARKET_HOURS = DANGER_MARKET_HOUR_INDEXES.map((hour) => ({
  hour,
  label: DANGER_HOURS[hour],
}));
const DANGER_DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const DANGER_MATRIX_SEED = [
  [0, 1, 0, 0, 2, 0, 0, 3, 2, 0, 0, 0],
  [1, 0, 0, 2, 0, 0, 0, 2, 3, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0],
  [0, 0, 0, 0, 1, 0, 0, 3, 0, 0, 0, 0],
  [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
  [0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0],
];
const DANGER_MATRIX = DANGER_MATRIX_SEED.map((row) => [...Array(9).fill(0), ...row, ...Array(3).fill(0)]);

function normalizeDangerHoursMatrix(matrix?: number[][]) {
  return Array.from({ length: DANGER_DAY_COUNT }, (_, weekday) =>
    Array.from({ length: DANGER_HOUR_COUNT }, (_, hour) => {
      const rawValue = matrix?.[weekday]?.[hour];
      if (!Number.isFinite(rawValue)) return 0;
      return Math.max(0, Math.round(rawValue as number));
    })
  );
}

function formatPnlCurrency(value: number) {
  const currency = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
  return value > 0 ? `+${currency.format(value)}` : currency.format(value);
}

function resolveDangerCellToneClass(value: number, maxValue: number) {
  if (value <= 0) return "";

  // Preserve old fixed bins for small integer data, and scale for larger backend counts.
  if (maxValue <= 4) {
    if (value === 1) return "dangerCell--cool";
    if (value === 2) return "dangerCell--warm";
    if (value === 3) return "dangerCell--hot";
    return "dangerCell--critical";
  }

  const ratio = value / maxValue;
  if (ratio >= 0.75) return "dangerCell--critical";
  if (ratio >= 0.5) return "dangerCell--hot";
  if (ratio >= 0.25) return "dangerCell--warm";
  return "dangerCell--cool";
}

const BIAS_LABELS: Record<BiasKey, string> = {
  overtrading: "Overtrading",
  loss_aversion: "Loss Aversion",
  revenge_trading: "Revenge Trading",
  recency_bias: "Recency Bias",
};

const BIAS_BAR_CLASS_BY_KEY: Record<BiasKey, string> = {
  overtrading: "barChart__bar--bias-overtrading",
  loss_aversion: "barChart__bar--bias-loss-aversion",
  revenge_trading: "barChart__bar--bias-revenge-trading",
  recency_bias: "barChart__bar--bias-recency-bias",
};

const BIAS_KEY_TO_CANONICAL: Record<BiasKey, CanonicalBiasType> = {
  overtrading: "OVERTRADING",
  loss_aversion: "LOSS_AVERSION",
  revenge_trading: "REVENGE_TRADING",
  recency_bias: "RECENCY_BIAS",
};

const CANONICAL_BIAS_LABELS: Record<CanonicalBiasType, string> = {
  OVERTRADING: "Overtrading",
  LOSS_AVERSION: "Loss Aversion",
  REVENGE_TRADING: "Revenge Trading",
  RECENCY_BIAS: "Recency Bias",
};

const EMOTIONAL_CHECKIN_QUESTIONS: Record<CanonicalBiasType, [string, string, string]> = {
  REVENGE_TRADING: [
    "After a loss, do you feel a strong urge to \u2018win it back\u2019 right away?",
    "When you feel frustrated or embarrassed by a trade, is it harder to pause before taking the next one?",
    "Do you tend to increase risk when you feel like you\u2019re \u2018behind\u2019 for the day?",
  ],
  LOSS_AVERSION: [
    "Do you avoid closing a losing trade because accepting the loss feels emotionally painful?",
    "When a trade goes against you, do you keep holding mainly because you hope it will turn around?",
    "Do you feel more regret from taking a loss than satisfaction from taking a similar-sized gain?",
  ],
  RECENCY_BIAS: [
    "After a recent win or loss streak, do you feel unusually confident or unusually doubtful about the next trade?",
    "Do your last few trades strongly influence how you size positions right now?",
    "When the market just moved, do you feel pressure to act quickly so you don\u2019t miss out?",
  ],
  OVERTRADING: [
    "When you\u2019re bored, anxious, or restless, do you trade just to feel engaged?",
    "Do you feel uneasy when you\u2019re not in a trade, like you\u2019re missing opportunities?",
    "On stressful days, do you notice you take more trades than you planned, even without clear setups?",
  ],
};

function createEmptyCheckinAnswerMap(): Record<CanonicalBiasType, EmotionalCheckInAnswer[]> {
  return {
    REVENGE_TRADING: [null, null, null],
    LOSS_AVERSION: [null, null, null],
    RECENCY_BIAS: [null, null, null],
    OVERTRADING: [null, null, null],
  };
}

type TopBiasEntry = {
  biasType: CanonicalBiasType;
  score: number;
};

function getTopTwoBiasesFromScores(
  scores?: AnalysisOutput["bias_scores"]
): TopBiasEntry[] {
  if (!scores) return [];

  return (Object.entries(scores) as Array<[BiasKey, number]>)
    .map(([biasKey, rawScore]) => ({
      biasType: BIAS_KEY_TO_CANONICAL[biasKey],
      score: Number.isFinite(rawScore) ? Math.round(rawScore) : 0,
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 2);
}

function getEmotionalCheckinStorageKey(sessionId: string, biasType: CanonicalBiasType) {
  return `emotional-checkin:${sessionId}:${biasType}`;
}

function readStoredBiasAnswers(sessionId: string, biasType: CanonicalBiasType): EmotionalCheckInAnswer[] {
  try {
    const raw = window.localStorage.getItem(getEmotionalCheckinStorageKey(sessionId, biasType));
    if (!raw) return [null, null, null];
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return [null, null, null];

    return [0, 1, 2].map((idx) => {
      const value = parsed[idx];
      return value === "YES" || value === "NO" ? value : null;
    });
  } catch {
    return [null, null, null];
  }
}

function writeStoredBiasAnswers(
  sessionId: string,
  biasType: CanonicalBiasType,
  answers: EmotionalCheckInAnswer[]
) {
  try {
    window.localStorage.setItem(getEmotionalCheckinStorageKey(sessionId, biasType), JSON.stringify(answers));
  } catch {
    // Ignore localStorage failures (private mode/storage quota)
  }
}

function createBiasCounter(): Record<BiasKey, number> {
  return {
    overtrading: 0,
    loss_aversion: 0,
    revenge_trading: 0,
    recency_bias: 0,
  };
}

function resolveDominantBias(counter: Record<BiasKey, number>): { bias: BiasKey; count: number } | null {
  const entries = Object.entries(counter) as Array<[BiasKey, number]>;
  let dominant: { bias: BiasKey; count: number } | null = null;

  for (const [bias, count] of entries) {
    if (count <= 0) continue;
    if (!dominant || count > dominant.count) dominant = { bias, count };
  }

  return dominant;
}

function resolveDominantBiasFromScores(
  scores?: AnalysisOutput["bias_scores"]
): { bias: BiasKey; score: number } | null {
  if (!scores) return null;

  const entries = Object.entries(scores) as Array<[BiasKey, number]>;
  let dominant: { bias: BiasKey; score: number } | null = null;

  for (const [bias, score] of entries) {
    if (!Number.isFinite(score)) continue;
    if (!dominant || score > dominant.score) dominant = { bias, score };
  }

  return dominant;
}

function parseFlaggedTradeTimestamp(timestamp?: string): { weekday: number | null; hour: number } | null {
  if (!timestamp) return null;
  const raw = timestamp.trim();
  if (!raw) return null;

  const normalized = raw.includes("T") ? raw : raw.replace(" ", "T");
  const parsed = new Date(normalized);
  if (!Number.isNaN(parsed.getTime())) {
    return {
      weekday: (parsed.getDay() + 6) % 7,
      hour: parsed.getHours(),
    };
  }

  const timeMatch = raw.match(/^(\d{1,2}):(\d{2})(?::(\d{2}))?$/);
  if (!timeMatch) return null;
  const hour = Number(timeMatch[1]);
  if (!Number.isFinite(hour) || hour < 0 || hour >= DANGER_HOUR_COUNT) return null;
  return { weekday: null, hour };
}

function resolveDangerRiskLabel(value: number, maxValue: number) {
  if (value <= 0) return "Low";
  const toneClass = resolveDangerCellToneClass(value, maxValue);
  if (toneClass === "dangerCell--critical") return "Critical";
  if (toneClass === "dangerCell--hot") return "High";
  if (toneClass === "dangerCell--warm") return "Elevated";
  return "Moderate";
}

type ChartGranularity = "1m" | "5m" | "15m" | "1h" | "1d";

const CHART_GRANULARITY_OPTIONS: Array<{ label: string; value: ChartGranularity }> = [
  { label: "1m", value: "1m" },
  { label: "5m", value: "5m" },
  { label: "15m", value: "15m" },
  { label: "1h", value: "1h" },
  { label: "1d", value: "1d" },
];
const SNAPSHOT_VISIBLE_BUCKETS = 60;
const SNAPSHOT_LOOKBACK_MULTIPLIER = 2;

type TimelineTradePoint = {
  timestamp: string;
  pnl: number;
  tradeCount?: number;
  wins?: number;
  losses?: number;
  flat?: number;
};

type SnapshotBucketSeed = {
  key: string;
  label: string;
  trades: number;
  pnl: number;
  wins: number;
  losses: number;
  flat: number;
  flaggedTrades: number;
  confidenceSum: number;
  biasCounters: Record<BiasKey, number>;
};

type SnapshotBucket = {
  key: string;
  label: string;
  trades: number;
  pnl: number;
  cumulativePnl: number;
  wins: number;
  losses: number;
  flat: number;
  flaggedTrades: number;
  confidenceSum: number;
  avgConfidence: number;
  weightedConfidenceTrades: number;
  winRate: number;
  biasIncidents: number;
  dominantBias: BiasKey | null;
  dominantBiasCount: number;
};

type SnapshotBarDatum = {
  key: string;
  label: string;
  value: number;
  tone?: "neutral" | "positive" | "negative" | "warning";
  barClassName?: string;
  tooltip: React.ReactNode;
};

type ChartLegendItem = {
  key: string;
  label: string;
  barClassName: string;
};

function normalizeConfidencePercent(confidence: unknown): number {
  if (typeof confidence !== "number" || !Number.isFinite(confidence)) return 0;
  const normalized = confidence >= 0 && confidence <= 1 ? confidence * 100 : confidence;
  return Math.max(0, Math.min(100, normalized));
}

function parseTimestampToDate(timestamp?: string): Date | null {
  if (!timestamp) return null;
  const raw = timestamp.trim();
  if (!raw) return null;
  const normalized = raw.includes("T") ? raw : raw.replace(" ", "T");
  const parsed = new Date(normalized);
  if (!Number.isNaN(parsed.getTime())) return parsed;

  const timeMatch = raw.match(/^(\d{1,2}):(\d{2})(?::(\d{2}))?$/);
  if (!timeMatch) return null;
  const hours = Number(timeMatch[1]);
  const minutes = Number(timeMatch[2]);
  const seconds = Number(timeMatch[3] ?? "0");
  if (!Number.isFinite(hours) || !Number.isFinite(minutes) || !Number.isFinite(seconds)) return null;
  if (hours < 0 || hours > 23 || minutes < 0 || minutes > 59 || seconds < 0 || seconds > 59) return null;
  const fallback = new Date();
  fallback.setHours(hours, minutes, seconds, 0);
  return fallback;
}

function detectDefaultChartGranularity(
  tradeTimeline?: AnalysisOutput["trade_timeline"],
  flaggedTrades?: AnalysisOutput["flagged_trades"],
  dailyPnl?: AnalysisOutput["daily_pnl"]
): ChartGranularity {
  const timelineDates = (tradeTimeline ?? [])
    .map((point) => parseTimestampToDate(point.timestamp))
    .filter((date): date is Date => date !== null);

  const fallbackDates = (flaggedTrades ?? [])
    .map((trade) => parseTimestampToDate(trade.timestamp))
    .filter((date): date is Date => date !== null);

  const dailyDates = (dailyPnl ?? [])
    .map((point) => parseTimestampToDate(`${point.date}T12:00:00`))
    .filter((date): date is Date => date !== null);

  const dates = timelineDates.length > 0 ? timelineDates : fallbackDates.length > 0 ? fallbackDates : dailyDates;
  if (dates.length < 2) return "1d";

  const sorted = [...dates].sort((a, b) => a.getTime() - b.getTime());
  const spanMs = sorted[sorted.length - 1].getTime() - sorted[0].getTime();
  const spanHours = spanMs / (1000 * 60 * 60);
  if (spanHours <= 2) return "1m";
  if (spanHours <= 8) return "5m";
  if (spanHours <= 24) return "15m";
  if (spanHours <= 24 * 7) return "1h";
  return "1d";
}

function resolveBucketStepMinutes(granularity: ChartGranularity) {
  if (granularity === "1m") return 1;
  if (granularity === "5m") return 5;
  if (granularity === "15m") return 15;
  if (granularity === "1h") return 60;
  return 0;
}

function resolveBucketStepMs(granularity: ChartGranularity) {
  if (granularity === "1d") return 24 * 60 * 60 * 1000;
  return resolveBucketStepMinutes(granularity) * 60 * 1000;
}

function getSparkBucketKey(date: Date, granularity: ChartGranularity): string {
  const yyyy = date.getFullYear();
  const mm = String(date.getMonth() + 1).padStart(2, "0");
  const dd = String(date.getDate()).padStart(2, "0");
  if (granularity === "1d") return `${yyyy}-${mm}-${dd}`;

  const step = resolveBucketStepMinutes(granularity);
  const hh = String(date.getHours()).padStart(2, "0");
  const bucketMinute = Math.floor(date.getMinutes() / step) * step;
  const minute = String(bucketMinute).padStart(2, "0");
  return `${yyyy}-${mm}-${dd} ${hh}:${minute}`;
}

function formatSnapshotBucketLabel(key: string, granularity: ChartGranularity): string {
  if (granularity === "1d") return key;
  const parts = key.split(" ");
  return parts.length > 1 ? parts[1] : key;
}

function buildTimelineFromAnalysis(
  tradeTimeline?: AnalysisOutput["trade_timeline"],
  flaggedTrades?: AnalysisOutput["flagged_trades"],
  dailyPnl?: AnalysisOutput["daily_pnl"]
): TimelineTradePoint[] {
  const directTimeline = (tradeTimeline ?? [])
    .filter((point) => typeof point.timestamp === "string" && Number.isFinite(point.pnl))
    .map((point) => ({
      timestamp: point.timestamp,
      pnl: Number(point.pnl),
      tradeCount:
        typeof point.trade_count === "number" && Number.isFinite(point.trade_count)
          ? Math.max(0, Math.round(point.trade_count))
          : undefined,
      wins: typeof point.wins === "number" && Number.isFinite(point.wins) ? Math.max(0, Math.round(point.wins)) : undefined,
      losses:
        typeof point.losses === "number" && Number.isFinite(point.losses)
          ? Math.max(0, Math.round(point.losses))
          : undefined,
      flat: typeof point.flat === "number" && Number.isFinite(point.flat) ? Math.max(0, Math.round(point.flat)) : undefined,
    }));

  if (directTimeline.length > 0) {
    // Timeline is already sorted when normalized in the API client.
    return directTimeline;
  }

  const fromFlaggedTrades = (flaggedTrades ?? [])
    .filter((trade) => typeof trade.timestamp === "string" && Number.isFinite(trade.profit_loss))
    .map((trade) => ({ timestamp: trade.timestamp!, pnl: Number(trade.profit_loss ?? 0), tradeCount: 1 }))
    .sort((a, b) => a.timestamp.localeCompare(b.timestamp));
  if (fromFlaggedTrades.length > 0) return fromFlaggedTrades;

  return (dailyPnl ?? [])
    .map((point) => ({ timestamp: `${point.date}T12:00:00`, pnl: Number(point.pnl), tradeCount: 1 }))
    .sort((a, b) => a.timestamp.localeCompare(b.timestamp));
}

type SelectedTimelineWindow = {
  points: TimelineTradePoint[];
  cutoffMs: number | null;
};

function selectRecentTimelineWindow(
  source: TimelineTradePoint[],
  granularity: ChartGranularity,
  maxBuckets: number,
  lookbackMultiplier: number
): SelectedTimelineWindow {
  if (source.length === 0) return { points: source, cutoffMs: null };

  const safeBuckets = Math.max(1, Math.floor(maxBuckets));
  const safeMultiplier = Math.max(1, lookbackMultiplier);
  const lookbackMs = resolveBucketStepMs(granularity) * safeBuckets * safeMultiplier;

  let latestMs: number | null = null;
  for (let index = source.length - 1; index >= 0; index -= 1) {
    const parsed = parseTimestampToDate(source[index].timestamp);
    if (!parsed) continue;
    latestMs = parsed.getTime();
    break;
  }

  if (latestMs === null) {
    return { points: source.slice(-safeBuckets), cutoffMs: null };
  }

  const cutoffMs = latestMs - lookbackMs;
  const selected: TimelineTradePoint[] = [];

  for (let index = source.length - 1; index >= 0; index -= 1) {
    const point = source[index];
    const parsed = parseTimestampToDate(point.timestamp);
    if (!parsed) continue;
    if (parsed.getTime() < cutoffMs) break;
    selected.push(point);
  }

  if (selected.length === 0) {
    return { points: source.slice(-safeBuckets), cutoffMs };
  }

  selected.reverse();
  return { points: selected, cutoffMs };
}

function buildSnapshotBuckets(
  tradeTimeline?: AnalysisOutput["trade_timeline"],
  flaggedTrades?: AnalysisOutput["flagged_trades"],
  dailyPnl?: AnalysisOutput["daily_pnl"],
  granularity: ChartGranularity = "1h",
  maxBuckets: number = SNAPSHOT_VISIBLE_BUCKETS,
  lookbackMultiplier: number = SNAPSHOT_LOOKBACK_MULTIPLIER
) : SnapshotBucket[] {
  const rawSource = buildTimelineFromAnalysis(tradeTimeline, flaggedTrades, dailyPnl);
  const selectedWindow = selectRecentTimelineWindow(rawSource, granularity, maxBuckets, lookbackMultiplier);
  const source = selectedWindow.points;
  const cutoffMs = selectedWindow.cutoffMs;
  const bucketMap = new Map<string, SnapshotBucketSeed>();
  let fallbackTimelineIndex = 0;
  let fallbackFlaggedIndex = 0;

  const ensureBucket = (key: string): SnapshotBucketSeed => {
    const existing = bucketMap.get(key);
    if (existing) return existing;
    const seed: SnapshotBucketSeed = {
      key,
      label: formatSnapshotBucketLabel(key, granularity),
      trades: 0,
      pnl: 0,
      wins: 0,
      losses: 0,
      flat: 0,
      flaggedTrades: 0,
      confidenceSum: 0,
      biasCounters: createBiasCounter(),
    };
    bucketMap.set(key, seed);
    return seed;
  };

  for (const point of source) {
    const parsed = parseTimestampToDate(point.timestamp);
    const key = parsed
      ? getSparkBucketKey(parsed, granularity)
      : `zz-unknown-t-${String(fallbackTimelineIndex++).padStart(5, "0")}`;
    const bucket = ensureBucket(key);
    const tradeCount = Math.max(1, Math.round(point.tradeCount ?? 1));
    let winCount = point.wins ?? 0;
    let lossCount = point.losses ?? 0;
    let flatCount = point.flat ?? 0;
    const hasDetailedCounts = point.wins != null || point.losses != null || point.flat != null;
    if (!hasDetailedCounts) {
      winCount = point.pnl > 0 ? tradeCount : 0;
      lossCount = point.pnl < 0 ? tradeCount : 0;
      flatCount = point.pnl === 0 ? tradeCount : 0;
    } else {
      winCount = Math.max(0, Math.round(winCount));
      lossCount = Math.max(0, Math.round(lossCount));
      flatCount = Math.max(0, Math.round(flatCount));
      const known = winCount + lossCount + flatCount;
      if (known < tradeCount) {
        flatCount += tradeCount - known;
      }
    }
    bucket.trades += tradeCount;
    bucket.pnl += Number(point.pnl);
    bucket.wins += winCount;
    bucket.losses += lossCount;
    bucket.flat += flatCount;
  }

  for (const trade of flaggedTrades ?? []) {
    const parsed = parseTimestampToDate(trade.timestamp);
    if (cutoffMs !== null && parsed && parsed.getTime() < cutoffMs) continue;
    const key = parsed
      ? getSparkBucketKey(parsed, granularity)
      : `zz-unknown-f-${String(fallbackFlaggedIndex++).padStart(5, "0")}`;
    const bucket = ensureBucket(key);
    const normalizedConfidence = normalizeConfidencePercent(trade.confidence);
    bucket.flaggedTrades += 1;
    bucket.confidenceSum += normalizedConfidence;

    const biasTypes = Array.isArray(trade.bias_types) && trade.bias_types.length > 0 ? trade.bias_types : [trade.bias];
    for (const biasType of biasTypes) {
      if (!biasType || !(biasType in bucket.biasCounters)) continue;
      bucket.biasCounters[biasType as BiasKey] += 1;
    }
  }

  const ordered = [...bucketMap.values()].sort((a, b) => a.key.localeCompare(b.key));
  const boundedOrdered = ordered.length > maxBuckets ? ordered.slice(-maxBuckets) : ordered;
  const output: SnapshotBucket[] = [];
  let cumulativePnl = 0;

  for (const bucket of boundedOrdered) {
    cumulativePnl += bucket.pnl;
    const avgConfidence = bucket.flaggedTrades > 0 ? bucket.confidenceSum / bucket.flaggedTrades : 0;
    const biasIncidents = Object.values(bucket.biasCounters).reduce((sum, value) => sum + value, 0);
    const dominantBias = resolveDominantBias(bucket.biasCounters);

    output.push({
      key: bucket.key,
      label: bucket.label,
      trades: bucket.trades,
      pnl: Number(bucket.pnl.toFixed(2)),
      cumulativePnl: Number(cumulativePnl.toFixed(2)),
      wins: bucket.wins,
      losses: bucket.losses,
      flat: bucket.flat,
      flaggedTrades: bucket.flaggedTrades,
      confidenceSum: Number(bucket.confidenceSum.toFixed(2)),
      avgConfidence: Number(avgConfidence.toFixed(2)),
      weightedConfidenceTrades: Number((bucket.trades * (avgConfidence / 100)).toFixed(2)),
      winRate: bucket.trades > 0 ? Number(((bucket.wins / bucket.trades) * 100).toFixed(2)) : 0,
      biasIncidents,
      dominantBias: dominantBias?.bias ?? null,
      dominantBiasCount: dominantBias?.count ?? 0,
    });
  }

  return output;
}


function formatCompactUsd(value: number) {
  if (!Number.isFinite(value)) return "$0";
  const abs = Math.abs(value);
  const sign = value < 0 ? "-" : "";
  if (abs >= 1000) return `${sign}$${(abs / 1000).toFixed(2)}k`;
  if (abs >= 100) return `${sign}$${abs.toFixed(0)}`;
  return `${sign}$${abs.toFixed(2)}`;
}

function formatChartAxisValue(value: number) {
  if (!Number.isFinite(value)) return "0";
  const abs = Math.abs(value);
  const sign = value < 0 ? "-" : "";
  if (abs >= 1000) {
    const compact = abs >= 10000 ? (abs / 1000).toFixed(0) : (abs / 1000).toFixed(1);
    return `${sign}${compact}k`;
  }
  if (abs >= 100) return `${sign}${abs.toFixed(0)}`;
  if (abs >= 10) return `${sign}${abs.toFixed(1)}`;
  if (abs >= 1) return `${sign}${abs.toFixed(2)}`;
  return `${sign}${abs.toFixed(3)}`;
}


function SnapshotTooltipContent({
  title,
  rows,
}: {
  title: string;
  rows: Array<{ label: string; value: string; tone?: "pos" | "neg" | "warn" }>;
}) {
  return (
    <div className="barTooltipCard">
      <div className="barTooltipCard__title mono">{title}</div>
      {rows.map((row, index) => (
        <div key={`${row.label}-${index}`} className="barTooltipCard__row">
          <span>{row.label}</span>
          <span
            className={cn(
              "barTooltipCard__value mono",
              row.tone === "pos" && "pos",
              row.tone === "neg" && "neg",
              row.tone === "warn" && "barTooltipCard__value--warn"
            )}
          >
            {row.value}
          </span>
        </div>
      ))}
    </div>
  );
}

function BarChartCard({
  title,
  value,
  bars,
  legend,
  valueTone,
  size = "compact",
}: {
  title: string;
  value: string;
  bars: SnapshotBarDatum[];
  legend?: ChartLegendItem[];
  valueTone?: "pos" | "neg";
  size?: "compact" | "large";
}) {
  const fallbackBars: SnapshotBarDatum[] =
    bars.length > 0
      ? bars
      : [
          {
            key: "empty",
            label: "N/A",
            value: 0,
            tone: "neutral",
            tooltip: <SnapshotTooltipContent title="No Data" rows={[{ label: "Status", value: "Upload trades to populate charts." }]} />,
          },
        ];
  const maxAbs = Math.max(...fallbackBars.map((bar) => Math.abs(bar.value)), 1);
  const hasNegative = fallbackBars.some((bar) => bar.value < 0);
  const firstLabel = fallbackBars[0]?.label ?? "";
  const lastLabel = fallbackBars[fallbackBars.length - 1]?.label ?? "";
  const chartHeight = size === "large" ? 236 : 132;
  const slotWidth = size === "large" ? 14 : 10;
  const slotGap = size === "large" ? 4 : 3;
  const trackMinWidth = Math.max(320, fallbackBars.length * (slotWidth + slotGap) + 24);
  const halfTrackHeight = Math.max(40, chartHeight / 2 - 8);
  const fullTrackHeight = Math.max(72, chartHeight - 12);
  const scaleMax = Math.max(maxAbs * 1.08, 1);
  const yAxisTicks = hasNegative ? [maxAbs, 0, -maxAbs] : [maxAbs, maxAbs / 2, 0];

  return (
    <div className={cn("sparkCard barCard", size === "large" && "barCard--large")}>
      <div className="sparkCard__meta">
        <span>{title}</span>
        <span className={cn("mono", valueTone)}>{value}</span>
      </div>

      <div className="barChartFrame">
        <div className="barChart__yAxis mono muted">
          {yAxisTicks.map((tick, index) => (
            <span key={`${tick}-${index}`} className="barChart__yTick">
              {formatChartAxisValue(tick)}
            </span>
          ))}
        </div>
        <div className="barChartScroller">
          <div
            className={cn("barChart", hasNegative && "barChart--centered")}
            style={{ height: chartHeight, width: "100%", minWidth: trackMinWidth }}
          >
            {hasNegative ? <span className="barChart__zero" /> : null}
            {fallbackBars.map((bar) => {
              const safeValue = Number.isFinite(bar.value) ? bar.value : 0;
              const ratio = Math.min(Math.abs(safeValue) / scaleMax, 1);
              const normalizedHeight = Math.max((hasNegative ? halfTrackHeight : fullTrackHeight) * ratio, safeValue === 0 ? 2 : 0);
              const toneClass = bar.barClassName
                ? bar.barClassName
                : bar.tone === "neutral"
                  ? "barChart__bar--neutral"
                  : bar.tone === "warning"
                    ? "barChart__bar--warn"
                    : bar.tone === "negative"
                      ? "barChart__bar--neg"
                      : bar.tone === "positive"
                        ? "barChart__bar--pos"
                        : safeValue < 0
                          ? "barChart__bar--neg"
                          : "barChart__bar--pos";
              const barStyle = hasNegative
                ? safeValue >= 0
                  ? ({ height: `${normalizedHeight}px`, bottom: "50%" } as React.CSSProperties)
                  : ({ height: `${normalizedHeight}px`, top: "50%" } as React.CSSProperties)
                : ({ height: `${normalizedHeight}px`, bottom: "6px" } as React.CSSProperties);

              return (
                <AntTooltip
                  key={bar.key}
                  classNames={{ root: "barChartTooltip" }}
                  title={bar.tooltip}
                  placement="top"
                >
                  <div
                    className="barChart__slot"
                    style={{ width: slotWidth, minWidth: slotWidth, marginRight: slotGap }}
                  >
                    <span className={cn("barChart__bar", toneClass)} style={barStyle} />
                  </div>
                </AntTooltip>
              );
            })}
          </div>
        </div>
      </div>

      <div className="barChart__axis muted mono">
        <span>{fallbackBars.length} buckets</span>
        <span>
          {firstLabel} - {lastLabel}
        </span>
      </div>

      {legend && legend.length > 0 ? (
        <div className="barChartLegend">
          {legend.map((item) => (
            <span key={item.key} className="barChartLegend__item">
              <span className={cn("barChartLegend__swatch", item.barClassName)} />
              <span className="barChartLegend__label">{item.label}</span>
            </span>
          ))}
        </div>
      ) : null}
    </div>
  );
}

type SnapshotChartCardModel = {
  id: string;
  title: string;
  value: string;
  valueTone?: "pos" | "neg";
  bars: SnapshotBarDatum[];
  legend?: ChartLegendItem[];
};

type SnapshotChartBundle = {
  cards: SnapshotChartCardModel[];
};

type SnapshotAnalysisLike = {
  trade_timeline?: AnalysisOutput["trade_timeline"];
  flagged_trades?: AnalysisOutput["flagged_trades"];
  daily_pnl?: AnalysisOutput["daily_pnl"];
  summary?: AnalysisOutput["summary"];
};

function buildSnapshotChartBundle(
  analysis: SnapshotAnalysisLike | null | undefined,
  granularity: ChartGranularity,
  maxBuckets: number = SNAPSHOT_VISIBLE_BUCKETS
): SnapshotChartBundle {
  const snapshotBuckets = buildSnapshotBuckets(
    analysis?.trade_timeline,
    analysis?.flagged_trades,
    analysis?.daily_pnl,
    granularity,
    maxBuckets,
  );
  const summaryTotalTrades = analysis?.summary?.total_trades;
  const summaryTotalPnl = analysis?.summary?.total_pnl;
  const totalTradesFromBuckets = snapshotBuckets.reduce((sum, bucket) => sum + bucket.trades, 0);
  const totalPnlFromBuckets = snapshotBuckets.reduce((sum, bucket) => sum + bucket.pnl, 0);
  const totalWins = snapshotBuckets.reduce((sum, bucket) => sum + bucket.wins, 0);
  const totalBiasIncidents = snapshotBuckets.reduce((sum, bucket) => sum + bucket.biasIncidents, 0);
  const totalTrades =
    typeof summaryTotalTrades === "number" && Number.isFinite(summaryTotalTrades)
      ? Math.round(summaryTotalTrades)
      : totalTradesFromBuckets;
  const realizedPnlValue =
    typeof summaryTotalPnl === "number" && Number.isFinite(summaryTotalPnl)
      ? summaryTotalPnl
      : totalPnlFromBuckets;
  const summaryWinRate = analysis?.summary?.win_rate;
  const normalizedSummaryWinRate =
    typeof summaryWinRate === "number" && Number.isFinite(summaryWinRate)
      ? summaryWinRate <= 1
        ? summaryWinRate * 100
        : summaryWinRate
      : null;
  const overallWinRate = normalizedSummaryWinRate ?? (totalTrades > 0 ? (totalWins / totalTrades) * 100 : 0);

  const tradeFrequencyBars: SnapshotBarDatum[] = snapshotBuckets.map((bucket) => ({
    key: `freq-${bucket.key}`,
    label: bucket.label,
    value: bucket.trades,
    tone: "neutral",
    tooltip: (
      <SnapshotTooltipContent
        title={bucket.key}
        rows={[
          { label: "Trades", value: bucket.trades.toLocaleString() },
          { label: "Period PnL", value: formatPnlCurrency(bucket.pnl), tone: bucket.pnl >= 0 ? "pos" : "neg" },
          { label: "Wins / Losses", value: `${bucket.wins} / ${bucket.losses}` },
          {
            label: "Avg confidence",
            value: bucket.flaggedTrades > 0 ? `${bucket.avgConfidence.toFixed(1)}%` : "No flags",
            tone: bucket.flaggedTrades > 0 ? undefined : "warn",
          },
        ]}
      />
    ),
  }));

  const realizedPnlBars: SnapshotBarDatum[] = snapshotBuckets.map((bucket) => ({
    key: `pnl-${bucket.key}`,
    label: bucket.label,
    value: bucket.pnl,
    tone: bucket.pnl >= 0 ? "positive" : "negative",
    tooltip: (
      <SnapshotTooltipContent
        title={bucket.key}
        rows={[
          { label: "Period PnL", value: formatPnlCurrency(bucket.pnl), tone: bucket.pnl >= 0 ? "pos" : "neg" },
          {
            label: "Cumulative PnL",
            value: formatPnlCurrency(bucket.cumulativePnl),
            tone: bucket.cumulativePnl >= 0 ? "pos" : "neg",
          },
          { label: "Trades", value: bucket.trades.toLocaleString() },
          { label: "Wins / Losses", value: `${bucket.wins} / ${bucket.losses}` },
        ]}
      />
    ),
  }));

  const winRateBars: SnapshotBarDatum[] = snapshotBuckets.map((bucket) => ({
    key: `winrate-${bucket.key}`,
    label: bucket.label,
    value: bucket.winRate,
    tone: bucket.winRate >= 50 ? "positive" : "negative",
    tooltip: (
      <SnapshotTooltipContent
        title={bucket.key}
        rows={[
          { label: "Win rate", value: `${bucket.winRate.toFixed(1)}%`, tone: bucket.winRate >= 50 ? "pos" : "neg" },
          { label: "Wins", value: bucket.wins.toLocaleString() },
          { label: "Losses", value: bucket.losses.toLocaleString() },
          { label: "PnL", value: formatPnlCurrency(bucket.pnl), tone: bucket.pnl >= 0 ? "pos" : "neg" },
        ]}
      />
    ),
  }));

  const biasIncidentsBars: SnapshotBarDatum[] = snapshotBuckets.map((bucket) => {
    const dominantLabel = bucket.dominantBias ? BIAS_LABELS[bucket.dominantBias] : "None";
    const dominantShare =
      bucket.biasIncidents > 0 ? `${((bucket.dominantBiasCount / bucket.biasIncidents) * 100).toFixed(1)}%` : "0%";
    return {
      key: `bias-${bucket.key}`,
      label: bucket.label,
      value: bucket.biasIncidents,
      tone: bucket.biasIncidents > 0 ? undefined : "neutral",
      barClassName:
        bucket.biasIncidents > 0 && bucket.dominantBias
          ? BIAS_BAR_CLASS_BY_KEY[bucket.dominantBias]
          : "barChart__bar--neutral",
      tooltip: (
        <SnapshotTooltipContent
          title={bucket.key}
          rows={[
            { label: "Bias incidents", value: bucket.biasIncidents.toLocaleString() },
            { label: "Dominant bias", value: dominantLabel, tone: bucket.dominantBias ? "warn" : undefined },
            { label: "Dominant share", value: dominantShare },
            { label: "Trades", value: bucket.trades.toLocaleString() },
          ]}
        />
      ),
    };
  });

  const biasLegend: ChartLegendItem[] = (Object.keys(BIAS_LABELS) as BiasKey[]).map((biasKey) => ({
    key: `legend-${biasKey}`,
    label: BIAS_LABELS[biasKey],
    barClassName: BIAS_BAR_CLASS_BY_KEY[biasKey],
  }));

  return {
    cards: [
      {
        id: "trade_frequency",
        title: "Trade Frequency",
        value: `${totalTrades.toLocaleString()} total`,
        bars: tradeFrequencyBars,
      },
      {
        id: "realized_pnl",
        title: "Realized PnL",
        value: formatCompactUsd(realizedPnlValue),
        valueTone: realizedPnlValue >= 0 ? "pos" : "neg",
        bars: realizedPnlBars,
      },
      {
        id: "win_rate",
        title: "Win Rate",
        value: `${overallWinRate.toFixed(1)}%`,
        valueTone: overallWinRate >= 50 ? "pos" : "neg",
        bars: winRateBars,
      },
      {
        id: "bias_incidents",
        title: "Bias Incidents",
        value: `${totalBiasIncidents.toLocaleString()} tags`,
        bars: biasIncidentsBars,
        legend: biasLegend,
      },
    ],
  };
}

const COMMAND_SNAPSHOT_CARD_IDS = [
  "realized_pnl",
  "trade_frequency",
  "win_rate",
  "bias_incidents",
] as const;

function buildCommandSnapshotCards(
  analysis: SnapshotAnalysisLike | null | undefined,
  granularity: ChartGranularity
): SnapshotChartCardModel[] {
  const cards = buildSnapshotChartBundle(analysis, granularity, SNAPSHOT_VISIBLE_BUCKETS).cards;
  const cardsById = new Map(cards.map((card) => [card.id, card] as const));

  return COMMAND_SNAPSHOT_CARD_IDS.map((id) => {
    const card = cardsById.get(id);
    if (!card) return null;
    return {
      ...card,
      bars: card.bars.slice(-SNAPSHOT_VISIBLE_BUCKETS),
    };
  }).filter((card): card is SnapshotChartCardModel => card !== null);
}

type PnlSeriesPoint = {
  key: string;
  label: string;
  periodPnl: number;
  cumulativePnl: number;
};

function buildYearMonthSeries(
  year: number,
  monthPnlByKey: Map<string, number>
): PnlSeriesPoint[] {
  const yearStart = startOfYear(new Date(year, 0, 1));
  const yearEnd = endOfYear(yearStart);
  let cumulative = 0;

  return Array.from({ length: 12 }, (_, monthIndex) => {
    const monthDate = new Date(year, monthIndex, 1);
    if (monthDate < yearStart || monthDate > yearEnd) {
      return {
        key: `${year}-${String(monthIndex + 1).padStart(2, "0")}`,
        label: formatDate(monthDate, "MMM"),
        periodPnl: 0,
        cumulativePnl: Number(cumulative.toFixed(2)),
      };
    }

    const monthKey = `${year}-${String(monthIndex + 1).padStart(2, "0")}`;
    const periodPnl = Number((monthPnlByKey.get(monthKey) ?? 0).toFixed(2));
    cumulative += periodPnl;
    return {
      key: monthKey,
      label: formatDate(monthDate, "MMM"),
      periodPnl,
      cumulativePnl: Number(cumulative.toFixed(2)),
    };
  });
}

function PnlChartCard({
  card,
  analysis,
}: {
  card: SnapshotChartCardModel;
  analysis?: AnalysisOutput | null;
}) {
  const [viewMode, setViewMode] = useState<"granularity" | "calendar">("granularity");
  const [calendarMode, setCalendarMode] = useState<PnlCalendarMode>("month");

  const timelinePoints = useMemo(
    () => buildTimelineFromAnalysis(analysis?.trade_timeline, analysis?.flagged_trades, analysis?.daily_pnl),
    [analysis?.trade_timeline, analysis?.flagged_trades, analysis?.daily_pnl]
  );

  const latestParsedTimestamp = useMemo(() => {
    for (let index = timelinePoints.length - 1; index >= 0; index -= 1) {
      const raw = timelinePoints[index]?.timestamp;
      if (!raw) continue;
      const normalized = raw.includes("T") ? raw : raw.replace(" ", "T");
      const parsedFromIso = parseISO(normalized);
      if (!Number.isNaN(parsedFromIso.getTime())) return parsedFromIso;
      const fallback = parseTimestampToDate(raw);
      if (fallback) return fallback;
    }
    return new Date();
  }, [timelinePoints]);

  const [calendarValue, setCalendarValue] = useState<PnlCalendarValue>(() => ({
    year: latestParsedTimestamp.getFullYear(),
    month: latestParsedTimestamp.getMonth(),
  }));

  useEffect(() => {
    setCalendarValue({
      year: latestParsedTimestamp.getFullYear(),
      month: latestParsedTimestamp.getMonth(),
    });
  }, [latestParsedTimestamp]);

  const dailyPnlByDateKey = useMemo(() => {
    const map = new Map<string, number>();
    for (const point of timelinePoints) {
      const rawTimestamp = point.timestamp;
      const normalized = rawTimestamp.includes("T") ? rawTimestamp : rawTimestamp.replace(" ", "T");
      const parsedFromIso = parseISO(normalized);
      const parsed = !Number.isNaN(parsedFromIso.getTime()) ? parsedFromIso : parseTimestampToDate(rawTimestamp);
      if (!parsed) continue;

      const dayKey = formatDate(startOfDay(parsed), "yyyy-MM-dd");
      map.set(dayKey, Number((map.get(dayKey) ?? 0) + point.pnl));
    }
    return map;
  }, [timelinePoints]);

  const monthPnlByKey = useMemo(() => {
    const map = new Map<string, number>();
    for (const [dayKey, pnl] of dailyPnlByDateKey.entries()) {
      const monthKey = dayKey.slice(0, 7);
      map.set(monthKey, Number((map.get(monthKey) ?? 0) + pnl));
    }
    return map;
  }, [dailyPnlByDateKey]);

  const availableYears = useMemo(() => {
    const years = new Set<number>();
    for (const dayKey of dailyPnlByDateKey.keys()) {
      const year = Number(dayKey.slice(0, 4));
      if (Number.isFinite(year)) years.add(year);
    }

    if (years.size === 0) {
      years.add(latestParsedTimestamp.getFullYear());
    }

    return [...years].sort((a, b) => a - b);
  }, [dailyPnlByDateKey, latestParsedTimestamp]);

  useEffect(() => {
    if (availableYears.length === 0) return;
    if (!availableYears.includes(calendarValue.year)) {
      const fallbackYear = availableYears[availableYears.length - 1];
      setCalendarValue((previous) => ({
        ...previous,
        year: fallbackYear,
      }));
    }
  }, [availableYears, calendarValue.year]);

  const granularitySeries = useMemo<PnlSeriesPoint[]>(() => {
    let cumulative = 0;
    return (card.bars ?? []).map((bar) => {
      const periodPnl = Number((bar.value ?? 0).toFixed(2));
      cumulative += periodPnl;
      return {
        key: bar.key,
        label: bar.label,
        periodPnl,
        cumulativePnl: Number(cumulative.toFixed(2)),
      };
    });
  }, [card.bars]);

  const calendarSeries = useMemo<PnlSeriesPoint[]>(() => {
    if (calendarMode === "year") {
      return buildYearMonthSeries(calendarValue.year, monthPnlByKey);
    }

    const safeMonth = Math.max(0, Math.min(11, calendarValue.month ?? latestParsedTimestamp.getMonth()));
    const monthDate = new Date(calendarValue.year, safeMonth, 1);
    const start = startOfMonth(monthDate);
    const end = endOfMonth(monthDate);
    const dayRange = eachDayOfInterval({ start, end });
    let cumulative = 0;

    return dayRange.map((day) => {
      const dayKey = formatDate(day, "yyyy-MM-dd");
      const periodPnl = Number((dailyPnlByDateKey.get(dayKey) ?? 0).toFixed(2));
      cumulative += periodPnl;
      return {
        key: dayKey,
        label: formatDate(day, "d"),
        periodPnl,
        cumulativePnl: Number(cumulative.toFixed(2)),
      };
    });
  }, [calendarMode, calendarValue, dailyPnlByDateKey, latestParsedTimestamp, monthPnlByKey]);

  const activeSeries = viewMode === "calendar" ? calendarSeries : granularitySeries;
  const hasActiveTrades = activeSeries.some((point) => point.periodPnl !== 0);
  const periodTotal = activeSeries.reduce((sum, point) => sum + point.periodPnl, 0);
  const periodLabel =
    viewMode === "calendar"
      ? calendarMode === "month"
        ? formatDate(new Date(calendarValue.year, calendarValue.month ?? 0, 1), "MMM yyyy")
        : `Year ${calendarValue.year}`
      : "Selected granularity";

  return (
    <div className="sparkCard barCard pnlRechartsCard">
      <div className="sparkCard__meta">
        <span>{card.title}</span>
        <span className={cn("mono", periodTotal >= 0 ? "pos" : "neg")}>{formatCompactUsd(periodTotal)}</span>
      </div>

      <div className="pnlChartHeader">
        <TabsPrimitive.Root
          value={viewMode}
          onValueChange={(next) => setViewMode(next as "granularity" | "calendar")}
          className="pnlViewTabs"
        >
          <TabsPrimitive.List className="pnlViewTabs__list" aria-label="PnL view mode">
            <TabsPrimitive.Trigger className="pnlViewTabs__trigger" value="granularity">
              Granularity
            </TabsPrimitive.Trigger>
            <TabsPrimitive.Trigger className="pnlViewTabs__trigger" value="calendar">
              Calendar
            </TabsPrimitive.Trigger>
          </TabsPrimitive.List>
        </TabsPrimitive.Root>

        {viewMode === "calendar" ? (
          <div className="pnlCalendarHeader">
            <TabsPrimitive.Root
              value={calendarMode}
              onValueChange={(next) => setCalendarMode(next as PnlCalendarMode)}
              className="pnlSubTabs"
            >
              <TabsPrimitive.List className="pnlSubTabs__list" aria-label="Calendar mode">
                <TabsPrimitive.Trigger className="pnlSubTabs__trigger" value="month">
                  Month
                </TabsPrimitive.Trigger>
                <TabsPrimitive.Trigger className="pnlSubTabs__trigger" value="year">
                  Year
                </TabsPrimitive.Trigger>
              </TabsPrimitive.List>
            </TabsPrimitive.Root>

            <PnlCalendarControl
              mode={calendarMode}
              value={calendarValue}
              onChange={setCalendarValue}
              minYear={Math.min(...availableYears)}
              maxYear={Math.max(...availableYears)}
            />
          </div>
        ) : null}
      </div>

      <div className="pnlChartCaption muted mono">{periodLabel}</div>

      <div className="pnlRechartsFrame">
        <ResponsiveContainer width="100%" height={220}>
          <RechartsComposedChart data={activeSeries} margin={{ top: 12, right: 18, bottom: 8, left: 4 }}>
            <RechartsCartesianGrid strokeDasharray="3 3" stroke="rgba(120, 142, 172, 0.18)" />
            <RechartsXAxis
              dataKey="label"
              tickLine={false}
              axisLine={false}
              stroke="rgba(180, 197, 220, 0.9)"
              minTickGap={viewMode === "calendar" && calendarMode === "month" ? 10 : 20}
            />
            <RechartsYAxis
              yAxisId="left"
              tickLine={false}
              axisLine={false}
              width={48}
              stroke="rgba(180, 197, 220, 0.9)"
              tickFormatter={(value) => formatChartAxisValue(Number(value))}
            />
            <RechartsYAxis
              yAxisId="right"
              orientation="right"
              tickLine={false}
              axisLine={false}
              width={56}
              stroke="rgba(160, 188, 230, 0.88)"
              tickFormatter={(value) => formatChartAxisValue(Number(value))}
            />
            <RechartsTooltip
              cursor={{ fill: "rgba(255,255,255,0.06)" }}
              formatter={(value: number, name: string) => [
                formatPnlCurrency(Number(value)),
                name === "periodPnl" ? "Period PnL" : "Cumulative PnL",
              ]}
              labelFormatter={(label) => `Bucket ${label}`}
              contentStyle={{
                background: "rgba(6,10,16,.97)",
                border: "1px solid rgba(255,255,255,.14)",
                borderRadius: "10px",
              }}
            />
            <RechartsBar
              yAxisId="left"
              dataKey="periodPnl"
              name="periodPnl"
              fill="rgba(47,232,199,.88)"
              radius={[4, 4, 0, 0]}
            />
            <RechartsLine
              yAxisId="right"
              type="monotone"
              dataKey="cumulativePnl"
              name="cumulativePnL"
              stroke="rgba(124, 168, 255, .98)"
              strokeWidth={2}
              dot={false}
            />
          </RechartsComposedChart>
        </ResponsiveContainer>
      </div>

      {viewMode === "calendar" && !hasActiveTrades ? (
        <div className="pnlChartEmpty muted">No trades in this period.</div>
      ) : null}
    </div>
  );
}

function DangerHeatmap({
  dangerHours,
  biasScores,
  topTriggers,
  flaggedTrades,
  explainability,
}: {
  dangerHours?: number[][];
  biasScores?: AnalysisOutput["bias_scores"];
  topTriggers?: string[];
  flaggedTrades?: AnalysisOutput["flagged_trades"];
  explainability?: AnalysisOutput["explainability"];
}) {
  const hasBiasHeatmapData =
    Array.isArray(dangerHours) &&
    dangerHours.length > 0 &&
    dangerHours.some((row) => Array.isArray(row) && row.some((value) => Number.isFinite(value)));
  const normalizedDangerMatrix = normalizeDangerHoursMatrix(dangerHours);
  const maxDangerCount = normalizedDangerMatrix.reduce((max, row) => {
    const marketHourMax = DANGER_MARKET_HOUR_INDEXES.reduce((rowMax, hourIdx) => Math.max(rowMax, row[hourIdx]), 0);
    return Math.max(max, marketHourMax);
  }, 0);
  const dayTotals = normalizedDangerMatrix.map((row) =>
    DANGER_MARKET_HOUR_INDEXES.reduce((sum, hourIdx) => sum + row[hourIdx], 0)
  );
  const hourTotalsByHour = new Map<number, number>(
    DANGER_MARKET_HOUR_INDEXES.map((hourIdx) => [
      hourIdx,
      normalizedDangerMatrix.reduce((sum, row) => sum + row[hourIdx], 0),
    ])
  );
  const totalDangerTrades = dayTotals.reduce((sum, value) => sum + value, 0);
  const dominantSessionBias = resolveDominantBiasFromScores(biasScores);
  const fallbackBiasCounter = createBiasCounter();

  const cellBiasCounters = Array.from({ length: DANGER_DAY_COUNT }, () =>
    Array.from({ length: DANGER_HOUR_COUNT }, () => createBiasCounter())
  );
  const hourBiasCounters = Array.from({ length: DANGER_HOUR_COUNT }, () => createBiasCounter());

  for (const trade of flaggedTrades ?? []) {
    if (!trade?.bias) continue;
    const parsed = parseFlaggedTradeTimestamp(trade.timestamp);
    if (!parsed) continue;
    const hour = parsed.hour;
    if (hour < 0 || hour >= DANGER_HOUR_COUNT) continue;

    hourBiasCounters[hour][trade.bias] += 1;

    if (parsed.weekday != null && parsed.weekday >= 0 && parsed.weekday < DANGER_DAY_COUNT) {
      cellBiasCounters[parsed.weekday][hour][trade.bias] += 1;
    }
  }

  const columns = [
    {
      title: "Day",
      dataIndex: "day",
      key: "day",
      width: 54,
      render: (day: string) => <span className="mono">{day}</span>,
    },
    ...DANGER_MARKET_HOURS.map(({ hour: hourIdx, label }) => ({
      title: label,
      dataIndex: `h${hourIdx}`,
      key: label,
      width: 40,
      align: "center" as const,
      render: (cell: number, _record: Record<string, string | number>, rowIdx: number) => {
        const weekdayIdx = Number.isFinite(rowIdx) ? rowIdx : 0;
        const mainCellBias = resolveDominantBias(cellBiasCounters[weekdayIdx]?.[hourIdx] ?? fallbackBiasCounter);
        const mainHourBias = resolveDominantBias(hourBiasCounters[hourIdx] ?? fallbackBiasCounter);
        const mainBias = mainCellBias?.bias ?? mainHourBias?.bias ?? dominantSessionBias?.bias;
        const mainBiasLabel = mainBias ? BIAS_LABELS[mainBias] : "Unavailable";
        const mainBiasScore = mainBias != null ? biasScores?.[mainBias] : undefined;
        const topDriver = mainBias != null ? explainability?.[mainBias]?.[0] : undefined;
        const riskLabel = resolveDangerRiskLabel(cell, maxDangerCount);
        const dayTotal = dayTotals[weekdayIdx] || 0;
        const hourTotal = hourTotalsByHour.get(hourIdx) || 0;
        const shareOfDay = dayTotal > 0 ? (cell / dayTotal) * 100 : 0;
        const shareOfHour = hourTotal > 0 ? (cell / hourTotal) * 100 : 0;
        const shareOfAll = totalDangerTrades > 0 ? (cell / totalDangerTrades) * 100 : 0;
        const biasSource = mainCellBias
          ? "Detected in this weekday-hour"
          : mainHourBias
            ? "Detected for this hour across the week"
            : "Session-level dominant bias";

        const tooltipTitle = (
          <div className="dangerTooltipCard">
            <div className="dangerTooltipCard__title">
              {DANGER_DAYS[weekdayIdx]} {label}:00-{label}:59
            </div>
            <div className="dangerTooltipCard__row">
              <span>Trade count</span>
              <span className="dangerTooltipCard__value mono">{cell}</span>
            </div>
            <div className="dangerTooltipCard__row">
              <span>Risk level</span>
              <span className="dangerTooltipCard__value mono">{riskLabel}</span>
            </div>
            <div className="dangerTooltipCard__row">
              <span>Main bias</span>
              <span className="dangerTooltipCard__value mono">
                {mainBiasLabel}
                {Number.isFinite(mainBiasScore) ? ` (${Math.round(mainBiasScore as number)}/100)` : ""}
              </span>
            </div>
            <div className="dangerTooltipCard__row">
              <span>Share of weekday</span>
              <span className="dangerTooltipCard__value mono">{shareOfDay.toFixed(1)}%</span>
            </div>
            <div className="dangerTooltipCard__row">
              <span>Share of this hour</span>
              <span className="dangerTooltipCard__value mono">{shareOfHour.toFixed(1)}%</span>
            </div>
            <div className="dangerTooltipCard__row">
              <span>Share of all danger trades</span>
              <span className="dangerTooltipCard__value mono">{shareOfAll.toFixed(1)}%</span>
            </div>
            <div className="dangerTooltipCard__hint">{biasSource}</div>
            {topTriggers?.[0] ? <div className="dangerTooltipCard__hint">Top trigger: {topTriggers[0]}</div> : null}
            {topDriver?.detail ? <div className="dangerTooltipCard__hint">Driver: {topDriver.detail}</div> : null}
          </div>
        );

        return (
          <AntTooltip title={tooltipTitle} classNames={{ root: "dangerCellTooltip" }} placement="top">
            <span className={cn("dangerCell", resolveDangerCellToneClass(cell, maxDangerCount))} />
          </AntTooltip>
        );
      },
    })),
  ];

  const dataSource = normalizedDangerMatrix.map((row, rowIdx) => {
    const record: Record<string, string | number> = {
      key: DANGER_DAYS[rowIdx],
      day: DANGER_DAYS[rowIdx],
    };
    DANGER_MARKET_HOUR_INDEXES.forEach((hourIdx) => {
      record[`h${hourIdx}`] = row[hourIdx];
    });
    return record;
  });

  return (
    <div>
      <div className="heatmapHdr">
        <div>
          <div className="heatmapHdr__title">Behavior Risk Heatmap (Weekday x Hour)</div>
          <div className="heatmapHdr__axis muted">
            Y-axis: weekday (Mon-Sun) | X-axis: market hour bucket (09-17, local time)
          </div>
          <div className="muted">Risk incidents are based on detected flagged trades.</div>
        </div>
      </div>

      {!hasBiasHeatmapData ? (
        <div className="heatmapEmpty">
          <div className="heatmapEmpty__title">No Heatmap Data Yet</div>
          <div className="muted">Upload a trade CSV, then run analysis to populate weekday-hour danger data.</div>
        </div>
      ) : (
        <div className="dangerTableWrap">
          <AntTable
            className="dangerTable"
            size="small"
            columns={columns}
            dataSource={dataSource}
            pagination={false}
            scroll={{ x: "max-content" }}
          />
        </div>
      )}
    </div>
  );
}

function CommandWorkspace({
  behaviorIndex,
  biasScores,
  analysis,
}: {
  behaviorIndex: number;
  biasScores?: Record<string, number>;
  analysis?: AnalysisOutput | null;
}) {
  const riskScore = Math.max(0, Math.min(100, Math.round(behaviorIndex)));
  const [animatedRiskScore, setAnimatedRiskScore] = useState(0);
  const [chartGranularity, setChartGranularity] = useState<ChartGranularity>("15m");

  useEffect(() => {
    const animationMs = 950;
    const start = performance.now();
    let raf = 0;

    setAnimatedRiskScore(0);

    const tick = (now: number) => {
      const progress = Math.min((now - start) / animationMs, 1);
      const next = Math.round(riskScore * progress);
      setAnimatedRiskScore(next);
      if (progress < 1) raf = window.requestAnimationFrame(tick);
    };

    raf = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(raf);
  }, [riskScore]);

  useEffect(() => {
    const nextGranularity = detectDefaultChartGranularity(
      analysis?.trade_timeline,
      analysis?.flagged_trades,
      analysis?.daily_pnl
    );
    setChartGranularity(nextGranularity);
  }, [analysis?.trade_timeline, analysis?.flagged_trades, analysis?.daily_pnl]);

  const radius = 15.915;
  const circumference = 2 * Math.PI * radius;
  const stroke = `${(animatedRiskScore / 100) * circumference} ${circumference}`;
  const commandSnapshotCards = useMemo(
    () => buildCommandSnapshotCards(analysis, chartGranularity),
    [analysis, chartGranularity]
  );

  return (
    <div className="canvas">
      <div className="grid dangerFocusGrid">
        <Card title="Session Charts" className="span-5">
          <div className="snapshotControls">
            <AntSegmented
              className="chartGranularityTabs"
              value={chartGranularity}
              onChange={(value) => setChartGranularity(value as ChartGranularity)}
              options={CHART_GRANULARITY_OPTIONS}
            />
          </div>

          <div className="stack">
            {commandSnapshotCards.map((card) =>
              card.id === "realized_pnl" ? (
                <PnlChartCard key={card.id} card={card} analysis={analysis} />
              ) : (
                <BarChartCard
                  key={card.id}
                  title={card.title}
                  value={card.value}
                  valueTone={card.valueTone}
                  bars={card.bars}
                  legend={card.legend}
                  size="compact"
                />
              )
            )}
          </div>
        </Card>

        <div className="span-7 commandPane">
          <Card title="Behavior Index">
            <div className="behaviorTop">
              <div className="gauge">
                <svg className="gauge__svg" viewBox="0 0 36 36" aria-hidden="true">
                  <circle className="gauge__bg" cx="18" cy="18" r={radius} />
                  <circle className="gauge__fg" cx="18" cy="18" r={radius} strokeDasharray={stroke} />
                </svg>
                <div className="gauge__center">
                  <div className="gauge__num mono">{animatedRiskScore}%</div>
                </div>
              </div>
              <div className="kpiCol">
                <div className="kpiBig mono">{animatedRiskScore}%</div>
                <div className="muted">Bias index</div>
              </div>
            </div>

            <div className="divider" />

            <div className="commandTriggers">
              <div className="biasScoresSection">
                <div className="section__title">Individual Bias Scores</div>
                <div className="biasScoreRow">
                  <span className="biasLabel">Overtrading</span>
                  <span className="biasMono">{Math.round(biasScores?.overtrading || 0)}</span>
                </div>
                <div className="biasScoreRow">
                  <span className="biasLabel">Loss Aversion</span>
                  <span className="biasMono">{Math.round(biasScores?.loss_aversion || 0)}</span>
                </div>
                <div className="biasScoreRow">
                  <span className="biasLabel">Revenge Trading</span>
                  <span className="biasMono">{Math.round(biasScores?.revenge_trading || 0)}</span>
                </div>
                <div className="biasScoreRow">
                  <span className="biasLabel">Recency Bias</span>
                  <span className="biasMono">{Math.round(biasScores?.recency_bias || 0)}</span>
                </div>
              </div>
            </div>
          </Card>

          <div className="commandPane__bottom">
            <Card title="Danger Heatmaps">
              <DangerHeatmap
                dangerHours={analysis?.danger_hours}
                biasScores={analysis?.bias_scores}
                topTriggers={analysis?.top_triggers}
                flaggedTrades={analysis?.flagged_trades}
                explainability={analysis?.explainability}
              />
            </Card>

            <Card title="Trade Panel" className="compact commandPane__trade">
              <div className="miniList">
                <div className="miniRow">
                  <span>PnL</span>
                  <span className={`mono ${(analysis?.summary?.total_pnl || 0) >= 0 ? 'pos' : 'neg'}`}>
                    {(analysis?.summary?.total_pnl || 0) >= 0 ? '+' : ''}{(analysis?.summary?.total_pnl || 0).toFixed(2)}
                  </span>
                </div>
                <div className="miniRow">
                  <span>Drawdown</span>
                  <span className="mono neg">
                    -{(analysis?.summary?.max_drawdown || 0).toFixed(2)}
                  </span>
                </div>
                <div className="miniRow">
                  <span>Trade count</span>
                  <span className="mono">{analysis?.summary?.total_trades || 0}</span>
                </div>
                <div className="miniRow">
                  <span>Volatility</span>
                  <span className="mono">{(analysis?.summary?.volatility || 0).toFixed(2)}</span>
                </div>
              </div>

              <div className="divider" />
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}

function TradesWorkspace({
  trades,
  selected,
  onSelect,
  flaggedTradesRaw,
  biasScores,
}: {
  trades: Trade[];
  selected: Trade | null;
  onSelect: (t: Trade) => void;
  flaggedTradesRaw?: FlaggedTrade[];
  biasScores?: Record<string, number>;
}) {
  const [sortKey, setSortKey] = useState<"time" | "pnl" | null>(null);
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc");

  const sorted = useMemo(() => {
    if (!sortKey) return trades;
    return [...trades].sort((a, b) => {
      const mul = sortDir === "asc" ? 1 : -1;
      if (sortKey === "pnl") return (a.pnl - b.pnl) * mul;
      return a.time.localeCompare(b.time) * mul;
    });
  }, [trades, sortKey, sortDir]);

  const flaggedTradeById = useMemo(() => {
    const byId = new Map<string, FlaggedTrade>();
    for (const trade of flaggedTradesRaw ?? []) {
      if (!trade.trade_id) continue;
      byId.set(trade.trade_id, trade);
    }
    return byId;
  }, [flaggedTradesRaw]);

  const toggleSort = (key: "time" | "pnl") => {
    if (sortKey === key) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else { setSortKey(key); setSortDir("asc"); }
  };

  const parentRef = useRef<HTMLDivElement>(null);
  const rowVirtualizer = useVirtualizer({
    count: sorted.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 42,
    overscan: 10,
  });

  const cols = [
    { key: "time",       label: "Date",          sortable: true,  align: "left"   },
    { key: "symbol",     label: "Symbol",         sortable: false, align: "left"   },
    { key: "side",       label: "Side",           sortable: false, align: "left"   },
    { key: "size",       label: "Quantity",       sortable: false, align: "center" },
    { key: "pnl",        label: "Profit / Loss",  sortable: true,  align: "center" },
    { key: "balance",    label: "Balance",        sortable: false, align: "center" },
    { key: "flags",      label: "Bias Flag",      sortable: false, align: "left"   },
    { key: "confidence", label: "Confidence",     sortable: false, align: "center" },
  ] as const;

  const renderCell = (trade: Trade, key: string) => {
    switch (key) {
      case "time":       return <span className="mono">{trade.time}</span>;
      case "symbol":     return <span className="mono">{trade.symbol}</span>;
      case "side":       return <span>{trade.side}</span>;
      case "size":       return <span className="mono">{trade.size.toFixed(2)}</span>;
      case "pnl":        return <span className={cn("mono", trade.pnl >= 0 ? "pos" : "neg")}>{trade.pnl >= 0 ? `+${trade.pnl.toFixed(2)}` : trade.pnl.toFixed(2)}</span>;
      case "balance":    return trade.balance != null ? <span className="mono">${trade.balance.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span> : <span className="muted">N/A</span>;
      case "flags":      return <div className="chipsRow">{trade.flags.length ? trade.flags.map(f => <Pill key={f} tone="warn">{f}</Pill>) : <Pill tone="ok">Clean</Pill>}</div>;
      case "confidence": return <span className="mono">{trade.confidence ? `${trade.confidence}%` : "N/A"}</span>;
      default:           return null;
    }
  };

  return (
    <div className="canvas">
      <div className="grid">
        <Card title="Flagged Trades" className="span-7">
          <div className="vTableWrap">
            {/* Sticky header */}
            <div className="vTable__head">
              {cols.map((col) => (
                <div
                  key={col.key}
                  className={cn("vTable__th", col.sortable && "vTable__th--sort", col.align === "center" && "vTable__th--center")}
                  onClick={col.sortable ? () => toggleSort(col.key as "time" | "pnl") : undefined}
                >
                  {col.label}
                  {col.sortable && (
                    <span className="vTable__sorter">
                      <span className={cn("vTable__arrow", sortKey === col.key && sortDir === "asc" && "vTable__arrow--active")}>^</span>
                      <span className={cn("vTable__arrow", sortKey === col.key && sortDir === "desc" && "vTable__arrow--active")}>v</span>
                    </span>
                  )}
                </div>
              ))}
            </div>

            {/* Scrollable virtual body */}
            <div ref={parentRef} className="vTable__body">
              <div style={{ height: rowVirtualizer.getTotalSize(), position: "relative" }}>
                {rowVirtualizer.getVirtualItems().map((vRow) => {
                  const trade = sorted[vRow.index];
                  const isSelected = selected?.id === trade.id;
                  return (
                    <div
                      key={trade.id}
                      data-index={vRow.index}
                      ref={rowVirtualizer.measureElement}
                      className={cn("vTable__row", isSelected && "vTable__row--active")}
                      style={{ position: "absolute", top: vRow.start, left: 0, right: 0 }}
                      onClick={() => onSelect(trade)}
                      onMouseEnter={(e) => {
                        const el = e.currentTarget as HTMLElement;
                        el.style.transform = "scaleY(1.09) scaleX(1.005)";
                        el.style.boxShadow = "0 0 0 1px rgba(200,220,255,.12), 0 8px 28px rgba(0,0,0,.75), inset 0 1px 0 rgba(255,255,255,.07), inset 0 -1px 0 rgba(255,255,255,.03)";
                        el.style.background = "rgba(18,26,40,.92)";
                        el.style.zIndex = "3";
                      }}
                      onMouseLeave={(e) => {
                        const el = e.currentTarget as HTMLElement;
                        el.style.transform = "";
                        el.style.boxShadow = "";
                        el.style.background = "";
                        el.style.zIndex = "";
                      }}
                    >
                      {cols.map((col) => (
                        <div key={col.key} className={cn("vTable__td", col.align === "center" && "vTable__td--center")}>
                          {renderCell(trade, col.key)}
                        </div>
                      ))}
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </Card>

        <Card title="Trade Inspector" className="span-5">
          {selected && flaggedTradesRaw ? (
            (() => {
              const flaggedTrade = flaggedTradeById.get(selected.id);
              if (!flaggedTrade) return <div className="muted">Trade details not found.</div>;
              
              const biasDisplayNames: Record<BiasKey, string> = {
                "overtrading": "Overtrading",
                "loss_aversion": "Loss Aversion",
                "revenge_trading": "Revenge Trading",
                "recency_bias": "Recency Bias",
              };

              const allBiasTypes = flaggedTrade.bias_types || [flaggedTrade.bias];
              
              return (
                <div>
                  <div className="muted">
                    <strong>Trade ID:</strong> <span className="mono">{flaggedTrade.trade_id}</span>
                  </div>
                  <div className="divider" />
                  
                  <div className="section">
                    <div className="section__title">All Flags Raised ({allBiasTypes.length})</div>
                    {allBiasTypes.length > 0 ? (
                      allBiasTypes.map((biasKey) => {
                        const displayName = biasDisplayNames[biasKey] || biasKey;
                        const score = biasScores ? biasScores[biasKey] : 0;
                        return (
                          <div key={biasKey} style={{ marginBottom: "12px", padding: "10px", border: "1px solid rgba(255,255,255,.08)", borderRadius: "8px", background: "rgba(0,0,0,.2)" }}>
                            <div style={{ fontWeight: 600, marginBottom: "6px" }}>{displayName}</div>
                            <div className="muted" style={{ fontSize: "12px", marginBottom: "4px" }}>
                              <strong>Detector Score:</strong> <span className="mono">{Math.round(score)}/100</span>
                            </div>
                            <div className="muted" style={{ fontSize: "12px" }}>
                              <strong>Triggered By:</strong>
                            </div>
                            <div className="muted" style={{ fontSize: "11px", marginTop: "4px", fontStyle: "italic", color: "rgba(232,240,255,.7)" }}>
                              {flaggedTrade.evidence && flaggedTrade.evidence.length > 0
                                ? flaggedTrade.evidence.join(" | ")
                                : "No specific evidence recorded"}
                            </div>
                          </div>
                        );
                      })
                    ) : (
                      <div className="muted">No flags raised for this trade.</div>
                    )}
                  </div>
                </div>
              );
            })()
          ) : (
            <div className="muted">Pick a trade to inspect.</div>
          )}
        </Card>
      </div>
    </div>
  );
}

function SimulatorWorkspace({
  analysisData,
  datasetId,
}: {
  analysisData: AnalysisOutput | null;
  datasetId: string | null;
}) {
  const [cooldownMinutes, setCooldownMinutes] = useState(15);
  const [dailyTradeCap, setDailyTradeCap] = useState(10);
  const [positionSizeMultiplier, setPositionSizeMultiplier] = useState(1.5);
  const [maxConsecutiveLosses, setMaxConsecutiveLosses] = useState(3);
  const [simulationResult, setSimulationResult] = useState<any>(null);
  const [isRunning, setIsRunning] = useState(false);

  // Calculate max values from backend analysis summary
  const totalTrades = analysisData?.summary?.total_trades ?? 0;
  const maxDailyTradesCap = Math.max(50, totalTrades);
  const maxLossesStreak = Math.max(10, totalTrades);

  // Update default values when data is imported
  useEffect(() => {
    if (totalTrades > 0) {
      // Set reasonable defaults based on data size
      const defaultDailyCap = Math.ceil(totalTrades / 5); // Allow ~20% of trades per day
      setDailyTradeCap(Math.max(5, Math.min(defaultDailyCap, maxDailyTradesCap)));
      
      const defaultLossStreak = Math.max(3, Math.ceil(totalTrades * 0.1)); // ~10% of trades
      setMaxConsecutiveLosses(Math.max(2, Math.min(defaultLossStreak, maxLossesStreak)));
    }
  }, [totalTrades]);

  const runSimulation = async () => {
    if (!datasetId) {
      antMessage.warning("No trade data available. Please import a CSV first.");
      return;
    }

    setIsRunning(true);
    try {
      const result = await simulateDataset({
        dataset_id: datasetId,
        rules: {
          cooldown_after_loss_minutes: cooldownMinutes,
          daily_trade_cap: dailyTradeCap,
          max_position_size_multiplier: positionSizeMultiplier,
          stop_after_consecutive_losses: maxConsecutiveLosses,
        },
      });

      setSimulationResult(result);
      antMessage.success("Simulation complete!");
    } catch (e) {
      console.error(e);
      antMessage.error(`Simulation failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setIsRunning(false);
    }
  };

  const pnlImprovement = simulationResult
    ? ((simulationResult.simulated_pnl - simulationResult.baseline_pnl) / Math.abs(simulationResult.baseline_pnl)) * 100
    : 0;
  const tradeReduction = simulationResult
    ? ((simulationResult.baseline_trade_count - simulationResult.simulated_trade_count) / simulationResult.baseline_trade_count) * 100
    : 0;

  return (
    <div className="canvas">
      <div className="grid">
        <Card title="Simulation Rules" className="span-4 tall">
          <div className="stack">
            {totalTrades > 0 && (
              <div style={{ padding: "12px", borderRadius: "8px", background: "rgba(47,232,199,.08)", border: "1px solid rgba(47,232,199,.16)", marginBottom: "12px" }}>
                <div className="muted" style={{ fontSize: "11px" }}>Total trades available</div>
                <div className="mono" style={{ fontSize: "20px", fontWeight: 700, color: "rgba(47,232,199,1)" }}>{totalTrades}</div>
              </div>
            )}
            <div className="ruleConfig">
              <div className="muted">Wait time after loss</div>
              <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
                <AntSlider
                  min={1}
                  max={60}
                  value={cooldownMinutes}
                  onChange={(val) => setCooldownMinutes(val)}
                  style={{ flex: 1 }}
                  tooltip={{ open: false }}
                />
                <span className="mono" style={{ minWidth: "40px" }}>{cooldownMinutes}m</span>
              </div>
            </div>

            <div className="ruleConfig">
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div className="muted">Max trades per day</div>
                {totalTrades > 0 && <div className="muted" style={{ fontSize: "11px" }}>max: {maxDailyTradesCap}</div>}
              </div>
              <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
                <AntSlider
                  min={1}
                  max={maxDailyTradesCap}
                  value={dailyTradeCap}
                  onChange={(val) => setDailyTradeCap(val)}
                  style={{ flex: 1 }}
                  tooltip={{ open: false }}
                />
                <span className="mono" style={{ minWidth: "40px" }}>{dailyTradeCap}</span>
              </div>
            </div>

            <div className="ruleConfig">
              <div className="muted">Max position size</div>
              <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
                <AntSlider
                  min={0.5}
                  max={5}
                  step={0.1}
                  value={positionSizeMultiplier}
                  onChange={(val) => setPositionSizeMultiplier(val)}
                  style={{ flex: 1 }}
                  tooltip={{ open: false }}
                />
                <span className="mono" style={{ minWidth: "60px" }}>{positionSizeMultiplier.toFixed(1)}x</span>
              </div>
            </div>

            <div className="ruleConfig">
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div className="muted">Stop after N losses</div>
                {totalTrades > 0 && <div className="muted" style={{ fontSize: "11px" }}>max: {maxLossesStreak}</div>}
              </div>
              <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
                <AntSlider
                  min={1}
                  max={maxLossesStreak}
                  value={maxConsecutiveLosses}
                  onChange={(val) => setMaxConsecutiveLosses(val)}
                  style={{ flex: 1 }}
                  tooltip={{ open: false }}
                />
                <span className="mono" style={{ minWidth: "40px" }}>{maxConsecutiveLosses}</span>
              </div>
            </div>

            <div style={{ marginTop: "16px" }}>
              <AntButton
                onClick={runSimulation}
                loading={isRunning}
                type="primary"
                style={{ width: "100%" }}
              >
                Run Simulation
              </AntButton>
            </div>
          </div>
        </Card>

        {simulationResult && (
          <>
            <Card title="Baseline vs Simulated" className="span-4 tall">
              <div className="simGrid">
                <div className="simCard">
                  <div className="muted">Baseline PnL</div>
                  <div className={cn("simBig mono", simulationResult.baseline_pnl >= 0 ? "pos" : "neg")}>
                    ${simulationResult.baseline_pnl.toFixed(2)}
                  </div>
                </div>
                <div className="simCard">
                  <div className="muted">Simulated PnL</div>
                  <div className={cn("simBig mono", simulationResult.simulated_pnl >= 0 ? "pos" : "neg")}>
                    ${simulationResult.simulated_pnl.toFixed(2)}
                  </div>
                </div>
                <div className="simCard">
                  <div className="muted">PnL Impact</div>
                  <div className={cn("simBig mono", pnlImprovement >= 0 ? "pos" : "neg")}>
                    {pnlImprovement >= 0 ? "+" : ""}{pnlImprovement.toFixed(1)}%
                  </div>
                </div>
                <div className="simCard">
                  <div className="muted">Trades Filtered</div>
                  <div className="simBig mono">{tradeReduction.toFixed(1)}%</div>
                </div>
              </div>
            </Card>

            <Card title="Rule Impact Breakdown" className="span-4 tall">
              <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: "12px" }}>
                {simulationResult.simulation_log?.map((log: any, i: number) => (
                  <div
                    key={i}
                    style={{
                      padding: "14px",
                      border: "1px solid rgba(255,255,255,.08)",
                      borderRadius: "8px",
                      background: "rgba(0,0,0,.2)",
                    }}
                  >
                    <div style={{ marginBottom: "8px" }}>
                      <div className="muted" style={{ fontSize: "12px", marginBottom: "2px" }}>
                        {log.rule.replace(/_/g, " ")}
                      </div>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end" }}>
                      <div>
                        <div className="mono" style={{ fontSize: "18px", fontWeight: 700 }}>
                          {log.trades_skipped}
                        </div>
                        <div className="muted" style={{ fontSize: "11px" }}>blocked</div>
                      </div>
                      <div style={{ textAlign: "right" }}>
                        <div className={cn("mono", log.pnl_impact < 0 ? "neg" : "pos")} style={{ fontSize: "18px", fontWeight: 700 }}>
                          {log.pnl_impact < 0 ? "" : "+"}${log.pnl_impact.toFixed(0)}
                        </div>
                        <div className="muted" style={{ fontSize: "11px" }}>PnL impact</div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          </>
        )}

        {simulationResult && simulationResult.blocked_trades.length > 0 && (
          <Card title="Affected Trades" className="span-full">
            <div className="simTableWrap" style={{ maxHeight: "300px", overflow: "auto" }}>
              <table className="simTable">
                <thead>
                  <tr>
                    <th>Asset</th>
                    <th>Time</th>
                    <th>Original PnL</th>
                    <th>Action</th>
                    <th>Reason</th>
                  </tr>
                </thead>
                <tbody>
                  {simulationResult.blocked_trades.map((trade: any, i: number) => (
                    <tr key={i}>
                      <td className="mono">{trade.asset}</td>
                      <td className="mono">{new Date(trade.timestamp).toLocaleTimeString()}</td>
                      <td className={cn("mono", trade.profit_loss >= 0 ? "pos" : "neg")}>
                        ${trade.profit_loss.toFixed(2)}
                      </td>
                      <td>
                        <span style={{ color: "rgba(255,107,107,1)" }}>Blocked</span>
                      </td>
                      <td className="muted" style={{ fontSize: "12px" }}>
                        {trade.reason}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        )}
      </div>
    </div>
  );
}

interface QuizState {
  currentQuestionIndex: number;
  userAnswers: (number | null)[];
  showFeedback: boolean;
  isComplete: boolean;
  biasesCovered?: { bias: BiasKey; score: number }[];
}

interface PracticeWorkspaceProps {
  datasetId: string | null;
  scoringMode: ScoringMode;
  analysis: AnalysisOutput | null;
}

function PracticeWorkspace({ datasetId, scoringMode, analysis }: PracticeWorkspaceProps) {
  const [quizState, setQuizState] = useState<QuizState>({
    currentQuestionIndex: 0,
    userAnswers: [],
    showFeedback: false,
    isComplete: false,
    biasesCovered: [],
  });
  const [questions, setQuestions] = useState<PracticeQuestion[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Load practice questions when component mounts
  useEffect(() => {
    if (!datasetId) {
      setError("No dataset loaded. Import CSV data first to start practice.");
      return;
    }

    const loadQuestions = async () => {
      setLoading(true);
      setError("");
      try {
        const response = await generatePracticeQuestions({
          dataset_id: datasetId,
          scoring_mode: scoringMode,
        });
        setQuestions(response.questions);
        setQuizState({
          currentQuestionIndex: 0,
          userAnswers: new Array(response.questions.length).fill(null),
          showFeedback: false,
          isComplete: false,
          biasesCovered: response.biases_covered,
        });
      } catch (err: unknown) {
        setError(`Failed to load practice questions: ${err instanceof Error ? err.message : String(err)}`);
      } finally {
        setLoading(false);
      }
    };

    void loadQuestions();
  }, [datasetId, scoringMode]);

  const handleAnswerSelect = (choiceIndex: number) => {
    const newAnswers = [...quizState.userAnswers];
    newAnswers[quizState.currentQuestionIndex] = choiceIndex;
    setQuizState((prev) => ({
      ...prev,
      userAnswers: newAnswers,
      showFeedback: true,
    }));
  };

  const handleNextQuestion = () => {
    if (quizState.currentQuestionIndex < questions.length - 1) {
      setQuizState((prev) => ({
        ...prev,
        currentQuestionIndex: prev.currentQuestionIndex + 1,
        showFeedback: false,
      }));
    } else {
      setQuizState((prev) => ({
        ...prev,
        isComplete: true,
      }));
    }
  };

  const handleRestartQuiz = () => {
    setQuizState({
      currentQuestionIndex: 0,
      userAnswers: new Array(questions.length).fill(null),
      showFeedback: false,
      isComplete: false,
      biasesCovered: quizState.biasesCovered,
    });
  };

  // Calculate score
  const calculateScore = () => {
    let correct = 0;
    questions.forEach((q, idx) => {
      if (quizState.userAnswers[idx] === q.correct_index) correct++;
    });
    return { correct, total: questions.length };
  };

  // Group answers by bias
  const scoreByBias = () => {
    const scores: Record<string, { correct: number; total: number }> = {};
    questions.forEach((q, idx) => {
      if (!scores[q.bias]) scores[q.bias] = { correct: 0, total: 0 };
      scores[q.bias].total += 1;
      if (quizState.userAnswers[idx] === q.correct_index) scores[q.bias].correct += 1;
    });
    return scores;
  };

  // Get habit suggestion
  const getHabitSuggestion = () => {
    const scores = scoreByBias();
    const lowestBias = Object.entries(scores).reduce((prev, curr) =>
      curr[1].correct < prev[1].correct ? curr : prev
    );

    const suggestions: Record<string, string> = {
      overtrading:
        "Set a timer for every 2 hours. When it goes off, take a 15-minute break from trading. No exceptions.",
      loss_aversion:
        "Write your trade plan BEFORE entering. If the setup hasn't changed, don't exit until your target or your stop.",
      revenge_trading:
        "After any loss, wait 10 minutes before viewing charts. Use that time to do 10 pushups or drink water.",
      recency_bias:
        "Keep a 'macro view' chart open at all times. Check it before every trade to remember you're part of a long-term edge.",
    };

    return suggestions[lowestBias[0]] || "Stay disciplined. Your edge is long-term.";
  };

  if (error)
    return (
      <div className="canvas">
        <Card title="Practice Quiz" className="span-12">
          <div style={{ padding: "20px", textAlign: "center", color: "#ff6b6b" }}>{error}</div>
        </Card>
      </div>
    );

  if (loading)
    return (
      <div className="canvas">
        <Card title="Practice Quiz" className="span-12">
          <div style={{ padding: "20px", textAlign: "center" }}>Loading personalized practice questions...</div>
        </Card>
      </div>
    );

  if (questions.length === 0) return null;

  if (quizState.isComplete) {
    const score = calculateScore();
    const scores = scoreByBias();
    const habitSuggestion = getHabitSuggestion();

    return (
      <div className="canvas">
        <div className="grid">
          <Card title="Practice Complete" className="span-full">
            <div style={{ padding: "20px" }}>
              <div style={{ textAlign: "center", marginBottom: "40px" }}>
                <div style={{ fontSize: "72px", fontWeight: "bold", color: "#51cf66", marginBottom: "15px" }}>
                  {score.correct}/{score.total}
                </div>
                <div style={{ fontSize: "22px", color: "#888", marginBottom: "8px" }}>Questions Correct</div>
                <div style={{ fontSize: "16px", color: "#aaa" }}>
                  Great work! You're building better trading habits.
                </div>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "40px", marginBottom: "40px" }}>
                <div>
                  <div style={{ fontSize: "20px", fontWeight: "bold", marginBottom: "20px", textAlign: "left" }}>
                    Results by Bias
                  </div>
                  {Object.entries(scores).map(([bias, result]) => (
                    <div
                      key={bias}
                      style={{
                        padding: "16px",
                        marginBottom: "12px",
                        background: "rgba(100, 200, 100, 0.1)",
                        border: "1px solid rgba(100, 200, 100, 0.3)",
                        borderRadius: "8px",
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                      }}
                    >
                      <span style={{ textTransform: "capitalize", fontSize: "15px", fontWeight: "500" }}>
                        {bias.replace(/_/g, " ")}
                      </span>
                      <span
                        style={{
                          fontWeight: "bold",
                          color: result.correct === result.total ? "#51cf66" : "#ffa500",
                          fontSize: "16px",
                        }}
                      >
                        {result.correct}/{result.total}
                      </span>
                    </div>
                  ))}
                </div>

                <div
                  style={{
                    padding: "28px",
                    background: "rgba(100, 150, 255, 0.1)",
                    border: "1px solid rgba(100, 150, 255, 0.3)",
                    borderRadius: "12px",
                    display: "flex",
                    flexDirection: "column",
                    justifyContent: "center",
                  }}
                >
                  <div style={{ fontSize: "16px", fontWeight: "bold", marginBottom: "15px", color: "#51cf66" }}>Your Next Habit</div>
                  <div style={{ fontSize: "15px", lineHeight: "1.8", color: "#ddd" }}>
                    {habitSuggestion}
                  </div>
                </div>
              </div>
              <div style={{ fontSize: "16px", fontWeight: "bold", marginBottom: "15px", color: "#51cf66" }}>Your Next Habit</div>
              <div style={{ fontSize: "15px", lineHeight: "1.8", color: "#ddd" }}>{habitSuggestion}</div>
            </div>

            <div style={{ display: "flex", gap: "12px", justifyContent: "center" }}>
              <Button variant="primary" onClick={handleRestartQuiz}>
                Retake Quiz
              </Button>
              <Button variant="ghost" onClick={() => setQuizState({ ...quizState, isComplete: false })}>
                Review Answers
              </Button>
            </div>
          </Card>

          <Card title="Your Test Biases" className="span-full">
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
                gap: "16px",
                padding: "10px",
              }}
            >
              {quizState.biasesCovered?.map((b) => (
                <div
                  key={b.bias}
                  style={{
                    padding: "16px",
                    background: "rgba(255, 100, 100, 0.1)",
                    border: "1px solid rgba(255, 100, 100, 0.3)",
                    borderRadius: "8px",
                  }}
                >
                  <div style={{ fontWeight: "bold", textTransform: "capitalize", fontSize: "15px", marginBottom: "8px" }}>
                    {b.bias.replace(/_/g, " ")}
                  </div>
                  <div style={{ fontSize: "13px", color: "#888" }}>
                    Bias Score: <span style={{ color: "#ff8787", fontWeight: "bold" }}>{b.score}/100</span>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>
    );
  }

  const currentQuestion = questions[quizState.currentQuestionIndex];
  const isCurrentAnswered = quizState.userAnswers[quizState.currentQuestionIndex] !== null;
  const userSelectedIndex = quizState.userAnswers[quizState.currentQuestionIndex];
  const isCorrect = userSelectedIndex === currentQuestion.correct_index;

  return (
    <div className="canvas">
      <div className="grid">
        <Card title={`Adventure Practice: Trading Biases`} className="span-12">
          <div style={{ marginBottom: "20px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "10px" }}>
              <span style={{ fontSize: "14px", color: "#888" }}>
                Question {quizState.currentQuestionIndex + 1} of {questions.length}
              </span>
              <span style={{ fontSize: "12px", color: "#888" }}>
                Bias: <strong>{currentQuestion.bias.replace(/_/g, " ").toUpperCase()}</strong>
              </span>
            </div>
            <div
              style={{
                height: "4px",
                background: "rgba(255,255,255,0.1)",
                borderRadius: "2px",
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  height: "100%",
                  width: `${((quizState.currentQuestionIndex + 1) / questions.length) * 100}%`,
                  background: "#51cf66",
                  transition: "width 0.3s",
                }}
              />
            </div>
          </div>

          <div style={{ marginBottom: "30px" }}>
            <div style={{ fontSize: "16px", fontWeight: "bold", marginBottom: "15px", lineHeight: "1.6" }}>
              {currentQuestion.scenario}
            </div>
          </div>

          <div style={{ marginBottom: "30px" }}>
            <div style={{ fontSize: "14px", color: "#aaa", marginBottom: "12px" }}>What should you do?</div>
            <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
              {currentQuestion.choices.map((choice, idx) => (
                <button
                  key={idx}
                  onClick={() => !isCurrentAnswered && handleAnswerSelect(idx)}
                  disabled={isCurrentAnswered}
                  style={{
                    padding: "12px 16px",
                    textAlign: "left",
                    border: "1px solid",
                    borderRadius: "6px",
                    background:
                      isCurrentAnswered && idx === currentQuestion.correct_index
                        ? "rgba(81, 207, 102, 0.2)"
                        : isCurrentAnswered && idx === userSelectedIndex && !isCorrect
                        ? "rgba(255, 107, 107, 0.2)"
                        : "rgba(255, 255, 255, 0.05)",
                    borderColor:
                      isCurrentAnswered && idx === currentQuestion.correct_index
                        ? "rgba(81, 207, 102, 0.5)"
                        : isCurrentAnswered && idx === userSelectedIndex && !isCorrect
                        ? "rgba(255, 107, 107, 0.5)"
                        : "rgba(255, 255, 255, 0.1)",
                    color: "inherit",
                    cursor: isCurrentAnswered ? "default" : "pointer",
                    opacity: isCurrentAnswered && idx !== currentQuestion.correct_index && idx !== userSelectedIndex ? 0.5 : 1,
                    transition: "all 0.2s",
                  }}
                >
                  <span style={{ marginRight: "12px", fontWeight: "bold" }}>{String.fromCharCode(65 + idx)})</span>
                  {choice}
                  {isCurrentAnswered && idx === currentQuestion.correct_index && (
                    <span style={{ marginLeft: "auto", color: "#51cf66" }}>Correct</span>
                  )}
                  {isCurrentAnswered && idx === userSelectedIndex && !isCorrect && (
                    <span style={{ marginLeft: "auto", color: "#ff6b6b" }}>Not quite</span>
                  )}
                </button>
              ))}
            </div>
          </div>

          {quizState.showFeedback && (
            <div
              style={{
                padding: "16px",
                background: isCorrect ? "rgba(81, 207, 102, 0.1)" : "rgba(255, 107, 107, 0.1)",
                border: `1px solid ${isCorrect ? "rgba(81, 207, 102, 0.3)" : "rgba(255, 107, 107, 0.3)"}`,
                borderRadius: "8px",
                marginBottom: "20px",
              }}
            >
              <div style={{ fontWeight: "bold", marginBottom: "8px", color: isCorrect ? "#51cf66" : "#ff8787" }}>
                {isCorrect ? "Excellent!" : "Let's learn from this."}
              </div>
              <div style={{ fontSize: "14px", lineHeight: "1.6", color: "#ddd" }}>{currentQuestion.explanation}</div>
            </div>
          )}

          {isCurrentAnswered && (
            <div style={{ display: "flex", gap: "10px", justifyContent: "center" }}>
              <Button
                variant="primary"
                onClick={handleNextQuestion}
                disabled={!isCurrentAnswered}
              >
                {quizState.currentQuestionIndex === questions.length - 1 ? "See Results" : "Next Question"}
              </Button>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}

function PulseWorkspace({
  datasetId,
  analysis,
}: {
  datasetId: string | null;
  analysis: AnalysisOutput | null;
}) {
  const topBiases = useMemo(() => getTopTwoBiasesFromScores(analysis?.bias_scores), [analysis]);
  const [answersByBias, setAnswersByBias] = useState<Record<CanonicalBiasType, EmotionalCheckInAnswer[]>>(
    createEmptyCheckinAnswerMap
  );
  const [explainerData, setExplainerData] = useState<BiasContextExplainerResponse | null>(null);
  const [loadingContext, setLoadingContext] = useState(false);

  const hydrateAnswersFromStorage = useCallback(() => {
    if (!datasetId) {
      setAnswersByBias(createEmptyCheckinAnswerMap());
      return;
    }

    const nextAnswers = createEmptyCheckinAnswerMap();
    topBiases.forEach((bias) => {
      nextAnswers[bias.biasType] = readStoredBiasAnswers(datasetId, bias.biasType);
    });

    setAnswersByBias(nextAnswers);
  }, [datasetId, topBiases]);

  useEffect(() => {
    if (!datasetId) {
      setExplainerData(null);
      setAnswersByBias(createEmptyCheckinAnswerMap());
      return;
    }

    hydrateAnswersFromStorage();
    setExplainerData(null);
  }, [datasetId, hydrateAnswersFromStorage]);

  const updateAnswer = (biasType: CanonicalBiasType, questionIndex: number, answer: EmotionalCheckInAnswer) => {
    if (!datasetId) return;

    setAnswersByBias((previous) => {
      const current = previous[biasType] ?? [null, null, null];
      const updatedAnswers: EmotionalCheckInAnswer[] = [current[0] ?? null, current[1] ?? null, current[2] ?? null];
      updatedAnswers[questionIndex] = answer;
      writeStoredBiasAnswers(datasetId, biasType, updatedAnswers);
      return {
        ...previous,
        [biasType]: updatedAnswers,
      };
    });
  };

  const generatePersonalizedContext = async (): Promise<boolean> => {
    if (!datasetId) {
      antMessage.warning("Please import a dataset first.");
      return false;
    }

    if (topBiases.length === 0) {
      antMessage.warning("No top biases available. Import a dataset and run analysis first.");
      return false;
    }

    const emotionalPayload: EmotionalCheckInBiasPayload[] = topBiases.map((bias) => ({
      bias_type: bias.biasType,
      score: bias.score,
      responses: EMOTIONAL_CHECKIN_QUESTIONS[bias.biasType].map((question, idx) => ({
        question_id: `${bias.biasType}_Q${idx + 1}`,
        question,
        answer: answersByBias[bias.biasType]?.[idx] ?? null,
      })),
    }));

    setLoadingContext(true);
    try {
      const data = await getBiasContextExplainer({
        dataset_id: datasetId,
        session_id: datasetId,
        top_two_biases: topBiases.map((bias) => ({
          bias_type: bias.biasType,
          score: bias.score,
        })),
        emotional_check_in: emotionalPayload,
      });

      setExplainerData(data);
      antMessage.success("Personalized bias context generated.");
      return true;
    } catch (error: unknown) {
      antMessage.error(getErrorMessage(error, "Failed to generate personalized context."));
      return false;
    } finally {
      setLoadingContext(false);
    }
  };

  return (
    <div className="canvas">
      <div className="grid">
        <div
          className="span-12"
          style={{
            display: "flex",
            gap: "12px",
            alignItems: "center",
            padding: "12px",
            backgroundColor: "var(--panel)",
            borderRadius: "var(--radius)",
            border: "1px solid var(--stroke)",
          }}
        >
          <div style={{ flex: 1, color: "var(--muted)" }}>
            {datasetId ? `Session: ${datasetId}` : "No dataset imported. Please import CSV first."}
          </div>
          <Button
            variant="ghost"
            onClick={hydrateAnswersFromStorage}
            disabled={!datasetId || topBiases.length === 0}
          >
            Reload Saved Answers
          </Button>
          <Button
            variant="primary"
            onClick={() => {
              void generatePersonalizedContext();
            }}
            disabled={!datasetId || loadingContext || topBiases.length === 0}
          >
            {loadingContext ? "Generating..." : "Generate Personalized Context"}
          </Button>
        </div>

        {analysis && topBiases.length > 0 ? (
          <div className="span-12" style={{ display: "grid", gap: "12px", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))" }}>
            {topBiases.map((bias) => (
              <div
                key={bias.biasType}
                style={{
                  padding: "12px",
                  backgroundColor: "var(--panel)",
                  borderRadius: "var(--radius)",
                  border: "1px solid var(--stroke)",
                }}
              >
                <div style={{ fontSize: "14px", fontWeight: 600 }}>{CANONICAL_BIAS_LABELS[bias.biasType]}</div>
                <div style={{ fontSize: "12px", color: "var(--muted)", marginTop: "4px" }}>Score: {bias.score}/100</div>
              </div>
            ))}
          </div>
        ) : null}

        {datasetId && topBiases.length === 0 ? (
          <div
            className="span-12"
            style={{
              padding: "12px",
              backgroundColor: "var(--panel)",
              borderRadius: "var(--radius)",
              border: "1px solid var(--stroke)",
              color: "var(--muted)",
            }}
          >
            No bias scores are available for this session yet.
          </div>
        ) : null}

        {explainerData ? (
          <>
            <div
              className="span-12"
              style={{
                padding: "12px",
                backgroundColor: "var(--panel)",
                borderRadius: "var(--radius)",
                border: "1px solid var(--stroke)",
              }}
            >
              <div className="section__title" style={{ marginBottom: "6px" }}>Real-World Bias Context</div>
              <div style={{ fontSize: "12px", color: "var(--muted)" }}>
                Recent market context is shown first. Personalized emotional context appears below.
              </div>
            </div>
            {explainerData.bias_contexts.map((context) => (
              <div
                key={`context-${context.bias_type}`}
                className="span-6"
                style={{
                  padding: "12px",
                  backgroundColor: "var(--panel)",
                  borderRadius: "var(--radius)",
                  border: "1px solid var(--stroke)",
                }}
              >
                <div className="section__title" style={{ fontSize: "16px" }}>{context.bias_name}</div>
                <div style={{ fontSize: "12px", color: "var(--muted)", marginBottom: "8px" }}>
                  Recent market and economic context (educational).
                </div>

                <div style={{ marginBottom: "12px" }}>
                  <div style={{ fontSize: "12px", fontWeight: 600, marginBottom: "6px" }}>Context Signals</div>
                  {context.market_events.map((event, eventIdx) => (
                    <div key={`${context.bias_type}-event-${eventIdx}`} style={{ fontSize: "12px", marginBottom: "6px" }}>
                      <span style={{ color: "var(--muted)", marginRight: "6px" }}>{event.date}</span>
                      <span>{event.headline}</span>
                    </div>
                  ))}
                </div>

                <div style={{ marginBottom: "10px", fontSize: "13px", lineHeight: 1.5 }}>
                  {context.connection_explanation}
                </div>

                <div
                  style={{
                    fontSize: "13px",
                    lineHeight: 1.5,
                    padding: "8px",
                    backgroundColor: "var(--surfaceA)",
                    borderRadius: "6px",
                  }}
                >
                  {context.practical_takeaway}
                </div>
              </div>
            ))}

            <div
              className="span-12"
              style={{
                padding: "12px",
                backgroundColor: "var(--panel)",
                borderRadius: "var(--radius)",
                border: "1px solid var(--stroke)",
              }}
            >
              <div className="section__title" style={{ marginBottom: "10px" }}>Personalized Bias Context</div>
              <div style={{ display: "grid", gap: "12px", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))" }}>
                {explainerData.personalized_sections.map((section) => (
                  <div
                    key={`personalized-${section.bias_type}`}
                    style={{
                      padding: "12px",
                      border: "1px solid var(--stroke)",
                      borderRadius: "8px",
                      backgroundColor: "var(--surfaceA)",
                    }}
                  >
                    <div style={{ fontWeight: 600, marginBottom: "6px" }}>{section.bias_name}</div>
                    <div style={{ fontSize: "12px", color: "var(--muted)", marginBottom: "10px" }}>{section.headline}</div>
                    <div style={{ fontSize: "13px", lineHeight: 1.5, marginBottom: "10px" }}>{section.supportive_explanation}</div>
                    <div style={{ fontSize: "12px", marginBottom: "8px" }}>
                      Yes: {section.checkin_summary.yes_count} | No: {section.checkin_summary.no_count} | Skip: {section.checkin_summary.skipped_count}
                    </div>
                    <div style={{ fontSize: "12px", fontWeight: 600, marginBottom: "6px" }}>Possible contributors</div>
                    <ul style={{ marginTop: 0, marginBottom: "10px", paddingLeft: "18px", fontSize: "12px", lineHeight: 1.5 }}>
                      {section.hypothetical_contributors.map((item, idx) => (
                        <li key={`${section.bias_type}-contributor-${idx}`}>{item}</li>
                      ))}
                    </ul>
                    <div style={{ fontSize: "12px", fontWeight: 600, marginBottom: "6px" }}>Gentle process habits</div>
                    <ul style={{ marginTop: 0, marginBottom: "10px", paddingLeft: "18px", fontSize: "12px", lineHeight: 1.5 }}>
                      {section.gentle_process_habits.map((habit, idx) => (
                        <li key={`${section.bias_type}-habit-${idx}`}>{habit}</li>
                      ))}
                    </ul>
                    <div style={{ fontSize: "12px", color: "var(--muted)" }}>{section.compassionate_note}</div>
                  </div>
                ))}
              </div>
            </div>

            <div
              className="span-12"
              style={{
                padding: "12px",
                backgroundColor: "var(--panel)",
                borderRadius: "var(--radius)",
                border: "1px solid var(--stroke)",
              }}
            >
              <div style={{ fontSize: "12px", color: "var(--muted)", lineHeight: 1.6 }}>
                <div>{explainerData.methodology}</div>
                <div style={{ marginTop: "6px" }}>{explainerData.global_note}</div>
              </div>
            </div>
          </>
        ) : null}

        {!datasetId ? (
          <div
            className="span-12"
            style={{
              textAlign: "center",
              padding: "40px",
              backgroundColor: "var(--panel)",
              borderRadius: "var(--radius)",
              border: "1px solid var(--stroke)",
            }}
          >
            <div style={{ color: "var(--muted2)" }}>Import a dataset to start emotional check-in and personalized bias context.</div>
          </div>
        ) : null}
      </div>

    </div>
  );
}
function CoachWorkspace({ datasetId }: { datasetId: string | null }) {
  const [chatHistory, setChatHistory] = useState<Array<{ role: "user" | "assistant"; content: string }>>([
    { role: "assistant", content: "Hey what can I help you with today" }
  ]);
  const [inputValue, setInputValue] = useState("");
  const [loading, setLoading] = useState(false);
  const [reflectionNotes] = useState<string[]>([]);
  const chatEndRef = useRef<HTMLDivElement>(null);

  const quickPrompts = [
    "What is CPI?",
    "What's a stop-loss?",
    "Explain diversification",
    "What is a limit order?"
  ];

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [chatHistory]);

  const handleSendMessage = async (message: string) => {
    if (!message.trim() || !datasetId) return;

    // Add user message to history
    const updatedHistory = [...chatHistory, { role: "user" as const, content: message }];
    setChatHistory(updatedHistory);
    setInputValue("");
    setLoading(true);

    try {
      const response = await coachChat({
        dataset_id: datasetId,
        chat_history: updatedHistory,
        reflection_notes: reflectionNotes,
      });

      // Add assistant response
      setChatHistory((prev) => [
        ...prev,
        { role: "assistant", content: response.assistant_response }
      ]);
    } catch (error) {
      console.error("Error:", error);
      setChatHistory((prev) => [
        ...prev,
        { role: "assistant", content: "Sorry, I encountered an error. Please try again." }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleQuickPrompt = (prompt: string) => {
    handleSendMessage(prompt);
  };

  return (
    <div className="canvas">
      <div className="grid">
        {/* Disclaimer Card */}
        <div className="span-12" style={{ padding: "12px", backgroundColor: "var(--panel)", borderRadius: "var(--radius)", border: "1px solid var(--stroke)" }}>
          <div style={{ fontSize: "12px", color: "var(--muted2)", lineHeight: "1.5", padding: "8px" }}>
            <strong>Educational Disclaimer:</strong> This coach provides educational information about trading concepts,
            risk management, and general strategies. It does NOT provide personalized financial advice, specific buy/sell
            recommendations, or investment recommendations. Always consult a financial advisor before making investment
            decisions.
          </div>
        </div>

        {/* Chat Window */}
        <div
          className="span-12"
          style={{
            display: "flex",
            flexDirection: "column",
            height: "560px",
            backgroundColor: "var(--panel)",
            borderRadius: "var(--radius)",
            border: "1px solid var(--stroke)",
            padding: "12px",
          }}
        >
          {/* Messages Container */}
          <div
            style={{
              flex: 1,
              overflowY: "auto",
              marginBottom: "12px",
              paddingRight: "8px",
            }}
          >
            {chatHistory.map((msg, idx) => (
              <div
                key={idx}
                style={{
                  marginBottom: "12px",
                  display: "flex",
                  justifyContent: msg.role === "user" ? "flex-end" : "flex-start",
                }}
              >
                <div
                  style={{
                    maxWidth: "70%",
                    padding: "10px 12px",
                    borderRadius: "8px",
                    backgroundColor:
                      msg.role === "user"
                        ? "var(--teal)"
                        : "var(--surfaceA)",
                    color: msg.role === "user" ? "#000" : "var(--text)",
                    fontSize: "13px",
                    lineHeight: "1.5",
                    wordWrap: "break-word",
                  }}
                >
                  {msg.content}
                </div>
              </div>
            ))}
            {loading && (
              <div style={{ display: "flex", justifyContent: "flex-start", marginBottom: "12px" }}>
                <div
                  style={{
                    padding: "10px 12px",
                    borderRadius: "8px",
                    backgroundColor: "var(--surfaceA)",
                    color: "var(--muted)",
                    fontSize: "13px",
                  }}
                >
                  Thinking...
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          {/* Quick Prompts */}
          <div style={{ marginBottom: "12px", display: "flex", gap: "8px", flexWrap: "wrap" }}>
            {quickPrompts.map((prompt, idx) => (
              <button
                key={idx}
                onClick={() => handleQuickPrompt(prompt)}
                style={{
                  padding: "6px 12px",
                  borderRadius: "4px",
                  border: "1px solid var(--stroke)",
                  backgroundColor: "var(--panel)",
                  color: "var(--muted)",
                  fontSize: "12px",
                  cursor: "pointer",
                  transition: "all 0.2s",
                }}
                onMouseOver={(e) => {
                  e.currentTarget.style.backgroundColor = "var(--surfaceA)";
                  e.currentTarget.style.color = "var(--text)";
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.backgroundColor = "var(--panel)";
                  e.currentTarget.style.color = "var(--muted)";
                }}
              >
                {prompt}
              </button>
            ))}
          </div>

          {/* Input Area */}
          <div style={{ display: "flex", gap: "8px" }}>
            <AntInput
              className="chat__input coachInput"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onPressEnter={() => handleSendMessage(inputValue)}
              placeholder="Ask a finance question..."
              disabled={!datasetId || loading}
              style={{ flex: 1 }}
            />
            <Button
              variant="primary"
              onClick={() => handleSendMessage(inputValue)}
              disabled={!inputValue.trim() || !datasetId || loading}
            >
              Send
            </Button>
          </div>
        </div>

        {/* Empty State */}
        {!datasetId && (
          <div className="span-12" style={{ textAlign: "center", padding: "20px", color: "var(--muted2)" }}>
            Import a dataset first to use the Coach
          </div>
        )}
      </div>
    </div>
  );
}
/* --------------------------- Right Rail + Status Strip (keep yours) --------------------------- */

function StatusStrip({
  biasRisk,
  workspace,
  scoringMode,
  datasetId,
}: {
  biasRisk: number;
  workspace: Workspace;
  scoringMode: ScoringMode;
  datasetId: string | null;
}) {
  return (
    <footer className="status">
      <div className="status__left">
        <span className="status__meta">Mode: {scoringMode === "hybrid" ? "Hybrid (Rules + ML)" : "Rule-only"}</span>
        <span className="status__meta mono">Dataset: {datasetId ?? "none"}</span>
      </div>
      <div className="status__mid">
        <div className="risk">
          <span className="risk__label">Bias Risk ({workspace})</span>
          <div className="risk__bar">
            <div className="risk__fill" style={{ width: `${biasRisk}%` }} />
          </div>
          <span className="risk__num">{biasRisk}%</span>
        </div>
      </div>
      <div className="status__right" />
    </footer>
  );
}

function RightRail({ workspace, selectedTrade }: { workspace: Workspace; selectedTrade: Trade | null }) {
  const [newsReloadKey, setNewsReloadKey] = useState(0);
  const [headlines, setHeadlines] = useState<YahooHeadline[]>([]);
  const [newsLoading, setNewsLoading] = useState(true);
  const [newsError, setNewsError] = useState("");
  const [newsUpdatedAt, setNewsUpdatedAt] = useState<string>("");
  const [newsKeywordDraft, setNewsKeywordDraft] = useState("");
  const [newsKeyword, setNewsKeyword] = useState("");
  const defaultYahooQuery = selectedTrade?.symbol
    ? `${selectedTrade.symbol} market`
    : workspace === "pulse"
      ? "market headlines"
      : "stock market";
  const yahooNewsQuery = newsKeyword || defaultYahooQuery;
  const isKeywordSearchActive = Boolean(newsKeyword.trim());
  const visibleHeadlines = headlines;

  const submitNewsSearch = (rawValue: string) => {
    const nextKeyword = rawValue.trim();
    setNewsKeywordDraft(rawValue);
    setNewsKeyword(nextKeyword);
    setNewsReloadKey((prev) => prev + 1);
  };

  useEffect(() => {
    let active = true;

    const loadYahooHeadlines = async () => {
      if (active) setNewsLoading(true);

      const requestUrl = `${YAHOO_FINANCE_SEARCH_API_URL}?q=${encodeURIComponent(yahooNewsQuery)}&newsCount=8&quotesCount=0&lang=en-US&region=US`;
      const sources = [
        requestUrl,
        `https://api.allorigins.win/raw?url=${encodeURIComponent(requestUrl)}`,
      ];

      try {
        let responseText = "";
        for (const source of sources) {
          try {
            const response = await fetch(source, { cache: "no-store" });
            if (!response.ok) continue;
            responseText = await response.text();
            if (responseText.trim()) break;
          } catch {
            // Try the next source.
          }
        }

        const parsed = parseYahooFinanceSearchResponse(responseText);
        if (!parsed.length) throw new Error("No headlines available");

        if (!active) return;
        setHeadlines(parsed);
        setNewsError("");
        setNewsUpdatedAt(new Date().toISOString());
      } catch {
        if (!active) return;
        setHeadlines([]);
        setNewsError("Live Yahoo Finance feed unavailable right now.");
      } finally {
        if (active) setNewsLoading(false);
      }
    };

    void loadYahooHeadlines();
    const intervalId = window.setInterval(() => {
      void loadYahooHeadlines();
    }, 10 * 60 * 1000);

    return () => {
      active = false;
      window.clearInterval(intervalId);
    };
  }, [newsReloadKey, yahooNewsQuery]);

  const yahooNewsSection = (
    <div className="yahooNews">
      <div className="yahooNews__top">
        <div className="section__title">Yahoo Finance Daily</div>
        <div className="yahooNews__actions">
          {newsUpdatedAt ? (
            <span className="yahooNews__stamp">Updated {new Date(newsUpdatedAt).toLocaleTimeString()}</span>
          ) : null}
          <AntButton
            type="text"
            size="small"
            className="yahooNews__refresh"
            onClick={() => setNewsReloadKey((prev) => prev + 1)}
            disabled={newsLoading}
          >
            Refresh
          </AntButton>
        </div>
      </div>
      <AntInput
        className="yahooNews__search"
        size="small"
        placeholder="Search news by keyword"
        value={newsKeywordDraft}
        onChange={(event) => {
          const next = event.target.value;
          setNewsKeywordDraft(next);
          if (!next.trim() && newsKeyword) {
            setNewsKeyword("");
            setNewsReloadKey((prev) => prev + 1);
          }
        }}
        onPressEnter={(event) => {
          submitNewsSearch((event.target as HTMLInputElement).value);
        }}
        suffix={
          <button
            type="button"
            className="yahooNews__searchTrigger"
            aria-label="Search Yahoo Finance news"
            onMouseDown={(event) => event.preventDefault()}
            onClick={() => submitNewsSearch(newsKeywordDraft)}
          >
            <SearchOutlined />
          </button>
        }
      />
      <div className="yahooNews__query muted">
        Results for: <span className="mono">{yahooNewsQuery}</span>
      </div>

      <div className="yahooNews__list">
        {visibleHeadlines.map((headline) => (
          <a
            key={headline.id}
            className="yahooNews__item"
            href={headline.link}
            target="_blank"
            rel="noreferrer"
          >
            <div className="yahooNews__title">{headline.title}</div>
            <div className="yahooNews__meta">{formatHeadlineAge(headline.publishedAt)} - Yahoo Finance</div>
          </a>
        ))}
      </div>

      {newsLoading ? <div className="muted">Loading market headlines...</div> : null}
      {!newsLoading && !newsError && visibleHeadlines.length === 0 ? (
        <div className="muted">
          {isKeywordSearchActive
            ? "No Yahoo Finance headlines returned for the current query."
            : "No latest Yahoo Finance headlines are available right now."}
        </div>
      ) : null}
      {newsError ? <div className="muted">{newsError}</div> : null}
    </div>
  );

  if (workspace === "replay") {
    return (
      <aside className="rail" aria-label="Context rail">
        <Card title="Adventure Practice">
          <div className="newsItem">
            <div className="newsItem__title">Test your trading wisdom</div>
            <div className="newsItem__meta">4 real-world scenarios for your top 2 biases.</div>
          </div>
          <div className="newsItem">
            <div className="newsItem__title">Learn from patterns</div>
            <div className="newsItem__meta">Each answer reveals how this bias appears in your data.</div>
          </div>
          <div className="newsItem">
            <div className="newsItem__title">Build better habits</div>
            <div className="newsItem__meta">The quiz ends with one small habit to practice today.</div>
          </div>

          <div className="divider" />

          <div className="section__title">How to Play</div>
          <div className="muted">
            You'll see 4 trading scenarios. Pick the best response. Then learn why the correct answer protects your PnL.
          </div>
        </Card>
      </aside>
    );
  }

  return (
    <aside className="rail" aria-label="Context rail">
      <Card title="Market Context" right={<Pill tone="muted">Live</Pill>}>
        {yahooNewsSection}
      </Card>
    </aside>
  );
}

function HomeEmotionalCheckInModal({
  open,
  datasetId,
  topBiases,
  answersByBias,
  onClose,
  onAnswer,
}: {
  open: boolean;
  datasetId: string | null;
  topBiases: TopBiasEntry[];
  answersByBias: Record<CanonicalBiasType, EmotionalCheckInAnswer[]>;
  onClose: () => void;
  onAnswer: (biasType: CanonicalBiasType, questionIndex: number, answer: EmotionalCheckInAnswer) => void;
}) {
  if (!open || !datasetId || topBiases.length === 0) return null;

  const answeredCount = topBiases.reduce((count, bias) => {
    const answers = answersByBias[bias.biasType] ?? [null, null, null];
    return count + answers.filter((answer) => answer !== null).length;
  }, 0);
  const totalCount = topBiases.length * 3;
  const choices: Array<{ label: string; value: EmotionalCheckInAnswer }> = [
    { label: "Yes", value: "YES" },
    { label: "No", value: "NO" },
    { label: "Skip", value: null },
  ];

  return (
    <div
      className="modalOverlay"
      role="dialog"
      aria-modal="true"
      aria-label="Emotional check-in modal"
      onMouseDown={onClose}
    >
      <div
        className="modal"
        style={{ maxWidth: "980px", width: "min(980px, 96vw)" }}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="modalHdr">
          <div className="modalHdr__left">
            <div className="modalTitle">Emotional Check-In</div>
            <div className="modalSub muted">
              Answer Yes, No, or Skip for each pre-made question. Answers are saved locally by session.
            </div>
          </div>
          <AntButton className="modalClose" onClick={onClose} aria-label="Close" type="text" icon={<Icon name="close" />} />
        </div>

        <div className="modalBody">
          <div style={{ display: "grid", gap: "12px", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))" }}>
            {topBiases.map((bias) => (
              <div
                key={`home-checkin-${bias.biasType}`}
                style={{
                  padding: "12px",
                  backgroundColor: "var(--panel)",
                  borderRadius: "var(--radius)",
                  border: "1px solid var(--stroke)",
                }}
              >
                <div className="section__title" style={{ fontSize: "15px" }}>
                  {CANONICAL_BIAS_LABELS[bias.biasType]}
                </div>
                <div style={{ fontSize: "12px", color: "var(--muted)", marginBottom: "10px" }}>Score: {bias.score}/100</div>

                {EMOTIONAL_CHECKIN_QUESTIONS[bias.biasType].map((question, idx) => {
                  const selected = answersByBias[bias.biasType]?.[idx] ?? null;
                  return (
                    <div
                      key={`home-checkin-${bias.biasType}-${idx}`}
                      style={{
                        marginBottom: "12px",
                        padding: "10px",
                        border: "1px solid var(--stroke)",
                        borderRadius: "8px",
                        backgroundColor: "var(--surfaceA)",
                      }}
                    >
                      <div style={{ fontSize: "13px", marginBottom: "8px", lineHeight: 1.4 }}>{idx + 1}. {question}</div>
                      <div style={{ display: "flex", gap: "8px" }}>
                        {choices.map((choice) => {
                          const active = selected === choice.value;
                          return (
                            <button
                              key={`home-checkin-${bias.biasType}-${idx}-${choice.label}`}
                              type="button"
                              onClick={() => onAnswer(bias.biasType, idx, choice.value)}
                              style={{
                                border: "1px solid",
                                borderColor: active ? "rgba(111, 255, 233, 0.6)" : "var(--stroke)",
                                background: active ? "rgba(111, 255, 233, 0.15)" : "var(--panel)",
                                color: "var(--text)",
                                padding: "6px 10px",
                                borderRadius: "6px",
                                cursor: "pointer",
                                fontSize: "12px",
                              }}
                            >
                              {choice.label}
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
        </div>

        <div className="modalFtr">
          <div className="modalFtr__left muted">
            Progress: <b>{answeredCount}</b> / <b>{totalCount}</b> answered
          </div>
          <div className="modalFtr__right">
            <Button variant="ghost" onClick={onClose} className="modalCancelBtn">
              Close
            </Button>
            <Button variant="primary" onClick={onClose} className="modalImportBtn">
              Continue
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
/* --------------------------- App Shell --------------------------- */

export default function App() {
  const [workspace, setWorkspace] = useState<Workspace>("command");
  const [selectedTrade, setSelectedTrade] = useState<Trade | null>(SAMPLE_TRADES[0]);
  const [scoringMode, setScoringMode] = useState<ScoringMode>("hybrid");
  const [datasetId, setDatasetId] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisOutput | null>(null);
  const topBiases = useMemo(() => getTopTwoBiasesFromScores(analysis?.bias_scores), [analysis]);
  const [apiBusy, setApiBusy] = useState(false);
  const [apiError, setApiError] = useState<string>("");

  const [csvModalOpen, setCsvModalOpen] = useState(false);
  const [homeCheckinModalOpen, setHomeCheckinModalOpen] = useState(false);
  const [homeCheckinAnswers, setHomeCheckinAnswers] = useState<Record<CanonicalBiasType, EmotionalCheckInAnswer[]>>(
    createEmptyCheckinAnswerMap
  );
  const skipNextAnalysisEffectRef = useRef(false);
  const previousWorkspaceRef = useRef<Workspace>("command");

  useEffect(() => {
    if (!datasetId) return;

    if (skipNextAnalysisEffectRef.current) {
      skipNextAnalysisEffectRef.current = false;
      return;
    }

    let active = true;

    const rerunAnalysis = async () => {
      setApiBusy(true);
      try {
        const refreshed = await analyzeDataset({ dataset_id: datasetId, scoring_mode: scoringMode });
        if (!active) return;
        setAnalysis(refreshed);
        setApiError("");
      } catch (error: unknown) {
        if (!active) return;
        setApiError(getErrorMessage(error, "Failed to analyze dataset with selected scoring mode."));
      } finally {
        if (active) setApiBusy(false);
      }
    };

    void rerunAnalysis();
    return () => {
      active = false;
    };
  }, [datasetId, scoringMode]);

  const biasRisk = (() => {
    if (analysis?.behavior_index != null) return Math.round(analysis.behavior_index);
    const base = workspace === "replay" ? 68 : 75;
    return base;
  })();

  useEffect(() => {
    if (!datasetId) {
      setHomeCheckinAnswers(createEmptyCheckinAnswerMap());
      setHomeCheckinModalOpen(false);
      return;
    }

    const nextAnswers = createEmptyCheckinAnswerMap();
    topBiases.forEach((bias) => {
      nextAnswers[bias.biasType] = readStoredBiasAnswers(datasetId, bias.biasType);
    });
    setHomeCheckinAnswers(nextAnswers);
  }, [datasetId, topBiases]);

  useEffect(() => {
    const previousWorkspace = previousWorkspaceRef.current;
    const enteredBiasContext = workspace === "pulse" && previousWorkspace !== "pulse";

    if (enteredBiasContext && datasetId && topBiases.length > 0) {
      setHomeCheckinModalOpen(true);
    }

    previousWorkspaceRef.current = workspace;
  }, [workspace, datasetId, topBiases.length]);

  const tradesWorkspaceRows = useMemo<Trade[]>(() => {
    const biasMap: Record<string, DisplayBiasType> = {
      "overtrading": "Overtrading",
      "loss_aversion": "Loss Aversion",
      "revenge_trading": "Revenge Trading",
      "recency_bias": "Recency Bias",
    };
    return (analysis?.flagged_trades ?? []).map((ft) => ({
      id: ft.trade_id || "",
      time: ft.timestamp
        ? new Date(ft.timestamp).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })
        : "",
      symbol: ft.symbol || "",
      side: (ft.side || "Buy") as "Buy" | "Sell",
      size: ft.quantity || 0,
      durationMin: 0,
      pnl: ft.profit_loss || 0,
      balance: ft.balance,
      flags: [biasMap[ft.bias] || ft.bias] as DisplayBiasType[],
      confidence: ft.confidence || 0,
      evidence: ft.evidence?.join(" | ") || "",
    }));
  }, [analysis?.flagged_trades]);

  useEffect(() => {
    if (tradesWorkspaceRows.length === 0) {
      if (selectedTrade !== null) setSelectedTrade(null);
      return;
    }
    const selectedId = selectedTrade?.id;
    if (!selectedId || !tradesWorkspaceRows.some((trade) => trade.id === selectedId)) {
      setSelectedTrade(tradesWorkspaceRows[0]);
    }
  }, [tradesWorkspaceRows, selectedTrade?.id]);

  const updateHomeCheckinAnswer = (biasType: CanonicalBiasType, questionIndex: number, answer: EmotionalCheckInAnswer) => {
    if (!datasetId) return;

    setHomeCheckinAnswers((previous) => {
      const current = previous[biasType] ?? [null, null, null];
      const updatedAnswers: EmotionalCheckInAnswer[] = [current[0] ?? null, current[1] ?? null, current[2] ?? null];
      updatedAnswers[questionIndex] = answer;
      writeStoredBiasAnswers(datasetId, biasType, updatedAnswers);
      return {
        ...previous,
        [biasType]: updatedAnswers,
      };
    });
  };

  const handleImported = async (payload: CsvImportPayload) => {
    setApiBusy(true);
    setApiError("");

    try {
      const imported = await importDatasetFile(payload.file, {
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC",
        session_template: "equities_rth",
      });
      skipNextAnalysisEffectRef.current = true;
      setDatasetId(imported.dataset_id);
      const analyzed = await analyzeDataset({ dataset_id: imported.dataset_id, scoring_mode: scoringMode });
      setAnalysis(analyzed);
      setHomeCheckinModalOpen(false);
      const importedRows = imported.stats?.rows ?? 0;
      antMessage.success(
        `Imported ${importedRows.toLocaleString()} rows from ${payload.file.name}. Behavior Index ${Math.round(analyzed.behavior_index)}.`
      );
    } catch (error: unknown) {
      skipNextAnalysisEffectRef.current = false;
      setDatasetId(null);
      setAnalysis(null);
      setHomeCheckinModalOpen(false);
      setApiError(getErrorMessage(error, "Import failed. Backend processing is required."));
      antMessage.error(`Failed to import ${payload.file.name}. Check backend logs and CSV format.`);
    } finally {
      setApiBusy(false);
    }
  };

  return (
    <div className="app">
      <TerminalBar
        scoringMode={scoringMode}
        onScoringModeChange={setScoringMode}
        apiBusy={apiBusy}
        onImport={() => setCsvModalOpen(true)}
      />

      <div className="body">
        <Dock active={workspace} onPick={setWorkspace} />

        <main className="main">
          {analysis ? (
            <div className="importBanner">
              <Pill tone="ok">Analyzed</Pill>
              <span className="muted">
                Dataset: <span className="mono">{datasetId}</span> - Behavior Index:{" "}
                <span className="mono">{Math.round(analysis.behavior_index)}</span>
              </span>
              <span className="muted">
                OT <span className="mono">{Math.round(analysis.bias_scores.overtrading)}</span> - LA{" "}
                <span className="mono">{Math.round(analysis.bias_scores.loss_aversion)}</span> - RT{" "}
                <span className="mono">{Math.round(analysis.bias_scores.revenge_trading)}</span>
              </span>
            </div>
          ) : (
            <div className="importBanner importBanner--empty">
              <Pill tone="muted">No dataset</Pill>
              <span className="muted">Import a CSV and run analysis to populate behavior scores.</span>
            </div>
          )}

          {apiError ? <AntAlert className="alert alert--warn" message={apiError} type="warning" showIcon /> : null}

          {workspace === "command" && (
            <CommandWorkspace
              behaviorIndex={biasRisk}
              biasScores={analysis?.bias_scores}
              analysis={analysis}
            />
          )}
          {workspace === "trades" && (
            <TradesWorkspace
              trades={tradesWorkspaceRows}
              selected={selectedTrade}
              onSelect={(t) => setSelectedTrade(t)}
              flaggedTradesRaw={analysis?.flagged_trades}
              biasScores={analysis?.bias_scores}
            />
          )}
          {workspace === "simulator" && <SimulatorWorkspace analysisData={analysis} datasetId={datasetId} />}
          {workspace === "replay" && (
            <PracticeWorkspace 
              datasetId={datasetId}
              scoringMode={scoringMode}
              analysis={analysis}
            />
          )}
          {workspace === "pulse" && <PulseWorkspace datasetId={datasetId} analysis={analysis} />}
          {workspace === "coach" && <CoachWorkspace datasetId={datasetId} />}
        </main>

        <RightRail workspace={workspace} selectedTrade={selectedTrade} />
      </div>

      <StatusStrip
        biasRisk={biasRisk}
        workspace={workspace}
        scoringMode={scoringMode}
        datasetId={datasetId}
      />

      <CsvUploadModal
        open={csvModalOpen}
        onClose={() => setCsvModalOpen(false)}
        onImported={handleImported}
      />
      <HomeEmotionalCheckInModal
        open={homeCheckinModalOpen}
        datasetId={datasetId}
        topBiases={topBiases}
        answersByBias={homeCheckinAnswers}
        onClose={() => setHomeCheckinModalOpen(false)}
        onAnswer={updateHomeCheckinAnswer}
      />
    </div>
  );
}





