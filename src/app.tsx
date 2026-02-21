import React, { useEffect, useState, useRef } from "react";
import { LeftOutlined, RightOutlined, SearchOutlined } from "@ant-design/icons";
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
import { analyzeDataset, importDataset, simulateDataset, generatePracticeQuestions, getBiasContextExplainer, coachChat } from "./api/signalforge";
import type {
  AnalysisOutput,
  BiasKey,
  CanonicalTradeField,
  FlaggedTrade,
  ImportDatasetRequest,
  PracticeQuestion,
  PracticeQuestionsResponse,
  ScoringMode,
} from "./types/signalforge";

const { TextArea: AntTextArea } = AntInput;

/* --------------------------- Types --------------------------- */

type Workspace =
  | "command"
  | "trades"
  | "heatmap"
  | "simulator"
  | "replay"
  | "pulse"
  | "coach";

type BiasType = "Overtrading" | "Loss Aversion" | "Revenge Trading" | "Recency Bias";

type Trade = {
  id: string;
  time: string;
  symbol: string;
  side: "Buy" | "Sell";
  size: number;
  durationMin: number;
  pnl: number;
  flags: BiasType[];
  confidence: number; // 0..100
  evidence: string;
};

type YahooHeadline = {
  id: string;
  title: string;
  link: string;
  publishedAt: string;
};

type CanonicalField =
  Exclude<CanonicalTradeField, "trade_id">;

type CsvImportPayload = {
  rawText: string;
  headers: string[];
  rows: Record<string, string>[];
  mapping: Partial<Record<CanonicalField, string>>;
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

function normalizeHeader(h: string) {
  return h.trim().toLowerCase().replace(/\s+/g, "_");
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

/**
 * Robust-ish CSV parse for typical broker exports:
 * - supports quoted fields with commas
 * - supports CRLF / LF
 * - first row is header
 * Returns rows as objects keyed by header names (original header strings).
 */
function parseCsv(text: string, maxRows: number = 5000): { headers: string[]; rows: Record<string, string>[] } {
  // Remove UTF-8 BOM if present
  const input = text.replace(/^\uFEFF/, "");

  const rows: string[][] = [];
  let cur: string[] = [];
  let field = "";
  let inQuotes = false;

  const pushField = () => {
    cur.push(field);
    field = "";
  };

  const pushRow = () => {
    // ignore fully empty trailing lines
    const allEmpty = cur.every((c) => c.trim() === "");
    if (!allEmpty) rows.push(cur);
    cur = [];
  };

  for (let i = 0; i < input.length; i++) {
    const ch = input[i];
    const next = input[i + 1];

    if (inQuotes) {
      if (ch === '"' && next === '"') {
        // Escaped quote
        field += '"';
        i++;
      } else if (ch === '"') {
        inQuotes = false;
      } else {
        field += ch;
      }
    } else {
      if (ch === '"') {
        inQuotes = true;
      } else if (ch === ",") {
        pushField();
      } else if (ch === "\n") {
        pushField();
        pushRow();
        if (rows.length >= maxRows + 1) break; // +1 because header row
      } else if (ch === "\r") {
        // handle CRLF by skipping \r (newline handled on \n)
      } else {
        field += ch;
      }
    }
  }
  // last field/row
  pushField();
  pushRow();

  if (rows.length === 0) return { headers: [], rows: [] };

  const headers = rows[0].map((h) => h.trim());
  const dataRows = rows.slice(1);

  const objects: Record<string, string>[] = dataRows.map((r) => {
    const obj: Record<string, string> = {};
    for (let i = 0; i < headers.length; i++) {
      obj[headers[i]] = (r[i] ?? "").trim();
    }
    return obj;
  });

  return { headers, rows: objects };
}

function guessMapping(headers: string[]): Partial<Record<CanonicalField, string>> {
  const hnorm = headers.map((h) => ({ raw: h, n: normalizeHeader(h) }));

  const find = (cands: string[]) => {
    const set = new Set(cands);
    const hit = hnorm.find((h) => set.has(h.n));
    return hit?.raw;
  };

  // Common aliases across platforms
  return {
    timestamp: find(["timestamp", "time", "datetime", "date_time", "filled_time", "execution_time"]),
    open_time: find(["open_time", "entry_time", "start_time"]),
    close_time: find(["close_time", "exit_time", "end_time"]),
    symbol: find(["symbol", "ticker", "instrument", "asset", "product"]),
    side: find(["side", "action", "buy_sell", "direction"]),
    size: find(["size", "qty", "quantity", "shares", "units", "amount"]),
    entry_price: find(["entry_price", "open_price", "avg_entry_price", "price_in", "buy_price"]),
    exit_price: find(["exit_price", "close_price", "avg_exit_price", "price_out", "sell_price"]),
    pnl: find(["pnl", "profit", "profit_loss", "realized_pnl", "realized_profit", "net_pnl"]),
    fees: find(["fees", "commission", "commissions", "fee"]),
  };
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
    { id: "heatmap", label: "Danger", icon: "heatmap" },
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
          popupClassName="appSelectDropdown"
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
  const [rawText, setRawText] = useState<string>("");
  const [fileName, setFileName] = useState<string>("");
  const [headers, setHeaders] = useState<string[]>([]);
  const [rows, setRows] = useState<Record<string, string>[]>([]);
  const [err, setErr] = useState<string>("");

  const [mapping, setMapping] = useState<Partial<Record<CanonicalField, string>>>({});

  const previewRows = rows.slice(0, 10);

  const required: CanonicalField[] = ["symbol", "side"];
  const hasSomeTime = Boolean(mapping.timestamp || (mapping.open_time && mapping.close_time));
  const canImport = headers.length > 0 && required.every((f) => mapping[f]) && hasSomeTime;

  const reset = () => {
    setDragOver(false);
    setRawText("");
    setFileName("");
    setHeaders([]);
    setRows([]);
    setErr("");
    setMapping({});
  };

  const close = () => {
    reset();
    onClose();
  };

  const loadText = async (file: File) => {
    setErr("");
    setFileName(file.name);

    const text = await file.text();
    setRawText(text);

    try {
      const parsed = parseCsv(text, 5000);
      if (!parsed.headers.length) throw new Error("No headers found.");
      setHeaders(parsed.headers);
      setRows(parsed.rows);

      const guess = guessMapping(parsed.headers);
      setMapping(guess);
    } catch (error: unknown) {
      setErr(getErrorMessage(error, "Failed to parse CSV."));
      setHeaders([]);
      setRows([]);
      setMapping({});
    }
  };

  const onFilePick = async (f: File | null) => {
    if (!f) return;
    if (!f.name.toLowerCase().endsWith(".csv")) {
      setErr("Please upload a .csv file.");
      return;
    }
    await loadText(f);
  };

  const onDrop = async (ev: React.DragEvent) => {
    ev.preventDefault();
    setDragOver(false);
    const f = ev.dataTransfer.files?.[0];
    await onFilePick(f ?? null);
  };

  const onImport = () => {
    if (!canImport) return;
    onImported({ rawText, headers, rows, mapping });
    close();
  };

  if (!open) return null;

  const fieldLabels: Array<{ field: CanonicalField; label: string; hint: string; optional?: boolean }> = [
    { field: "timestamp", label: "Timestamp", hint: "Single execution timestamp", optional: true },
    { field: "open_time", label: "Open time", hint: "Entry/open timestamp", optional: true },
    { field: "close_time", label: "Close time", hint: "Exit/close timestamp", optional: true },
    { field: "symbol", label: "Symbol", hint: "Ticker / instrument", optional: false },
    { field: "side", label: "Side", hint: "Buy/Sell or Long/Short", optional: false },
    { field: "size", label: "Size", hint: "Qty / shares / units", optional: true },
    { field: "entry_price", label: "Entry price", hint: "Average entry", optional: true },
    { field: "exit_price", label: "Exit price", hint: "Average exit", optional: true },
    { field: "pnl", label: "PnL", hint: "Profit/Loss", optional: true },
    { field: "fees", label: "Fees", hint: "Commissions/fees", optional: true },
  ];
  const previewColumnKeys = headers.slice(0, 8);
  const mappingSelectStyle = {
    ["--ant-select-background-color" as string]: "rgba(5,8,14,.9)",
    ["--ant-select-border-color" as string]: "rgba(255,255,255,.16)",
    ["--ant-select-color" as string]: "rgba(232,240,255,.94)",
  } as React.CSSProperties;
  const previewColumns = previewColumnKeys.map((header, index) => ({
    title: header,
    dataIndex: header,
    key: header,
    ellipsis: true,
    width: index === 0 ? 170 : undefined,
    render: (value: string) => <span className="mono previewTable__cell">{(value ?? "").slice(0, 40)}</span>,
  }));
  const previewData = previewRows.map((row, index) => ({ key: `preview-${index}`, ...row }));

  return (
    <div className="modalOverlay" role="dialog" aria-modal="true" aria-label="CSV import modal" onMouseDown={close}>
      <div className="modal" onMouseDown={(e) => e.stopPropagation()}>
        <div className="modalHdr">
          <div className="modalHdr__left">
            <div className="modalTitle">Import Trade Log (CSV)</div>
            <div className="modalSub muted">
              Upload a broker export. We'll detect columns and preview rows before importing.
            </div>
          </div>
          <AntButton className="modalClose" onClick={close} aria-label="Close" type="text" icon={<Icon name="close" />} />
        </div>

        <div className="modalBody">
          {/* Dropzone */}
          <div
            className={cn("dropzone", dragOver && "dropzone--over", headers.length > 0 && "dropzone--loaded")}
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
                {fileName ? (
                  <>
                    Loaded <span className="mono">{fileName}</span>
                  </>
                ) : (
                  "Drag & drop your CSV here"
                )}
              </div>
              <div className="muted">
                {fileName ? `${rows.length.toLocaleString()} rows detected` : "or choose a file from your computer"}
              </div>
            </div>

            <AntUpload
              accept=".csv,text/csv"
              showUploadList={false}
              beforeUpload={(file) => {
                void onFilePick(file);
                return false;
              }}
            >
              <AntButton className="fileBtn">Browse CSV</AntButton>
            </AntUpload>
          </div>

          {err ? <AntAlert className="alert alert--danger" message={`Import error: ${err}`} type="error" showIcon /> : null}

          {/* If loaded: Mapping + Preview */}
          {headers.length > 0 ? (
            <div className="modalGrid">
              <div className="mapCard">
                <div className="mapHdr">
                  <div className="mapTitle">Column Mapping</div>
                  <div className="muted">
                    Required: <b>Symbol</b>, <b>Side</b>, and <b>(Timestamp or Open+Close)</b>
                  </div>
                </div>

                <div className="mapList">
                  {fieldLabels.map((f) => {
                    const isRequired = !f.optional && (f.field === "symbol" || f.field === "side");

                    return (
                      <div className="mapRow" key={f.field}>
                        <div className="mapLeft">
                          <div className="mapLabel">
                            {f.label}{" "}
                            {isRequired ? <span className="req">*</span> : <span className="opt">optional</span>}
                          </div>
                          <div className="mapHint muted">{f.hint}</div>
                        </div>

                        <div className="mapRight">
                          <AntSelect
                            className={cn("topbar__select mapSelect", isRequired && !mapping[f.field] && "mapSelect--warn")}
                            popupClassName="appSelectDropdown"
                            getPopupContainer={(trigger) =>
                              (trigger.closest(".modal") as HTMLElement | null) ?? document.body
                            }
                            style={mappingSelectStyle}
                            value={mapping[f.field]}
                            allowClear
                            placeholder="Not mapped"
                            dropdownMatchSelectWidth
                            options={headers.map((h) => ({ label: h, value: h }))}
                            onChange={(value) => setMapping((m) => ({ ...m, [f.field]: (value as string) || undefined }))}
                          />
                          {isRequired && !mapping[f.field] ? <Pill tone="danger">Required</Pill> : null}
                          {!isRequired && !mapping[f.field] ? <Pill tone="muted">-</Pill> : null}
                        </div>
                      </div>
                    );
                  })}
                </div>

                <div className="mapFooter">
                  {!hasSomeTime ? (
                    <AntAlert
                      className="alert alert--warn"
                      message="Map Timestamp or both Open time and Close time."
                      type="warning"
                      showIcon
                    />
                  ) : null}
                </div>
              </div>

              <div className="previewCard">
                <div className="previewHdr">
                  <div className="mapTitle">Preview (first 10 rows)</div>
                  <div className="muted">{headers.length} columns</div>
                </div>

                <div className="previewTableWrap">
                  <AntTable
                    className="previewTable"
                    size="small"
                    columns={previewColumns}
                    dataSource={previewData}
                    pagination={false}
                    tableLayout="fixed"
                    scroll={{ y: 360 }}
                  />
                </div>

                <div className="muted" style={{ marginTop: 10 }}>
                  You can send <span className="mono">rawText</span> to your backend, or normalize rows using the mapping.
                </div>
              </div>
            </div>
          ) : null}
        </div>

        <div className="modalFtr">
          <div className="modalFtr__left muted">
            {headers.length > 0 ? (
              <>
                Ready: <b>{canImport ? "Yes" : "No"}</b> - Rows: <b>{rows.length.toLocaleString()}</b>
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

type HeatmapMode = "bias" | "pnl";

type PnlCalendarEntry = {
  key: string;
  date: Date;
  dayOfMonth: number;
  pnl: number;
};

function formatPnlCurrency(value: number) {
  const currency = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
  return value > 0 ? `+${currency.format(value)}` : currency.format(value);
}

function resolvePnlToneClass(value: number) {
  const abs = Math.abs(value);
  const strength = abs >= 450 ? "3" : abs >= 280 ? "2" : abs >= 120 ? "1" : "0";
  if (value > 0) return `pnlCell--pos${strength}`;
  if (value < 0) return `pnlCell--neg${strength}`;
  return "pnlCell--flat";
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

type TimelineTradePoint = {
  timestamp: string;
  pnl: number;
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
  tooltip: React.ReactNode;
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
  flaggedTrades?: AnalysisOutput["flagged_trades"]
): ChartGranularity {
  const timelineDates = (tradeTimeline ?? [])
    .map((point) => parseTimestampToDate(point.timestamp))
    .filter((date): date is Date => date !== null);

  const fallbackDates = (flaggedTrades ?? [])
    .map((trade) => parseTimestampToDate(trade.timestamp))
    .filter((date): date is Date => date !== null);

  const dates = timelineDates.length > 0 ? timelineDates : fallbackDates;
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
  flaggedTrades?: AnalysisOutput["flagged_trades"]
): TimelineTradePoint[] {
  const directTimeline = (tradeTimeline ?? [])
    .filter((point) => typeof point.timestamp === "string" && Number.isFinite(point.pnl))
    .map((point) => ({ timestamp: point.timestamp, pnl: Number(point.pnl) }));

  if (directTimeline.length > 0) {
    return directTimeline.sort((a, b) => a.timestamp.localeCompare(b.timestamp));
  }

  return (flaggedTrades ?? [])
    .filter((trade) => typeof trade.timestamp === "string" && Number.isFinite(trade.profit_loss))
    .map((trade) => ({ timestamp: trade.timestamp!, pnl: Number(trade.profit_loss ?? 0) }))
    .sort((a, b) => a.timestamp.localeCompare(b.timestamp));
}

function buildSnapshotBuckets(
  tradeTimeline?: AnalysisOutput["trade_timeline"],
  flaggedTrades?: AnalysisOutput["flagged_trades"],
  granularity: ChartGranularity = "1h"
) : SnapshotBucket[] {
  const source = buildTimelineFromAnalysis(tradeTimeline, flaggedTrades);
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
    bucket.trades += 1;
    bucket.pnl += Number(point.pnl);
    if (point.pnl > 0) bucket.wins += 1;
    else if (point.pnl < 0) bucket.losses += 1;
    else bucket.flat += 1;
  }

  for (const trade of flaggedTrades ?? []) {
    const parsed = parseTimestampToDate(trade.timestamp);
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
  const output: SnapshotBucket[] = [];
  let cumulativePnl = 0;

  for (const bucket of ordered) {
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

function buildDailyPnlSeries(
  dailyPnl?: AnalysisOutput["daily_pnl"],
  tradeTimeline?: AnalysisOutput["trade_timeline"],
  flaggedTrades?: AnalysisOutput["flagged_trades"]
): AnalysisOutput["daily_pnl"] {
  const normalizedDailyMap = new Map<string, number>();
  if (Array.isArray(dailyPnl)) {
    for (const point of dailyPnl) {
      if (!point || typeof point.date !== "string" || !Number.isFinite(point.pnl)) continue;
      const dateKey = point.date.slice(0, 10);
      if (!dateKey) continue;
      normalizedDailyMap.set(dateKey, (normalizedDailyMap.get(dateKey) ?? 0) + Number(point.pnl));
    }
  }

  if (normalizedDailyMap.size > 0) {
    return [...normalizedDailyMap.entries()]
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([date, pnl]) => ({ date, pnl: Number(pnl.toFixed(2)) }));
  }

  const timeline = buildTimelineFromAnalysis(tradeTimeline, flaggedTrades);
  const fallbackMap = new Map<string, number>();
  for (const point of timeline) {
    const parsed = parseTimestampToDate(point.timestamp);
    if (!parsed) continue;
    const dateKey = `${parsed.getFullYear()}-${String(parsed.getMonth() + 1).padStart(2, "0")}-${String(parsed.getDate()).padStart(2, "0")}`;
    fallbackMap.set(dateKey, (fallbackMap.get(dateKey) ?? 0) + Number(point.pnl));
  }

  return [...fallbackMap.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([date, pnl]) => ({ date, pnl: Number(pnl.toFixed(2)) }));
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

function buildMonthlyPnlCalendar(
  referenceDate: Date,
  dailyPnl?: AnalysisOutput["daily_pnl"]
): Array<PnlCalendarEntry | null> {
  const year = referenceDate.getFullYear();
  const month = referenceDate.getMonth();
  const firstDay = new Date(year, month, 1);
  const daysInMonth = new Date(year, month + 1, 0).getDate();
  const mondayFirstOffset = (firstDay.getDay() + 6) % 7; // convert Sunday=0 to Monday-last layout

  const monthPnlByDay = new Map<number, number>();
  if (Array.isArray(dailyPnl) && dailyPnl.length > 0) {
    for (const point of dailyPnl) {
      if (!point?.date) continue;
      const parsed = new Date(`${point.date}T00:00:00`);
      if (Number.isNaN(parsed.getTime())) continue;
      if (parsed.getFullYear() !== year || parsed.getMonth() !== month) continue;
      monthPnlByDay.set(parsed.getDate(), point.pnl);
    }
  }

  const hasBackendPnl = monthPnlByDay.size > 0;
  const entries: Array<PnlCalendarEntry | null> = [];

  for (let i = 0; i < mondayFirstOffset; i++) {
    entries.push(null);
  }

  for (let day = 1; day <= daysInMonth; day++) {
    const date = new Date(year, month, day);
    const weekday = (date.getDay() + 6) % 7;

    const pnl = hasBackendPnl
      ? Math.round(monthPnlByDay.get(day) ?? 0)
      : Math.round(Math.sin(day * 0.82) * 245 + (2 - weekday) * 34 + (day % 5 === 0 ? -92 : 74));

    entries.push({
      key: date.toISOString(),
      date,
      dayOfMonth: day,
      pnl,
    });
  }

  while (entries.length % 7 !== 0) {
    entries.push(null);
  }

  return entries;
}

function buildLocalFallbackAnalysis(): AnalysisOutput {
  return {
    behavior_index: 75,
    bias_scores: {
      overtrading: 72,
      loss_aversion: 79,
      revenge_trading: 86,
      recency_bias: 45,
    },
    top_triggers: ["Rapid re-entry after loss", "Oversize after drawdown", "Trade clustering in open hour"],
    danger_hours: normalizeDangerHoursMatrix(DANGER_MATRIX),
    daily_pnl: [],
    trade_timeline: [
      { timestamp: "2024-03-31T10:41:00", pnl: 228 },
      { timestamp: "2024-03-31T11:03:00", pnl: -78 },
      { timestamp: "2024-03-31T13:58:00", pnl: -312 },
    ],
    flagged_trades: [
      {
        trade_id: "T-19321",
        timestamp: "10:41",
        symbol: "TSLA",
        bias: "revenge_trading",
        confidence: 0.86,
        evidence: ["Re-entry in under 90 seconds", "Size above rolling median"],
      },
      {
        trade_id: "T-19322",
        timestamp: "11:03",
        symbol: "AAPL",
        bias: "overtrading",
        confidence: 0.72,
        evidence: ["Trade burst in 10-minute window", "Low hold-time percentile"],
      },
      {
        trade_id: "T-19323",
        timestamp: "13:58",
        symbol: "NVDA",
        bias: "loss_aversion",
        confidence: 0.79,
        evidence: ["Loss held far above personal median", "Discipline drift in stop behavior"],
      },
    ],
    explainability: {
      overtrading: [
        { feature: "trades_10m", contribution: 0.31, direction: "positive", detail: "Trade frequency above baseline" },
        { feature: "median_hold_min", contribution: -0.12, direction: "negative", detail: "Hold time compressed" },
      ],
      loss_aversion: [
        { feature: "loss_hold_percentile", contribution: 0.37, direction: "positive", detail: "Losses held longer" },
        { feature: "stop_override_rate", contribution: 0.19, direction: "positive", detail: "Stops overridden" },
      ],
      revenge_trading: [
        {
          feature: "reentry_seconds_after_loss",
          contribution: 0.42,
          direction: "positive",
          detail: "Rapid re-entry after negative outcome",
        },
        { feature: "post_loss_size_ratio", contribution: 0.22, direction: "positive", detail: "Size increases post-loss" },
      ],
      recency_bias: [
        { feature: "short_term_trades", contribution: 0.35, direction: "positive", detail: "Recent trades cluster tightly" },
        { feature: "recent_pnl_influence", contribution: 0.28, direction: "positive", detail: "Recent wins affect position sizing" },
      ],
    },
  };
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
  valueTone,
  size = "compact",
}: {
  title: string;
  value: string;
  bars: SnapshotBarDatum[];
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
              const toneClass =
                bar.tone === "neutral"
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
                  overlayClassName="barChartTooltip"
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
    </div>
  );
}

type SnapshotChartCardModel = {
  id: string;
  title: string;
  value: string;
  valueTone?: "pos" | "neg";
  bars: SnapshotBarDatum[];
};

type SnapshotChartBundle = {
  cards: SnapshotChartCardModel[];
};

type SnapshotAnalysisLike = {
  trade_timeline?: AnalysisOutput["trade_timeline"];
  flagged_trades?: AnalysisOutput["flagged_trades"];
  summary?: AnalysisOutput["summary"];
};

function buildSnapshotChartBundle(
  analysis: SnapshotAnalysisLike | null | undefined,
  granularity: ChartGranularity
): SnapshotChartBundle {
  const snapshotBuckets = buildSnapshotBuckets(analysis?.trade_timeline, analysis?.flagged_trades, granularity);
  const summaryTotalTrades = analysis?.summary?.total_trades;
  const summaryTotalPnl = analysis?.summary?.total_pnl;
  const totalTradesFromBuckets = snapshotBuckets.reduce((sum, bucket) => sum + bucket.trades, 0);
  const totalPnlFromBuckets = snapshotBuckets.reduce((sum, bucket) => sum + bucket.pnl, 0);
  const totalConfidenceSum = snapshotBuckets.reduce((sum, bucket) => sum + bucket.confidenceSum, 0);
  const totalFlaggedTrades = snapshotBuckets.reduce((sum, bucket) => sum + bucket.flaggedTrades, 0);
  const totalWins = snapshotBuckets.reduce((sum, bucket) => sum + bucket.wins, 0);
  const totalBiasIncidents = snapshotBuckets.reduce((sum, bucket) => sum + bucket.biasIncidents, 0);
  const avgConfidence = totalFlaggedTrades > 0 ? totalConfidenceSum / totalFlaggedTrades : 0;
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

  const tradesConfidenceBars: SnapshotBarDatum[] = snapshotBuckets.map((bucket) => ({
    key: `confidence-${bucket.key}`,
    label: bucket.label,
    value: bucket.weightedConfidenceTrades,
    tone: "positive",
    tooltip: (
      <SnapshotTooltipContent
        title={bucket.key}
        rows={[
          { label: "Weighted score", value: bucket.weightedConfidenceTrades.toFixed(2) },
          { label: "Avg confidence", value: `${bucket.avgConfidence.toFixed(1)}%` },
          { label: "Flagged trades", value: bucket.flaggedTrades.toLocaleString() },
          { label: "Total trades", value: bucket.trades.toLocaleString() },
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
      tone: bucket.biasIncidents > 0 ? "warning" : "neutral",
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

  return {
    cards: [
      {
        id: "trade_frequency",
        title: "Trade Frequency",
        value: `${totalTrades.toLocaleString()} total`,
        bars: tradeFrequencyBars,
      },
      {
        id: "confidence",
        title: "Trades + Confidence",
        value: `${totalTrades.toLocaleString()} | ${Math.round(avgConfidence)}%`,
        bars: tradesConfidenceBars,
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
      },
    ],
  };
}

const COMMAND_SNAPSHOT_CARD_IDS = [
  "trade_frequency",
  "confidence",
  "realized_pnl",
  "win_rate",
  "bias_incidents",
] as const;
const COMMAND_SNAPSHOT_MAX_BUCKETS = 60;

function buildCommandSnapshotCards(
  analysis: SnapshotAnalysisLike | null | undefined,
  granularity: ChartGranularity
): SnapshotChartCardModel[] {
  const cards = buildSnapshotChartBundle(analysis, granularity).cards;
  const cardsById = new Map(cards.map((card) => [card.id, card] as const));

  return COMMAND_SNAPSHOT_CARD_IDS.map((id) => {
    const card = cardsById.get(id);
    if (!card) return null;
    return {
      ...card,
      bars: card.bars.slice(-COMMAND_SNAPSHOT_MAX_BUCKETS),
    };
  }).filter((card): card is SnapshotChartCardModel => card !== null);
}

function SnapshotChartsCarousel({
  analysis,
  granularity,
  onGranularityChange,
  size = "compact",
}: {
  analysis: SnapshotAnalysisLike | null | undefined;
  granularity: ChartGranularity;
  onGranularityChange: (next: ChartGranularity) => void;
  size?: "compact" | "large";
}) {
  const bundle = buildSnapshotChartBundle(analysis, granularity);
  const cards = bundle.cards;
  const [activeIndex, setActiveIndex] = useState(0);
  const carouselRef = useRef<any>(null);
  const isLarge = size === "large";

  useEffect(() => {
    setActiveIndex(0);
    if (carouselRef.current) carouselRef.current.goTo(0, true);
  }, [granularity, cards.length]);

  return (
    <div className={cn("snapshotCarousel", isLarge && "snapshotCarousel--large")}>
      <div className="snapshotControls">
        <AntSegmented
          className="chartGranularityTabs"
          value={granularity}
          onChange={(value) => onGranularityChange(value as ChartGranularity)}
          options={CHART_GRANULARITY_OPTIONS}
        />
        <div className="snapshotCarousel__controls">
          <AntButton
            className="snapshotCarousel__arrow"
            type="text"
            icon={<LeftOutlined />}
            onClick={() => carouselRef.current?.prev()}
            disabled={cards.length <= 1}
            aria-label="Previous chart"
          />
          <span className="snapshotCarousel__counter mono">
            {cards.length > 0 ? activeIndex + 1 : 0}/{cards.length}
          </span>
          <AntButton
            className="snapshotCarousel__arrow"
            type="text"
            icon={<RightOutlined />}
            onClick={() => carouselRef.current?.next()}
            disabled={cards.length <= 1}
            aria-label="Next chart"
          />
        </div>
      </div>

      <AntCarousel ref={carouselRef} dots={{ className: "snapshotCarousel__dots" }} afterChange={setActiveIndex}>
        {cards.map((card) => (
          <div key={card.id} className="snapshotCarousel__slide">
            <BarChartCard
              title={card.title}
              value={card.value}
              valueTone={card.valueTone}
              bars={card.bars}
              size={size}
            />
          </div>
        ))}
      </AntCarousel>
    </div>
  );
}

function DangerHeatmap({
  dangerHours,
  dailyPnl,
  tradeTimeline,
  biasScores,
  topTriggers,
  flaggedTrades,
  explainability,
}: {
  dangerHours?: number[][];
  dailyPnl?: AnalysisOutput["daily_pnl"];
  tradeTimeline?: AnalysisOutput["trade_timeline"];
  biasScores?: AnalysisOutput["bias_scores"];
  topTriggers?: string[];
  flaggedTrades?: AnalysisOutput["flagged_trades"];
  explainability?: AnalysisOutput["explainability"];
}) {
  const [mode, setMode] = useState<HeatmapMode>("bias");
  const hasBiasHeatmapData =
    Array.isArray(dangerHours) &&
    dangerHours.length > 0 &&
    dangerHours.some((row) => Array.isArray(row) && row.some((value) => Number.isFinite(value)));
  const normalizedDailyPnl = buildDailyPnlSeries(dailyPnl, tradeTimeline, flaggedTrades);
  const hasPnlHeatmapData = normalizedDailyPnl.length > 0;
  const hasAnyHeatmapData = hasBiasHeatmapData || hasPnlHeatmapData;

  const latestPnlDate = (() => {
    if (!hasPnlHeatmapData) return null;
    const parsedDates = normalizedDailyPnl
      .map((point) => new Date(`${point.date}T00:00:00`))
      .filter((date) => !Number.isNaN(date.getTime()))
      .sort((a, b) => a.getTime() - b.getTime());
    return parsedDates.length ? parsedDates[parsedDates.length - 1] : null;
  })();
  const referenceDate = latestPnlDate ?? new Date();
  const monthYearLabel = new Intl.DateTimeFormat("en-US", { month: "long", year: "numeric" }).format(referenceDate);
  const pnlCalendar = buildMonthlyPnlCalendar(referenceDate, normalizedDailyPnl);
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
          <AntTooltip title={tooltipTitle} overlayClassName="dangerCellTooltip" placement="top">
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

  const heatmapTitle =
    mode === "bias" ? "Behavior Risk Heatmap (Weekday x Hour)" : "Realized PnL Calendar (Day x Week)";
  const heatmapAxisHint =
    mode === "bias"
      ? "Y-axis: weekday (Mon-Sun) | X-axis: market hour bucket (09-17, local time)"
      : "Y-axis: week of month | X-axis: weekday (Mon-Sun)";

  return (
    <div>
      <div className="heatmapHdr">
        <div>
          <div className="heatmapHdr__title">{heatmapTitle}</div>
          <div className="heatmapHdr__axis muted">{heatmapAxisHint}</div>
          <div className="muted">
            {hasPnlHeatmapData ? `Month: ${monthYearLabel}` : "Import a CSV to generate heatmap data."}
          </div>
        </div>
        <AntSegmented
          className="heatmapModeTabs"
          value={mode}
          onChange={(next) => setMode(next as HeatmapMode)}
          disabled={!hasAnyHeatmapData}
          options={[
            { label: "Risk", value: "bias" },
            { label: "PnL", value: "pnl" },
          ]}
        />
      </div>

      {!hasAnyHeatmapData ? (
        <div className="heatmapEmpty">
          <div className="heatmapEmpty__title">No Heatmap Data Yet</div>
          <div className="muted">Upload a trade CSV, then run analysis to populate danger hours and PnL maps.</div>
        </div>
      ) : mode === "bias" ? (
        !hasBiasHeatmapData ? (
          <div className="heatmapEmpty heatmapEmpty--compact">
            <div className="heatmapEmpty__title">Danger Hours Unavailable</div>
            <div className="muted">This analysis did not return weekday-hour danger data.</div>
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
        )
      ) : (
        !hasPnlHeatmapData ? (
          <div className="heatmapEmpty heatmapEmpty--compact">
            <div className="heatmapEmpty__title">PnL Calendar Unavailable</div>
            <div className="muted">This analysis did not return daily PnL points.</div>
          </div>
        ) : (
        <div className="pnlHeatmapWrap">
          <div className="pnlHeatmapWeekdays">
            {DANGER_DAYS.map((day) => (
              <span key={day} className="pnlHeatmapWeekdays__day mono">
                {day}
              </span>
            ))}
          </div>

          <div className="pnlHeatmapGrid">
            {pnlCalendar.map((entry, index) => {
              if (!entry) return <div key={`empty-${index}`} className="pnlCell pnlCell--empty" aria-hidden="true" />;

              const dayLabel = new Intl.DateTimeFormat("en-US", {
                weekday: "short",
                month: "short",
                day: "numeric",
                year: "numeric",
              }).format(entry.date);

              return (
                <AntTooltip
                  key={entry.key}
                  overlayClassName="pnlCellTooltip"
                  title={
                    <div className="pnlCellTooltip__content">
                      <span className="muted">{dayLabel}:</span>{" "}
                      <span className={cn("mono", "pnlCellTooltip__value", entry.pnl >= 0 ? "pnlCellTooltip__value--pos" : "pnlCellTooltip__value--neg")}>
                        {formatPnlCurrency(entry.pnl)}
                      </span>
                    </div>
                  }
                >
                  <div className={cn("pnlCell", resolvePnlToneClass(entry.pnl))}>
                    <span className="pnlCell__day mono">{entry.dayOfMonth}</span>
                  </div>
                </AntTooltip>
              );
            })}
          </div>

          <div className="pnlHeatmapLegend muted">
            <span className="legend__dot pnlLegend__dot--neg" />
            Loss days
            <span className="legend__dot pnlLegend__dot--flat" />
            Flat
            <span className="legend__dot pnlLegend__dot--pos" />
            Profit days
          </div>
        </div>
        )
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

  const radius = 15.915;
  const circumference = 2 * Math.PI * radius;
  const stroke = `${(animatedRiskScore / 100) * circumference} ${circumference}`;
  const commandSnapshotCards = buildCommandSnapshotCards(analysis, chartGranularity);

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
            {commandSnapshotCards.map((card) => (
              <BarChartCard
                key={card.id}
                title={card.title}
                value={card.value}
                valueTone={card.valueTone}
                bars={card.bars}
                size="compact"
              />
            ))}
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
            <Card title="Danger Heatmaps" className="dangerFocusCard">
              <DangerHeatmap
                dangerHours={analysis?.danger_hours}
                dailyPnl={analysis?.daily_pnl}
                tradeTimeline={analysis?.trade_timeline}
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
                    {(analysis?.summary?.total_pnl || 0) >= 0 ? '+' : ''}{analysis?.summary?.total_pnl || 0}
                  </span>
                </div>
                <div className="miniRow">
                  <span>Drawdown</span>
                  <span className="mono neg">
                    -{analysis?.summary?.max_drawdown || 0}
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
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const columns = [
    {
      title: "Time",
      dataIndex: "time",
      key: "time",
      render: (value: string) => <span className="mono">{value}</span>,
    },
    {
      title: "Symbol",
      dataIndex: "symbol",
      key: "symbol",
      render: (value: string) => <span className="mono">{value}</span>,
    },
    { title: "Side", dataIndex: "side", key: "side" },
    {
      title: "Size",
      dataIndex: "size",
      key: "size",
      render: (value: number) => <span className="mono">{value.toFixed(2)}</span>,
    },
    {
      title: "Dur",
      dataIndex: "durationMin",
      key: "durationMin",
      render: (value: number) => <span className="mono">{value}m</span>,
    },
    {
      title: "PnL",
      dataIndex: "pnl",
      key: "pnl",
      render: (value: number) => <span className={cn("mono", value >= 0 ? "pos" : "neg")}>{value >= 0 ? `+${value}` : value}</span>,
    },
    {
      title: "Prominent Flag",
      dataIndex: "flags",
      key: "flags",
      render: (flags: BiasType[]) => (
        <div className="chipsRow">
          {flags.length ? (
            flags.map((flag) => (
              <Pill key={flag} tone="warn">
                {flag}
              </Pill>
            ))
          ) : (
            <Pill tone="ok">Clean</Pill>
          )}
        </div>
      ),
    },
    {
      title: "Bias Conf",
      dataIndex: "confidence",
      key: "confidence",
      render: (value: number) => <span className="mono">{value ? `${value}%` : "-"}</span>,
    },
  ];

  return (
    <div className="canvas">
      <div className="grid">
        <Card title="Flagged Trades" className="span-7">
          <div className="tableWrap">
            <AntTable<Trade>
              className="table"
              size="small"
              columns={columns}
              dataSource={trades}
              rowKey="id"
              pagination={false}
              rowClassName={(record) => cn("tr", selected?.id === record.id && "tr--active")}
              onRow={(record) => {
                const isHovered = hoveredId === record.id;
                const isSelected = selected?.id === record.id;
                let backgroundColor = "transparent";
                let boxShadow = "none";
                
                if (isSelected && isHovered) {
                  backgroundColor = "rgba(47,232,199,.65)";
                  boxShadow = "inset 0 0 50px rgba(47,232,199,.8), 0 0 150px rgba(47,232,199,.6), 0 0 300px rgba(47,232,199,.4), 0 0 450px rgba(47,232,199,.2)";
                } else if (isSelected) {
                  backgroundColor = "rgba(47,232,199,.5)";
                  boxShadow = "inset 0 0 40px rgba(47,232,199,.7), 0 0 120px rgba(47,232,199,.5), 0 0 240px rgba(47,232,199,.3), 0 0 360px rgba(47,232,199,.15)";
                } else if (isHovered) {
                  backgroundColor = "rgba(47,232,199,.35)";
                  boxShadow = "0 0 90px rgba(47,232,199,.4), 0 0 180px rgba(47,232,199,.25), 0 0 300px rgba(47,232,199,.12)";
                }
                
                return {
                  onClick: () => onSelect(record),
                  onMouseEnter: () => setHoveredId(record.id),
                  onMouseLeave: () => setHoveredId(null),
                  style: { 
                    backgroundColor,
                    boxShadow,
                    cursor: "pointer",
                  },
                };
              }}
            />
          </div>
        </Card>

        <Card title="Trade Inspector" className="span-5">
          {selected && flaggedTradesRaw ? (
            (() => {
              const flaggedTrade = flaggedTradesRaw.find((ft) => ft.trade_id === selected.id);
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
                                ? flaggedTrade.evidence.join(" • ")
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

function HeatmapWorkspace({
  dangerHours,
  dailyPnl,
  tradeTimeline,
  biasScores,
  topTriggers,
  flaggedTrades,
  explainability,
}: {
  dangerHours?: number[][];
  dailyPnl?: AnalysisOutput["daily_pnl"];
  tradeTimeline?: AnalysisOutput["trade_timeline"];
  biasScores?: AnalysisOutput["bias_scores"];
  topTriggers?: string[];
  flaggedTrades?: AnalysisOutput["flagged_trades"];
  explainability?: AnalysisOutput["explainability"];
}) {
  const [chartGranularity, setChartGranularity] = useState<ChartGranularity>("15m");

  return (
    <div className="canvas">
      <div className="grid dangerFocusGrid">
        <Card title="Danger Heatmaps" className="span-8 tall dangerFocusCard">
          <DangerHeatmap
            dangerHours={dangerHours}
            dailyPnl={dailyPnl}
            tradeTimeline={tradeTimeline}
            biasScores={biasScores}
            topTriggers={topTriggers}
            flaggedTrades={flaggedTrades}
            explainability={explainability}
          />
        </Card>

        <Card title="Session Charts" className="span-4 tall dangerCarouselCard dangerFocusCard">
          <SnapshotChartsCarousel
            analysis={{ trade_timeline: tradeTimeline, flagged_trades: flaggedTrades }}
            granularity={chartGranularity}
            onGranularityChange={(next) => setChartGranularity(next)}
            size="large"
          />
        </Card>
      </div>
    </div>
  );
}

function SimulatorWorkspace({
  analysisData,
  importedCsvData,
}: {
  analysisData: AnalysisOutput | null;
  importedCsvData: any[];
}) {
  const [cooldownMinutes, setCooldownMinutes] = useState(15);
  const [dailyTradeCap, setDailyTradeCap] = useState(10);
  const [positionSizeMultiplier, setPositionSizeMultiplier] = useState(1.5);
  const [maxConsecutiveLosses, setMaxConsecutiveLosses] = useState(3);
  const [simulationResult, setSimulationResult] = useState<any>(null);
  const [isRunning, setIsRunning] = useState(false);

  // Calculate max values based on imported data
  const totalTrades = importedCsvData?.length || 0;
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
    if (!importedCsvData || importedCsvData.length === 0) {
      antMessage.warning("No trade data available. Please import a CSV first.");
      return;
    }

    setIsRunning(true);
    try {
      // Convert imported CSV to trade format
      const trades = importedCsvData.map((row: any) => ({
        timestamp: new Date(row.timestamp || row.date).toISOString(),
        asset: row.symbol || row.asset,
        side: row.side.toUpperCase() === "BUY" ? "BUY" : "SELL",
        quantity: parseFloat(row.size || row.quantity),
        entry_price: parseFloat(row.entry_price),
        exit_price: parseFloat(row.exit_price),
        profit_loss: parseFloat(row.pnl || row.profit_loss),
        balance: parseFloat(row.balance) || 0,
      }));

      const result = await simulateDataset({
        trades,
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
            <div style={{ maxHeight: "300px", overflow: "auto" }}>
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
          <Card title="Practice Complete! 🎉" className="span-full">
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
                  <div style={{ fontSize: "16px", fontWeight: "bold", marginBottom: "15px", color: "#51cf66" }}>🎯 Your Next Habit</div>
                  <div style={{ fontSize: "15px", lineHeight: "1.8", color: "#ddd" }}>
                    {habitSuggestion}
                  </div>
                </div>
              </div>
              <div style={{ fontSize: "16px", fontWeight: "bold", marginBottom: "15px", color: "#51cf66" }}>🎯 Your Next Habit</div>
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
                    <span style={{ marginLeft: "auto", color: "#51cf66" }}>✓ Correct</span>
                  )}
                  {isCurrentAnswered && idx === userSelectedIndex && !isCorrect && (
                    <span style={{ marginLeft: "auto", color: "#ff6b6b" }}>✗ Not quite</span>
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
                {isCorrect ? "Excellent! 🎯" : "Let's learn from this 📚"}
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

function PulseWorkspace({ datasetId }: { datasetId: string | null }) {
  const [explainerData, setExplainerData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [personalReasons, setPersonalReasons] = useState<{ [key: string]: string }>({});

  const loadContextExplainer = async () => {
    if (!datasetId) {
      antMessage.warning("Please import a dataset first.");
      return;
    }

    setLoading(true);
    try {
      // First, get the analysis results to get date range and top biases
      const analysis = await analyzeDataset({
        dataset_id: datasetId,
        scoring_mode: "hybrid",
      });

      // Get top 2 biases from bias_scores object
      const biasEntries = Object.entries(analysis.bias_scores || {})
        .map(([biasType, score]: [string, any]) => ({
          bias_type: biasType,
          score: typeof score === 'number' ? score : 0,
        }))
        .sort((a, b) => b.score - a.score)
        .slice(0, 2);

      if (biasEntries.length === 0) {
        antMessage.warning("No biases detected in this dataset.");
        setLoading(false);
        return;
      }

      // Call context explainer endpoint using proper API function
      const data = await getBiasContextExplainer({
        dataset_id: datasetId,
        top_two_biases: biasEntries,
      });

      setExplainerData(data);

      // Initialize personal reasons
      const reasons: { [key: string]: string } = {};
      biasEntries.forEach((bias: any) => {
        reasons[bias.bias_type] = "";
      });
      setPersonalReasons(reasons);
    } catch (error) {
      console.error("Error loading context explainer:", error);
      antMessage.error("Failed to load context explanation. Check console.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="canvas">
      <div className="grid">
        {/* Header with dataset selector and load button */}
        <div className="span-12" style={{ display: "flex", gap: "12px", alignItems: "center", padding: "12px", backgroundColor: "var(--panel)", borderRadius: "var(--radius)", border: "1px solid var(--stroke)" }}>
          <div style={{ flex: 1, color: "var(--muted)" }}>
            {datasetId ? `Dataset: ${datasetId}` : "No dataset imported. Please import CSV first."}
          </div>
          <Button
            variant="primary"
            onClick={loadContextExplainer}
            disabled={loading || !datasetId}
          >
            {loading ? "Loading..." : "Load Context Explainer"}
          </Button>
        </div>

        {/* Context Explainer Output */}
        {explainerData && (
          <>
            {/* Bias Explanations */}
            {explainerData.bias_contexts?.map((context: any, idx: number) => (
              <div key={idx} className="span-6 tall" style={{ padding: "12px", backgroundColor: "var(--panel)", borderRadius: "var(--radius)", border: "1px solid var(--stroke)" }}>
                <div className="section">
                  <div className="section__title" style={{ fontSize: "16px", fontWeight: "600" }}>
                    {context.bias_name}
                  </div>
                  <div className="muted" style={{ fontSize: "12px", marginBottom: "12px" }}>
                    Why this bias spiked during your trading period
                  </div>

                  {/* Market Events */}
                  <div style={{ marginBottom: "16px" }}>
                    <div style={{ fontSize: "12px", fontWeight: "600", marginBottom: "8px" }}>
                      📰 Relevant Market Events
                    </div>
                    {context.market_events?.map((event: any, i: number) => (
                      <div
                        key={i}
                        style={{
                          fontSize: "12px",
                          marginBottom: "8px",
                          paddingLeft: "8px",
                          borderLeft: "2px solid var(--stroke)",
                        }}
                      >
                        <div style={{ fontWeight: "500", color: "var(--muted)" }}>{event.date}</div>
                        <div>{event.headline}</div>
                      </div>
                    ))}
                  </div>

                  {/* Connection to Bias */}
                  <div style={{ marginBottom: "16px" }}>
                    <div style={{ fontSize: "12px", fontWeight: "600", marginBottom: "8px" }}>
                      🔗 How This Connected to Your Trading
                    </div>
                    <div style={{ fontSize: "13px", lineHeight: "1.5" }}>
                      {context.connection_explanation}
                    </div>
                  </div>

                  {/* Practical Takeaway */}
                  <div style={{ marginBottom: "16px" }}>
                    <div style={{ fontSize: "12px", fontWeight: "600", marginBottom: "8px" }}>
                      ✓ Practical Takeaway for Next Time
                    </div>
                    <div
                      style={{
                        fontSize: "13px",
                        lineHeight: "1.5",
                        padding: "8px",
                        backgroundColor: "var(--surfaceA)",
                        borderRadius: "4px",
                      }}
                    >
                      {context.practical_takeaway}
                    </div>
                  </div>

                  {/* Personal Reasons Section */}
                  <div>
                    <div style={{ fontSize: "12px", fontWeight: "600", marginBottom: "8px" }}>
                      💭 Personal Reasons (Fill In Your Own)
                    </div>
                    <div style={{ fontSize: "12px", color: "var(--muted2)", marginBottom: "6px" }}>
                      What personal factors may have contributed? (e.g., stress, fatigue, FOMO,
                      boredom, time pressure)
                    </div>
                    <AntTextArea
                      rows={3}
                      value={personalReasons[context.bias_name] || ""}
                      onChange={(e) =>
                        setPersonalReasons({
                          ...personalReasons,
                          [context.bias_name]: e.target.value,
                        })
                      }
                      placeholder="Write your personal factors here..."
                    />
                  </div>
                </div>
              </div>
            ))}

            {/* Evidence Summary */}
            {explainerData.date_range && (
              <div className="span-12" style={{ padding: "12px", backgroundColor: "var(--panel)", borderRadius: "var(--radius)", border: "1px solid var(--stroke)" }}>
                <div>
                  <div className="section__title" style={{ fontSize: "14px", marginBottom: "8px" }}>
                    📊 Analysis Period & Methodology
                  </div>
                  <div style={{ fontSize: "12px", color: "var(--muted)", lineHeight: "1.6" }}>
                    <div>
                      <strong>Trading Period:</strong> {explainerData.date_range.start} to{" "}
                      {explainerData.date_range.end}
                    </div>
                    <div>
                      <strong>Approach:</strong> This analysis uses publicly available market events
                      and economic data to explain potential factors behind detected bias patterns.
                      These are correlations and contextual connections, not causation. Your personal
                      factors, decisions, and emotions remain the primary drivers of your trading.
                    </div>
                  </div>
                </div>
              </div>
            )}
          </>
        )}

        {/* Empty State */}
        {!explainerData && !loading && (
          <div className="span-12" style={{ textAlign: "center", padding: "40px", backgroundColor: "var(--panel)", borderRadius: "var(--radius)", border: "1px solid var(--stroke)" }}>
            <div style={{ color: "var(--muted2)" }}>
              Select a dataset and click "Load Context Explainer" to see market events and
              explanations connected to your detected trading biases.
            </div>
          </div>
        )}
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
            <strong>⚠️ Educational Disclaimer:</strong> This coach provides educational information about trading concepts,
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
/* --------------------------- App Shell --------------------------- */

export default function App() {
  const [workspace, setWorkspace] = useState<Workspace>("command");
  const [selectedTrade, setSelectedTrade] = useState<Trade | null>(SAMPLE_TRADES[0]);
  const [scoringMode, setScoringMode] = useState<ScoringMode>("hybrid");
  const [datasetId, setDatasetId] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisOutput | null>(null);
  const [apiBusy, setApiBusy] = useState(false);
  const [apiError, setApiError] = useState<string>("");

  const [csvModalOpen, setCsvModalOpen] = useState(false);
  const [importedCsvData, setImportedCsvData] = useState<any[]>([]);

  useEffect(() => {
    if (!datasetId) return;
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
    const base = workspace === "replay" ? 68 : workspace === "heatmap" ? 62 : 75;
    return base;
  })();

  const handleImported = async (payload: CsvImportPayload) => {
    setApiBusy(true);
    setApiError("");
    setImportedCsvData(payload.rows);

    const importRequest: ImportDatasetRequest = {
      raw_csv: payload.rawText,
      rows: payload.rows,
      mapping: payload.mapping,
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC",
      session_template: "equities_rth",
    };

    try {
      const imported = await importDataset(importRequest);
      setDatasetId(imported.dataset_id);
      const analyzed = await analyzeDataset({ dataset_id: imported.dataset_id, scoring_mode: scoringMode });
      setAnalysis(analyzed);
      antMessage.success(
        `Imported ${payload.rows.length.toLocaleString()} rows. Behavior Index ${Math.round(analyzed.behavior_index)}.`
      );
    } catch (error: unknown) {
      const fallback = buildLocalFallbackAnalysis();
      setAnalysis(fallback);
      setDatasetId("local-preview");
      setApiError(getErrorMessage(error, "Backend unavailable. Running local preview mode."));
      antMessage.warning(
        `Imported ${payload.rows.length.toLocaleString()} rows in local preview mode. Start backend API to enable /import and /analyze.`
      );
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
              trades={analysis?.flagged_trades?.map((ft) => {
                const biasMap: Record<string, BiasType> = {
                  "overtrading": "Overtrading",
                  "loss_aversion": "Loss Aversion",
                  "revenge_trading": "Revenge Trading",
                  "recency_bias": "Recency Bias",
                };
                return {
                  id: ft.trade_id || "",
                  time: ft.timestamp ? new Date(ft.timestamp).toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" }) : "",
                  symbol: ft.symbol || "",
                  side: (ft.side || "Buy") as "Buy" | "Sell",
                  size: ft.quantity || 0,
                  durationMin: 0,
                  pnl: ft.profit_loss || 0,
                  flags: [biasMap[ft.bias] || ft.bias] as BiasType[],
                  confidence: ft.confidence || 0,
                  evidence: ft.evidence?.join(" | ") || "",
                };
              }) || []}
              selected={selectedTrade}
              onSelect={(t) => setSelectedTrade(t)}
              flaggedTradesRaw={analysis?.flagged_trades}
              biasScores={analysis?.bias_scores}
            />
          )}
          {workspace === "heatmap" && (
            <HeatmapWorkspace
              dangerHours={analysis?.danger_hours}
              dailyPnl={analysis?.daily_pnl}
              tradeTimeline={analysis?.trade_timeline}
              biasScores={analysis?.bias_scores}
              topTriggers={analysis?.top_triggers}
              flaggedTrades={analysis?.flagged_trades}
              explainability={analysis?.explainability}
            />
          )}
          {workspace === "simulator" && <SimulatorWorkspace analysisData={analysis} importedCsvData={importedCsvData} />}
          {workspace === "replay" && (
            <PracticeWorkspace 
              datasetId={datasetId}
              scoringMode={scoringMode}
              analysis={analysis}
            />
          )}
          {workspace === "pulse" && <PulseWorkspace datasetId={datasetId} />}
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
    </div>
  );
}



