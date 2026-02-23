# Insid-ur Trading

**Insid-ur Trading** is a full-stack trading psychology analytics platform that helps traders identify and overcome behavioral biases in their trading history. Upload a CSV of your trades and get a deep-dive report on overtrading, loss aversion, revenge trading, and recency bias — powered by rule-based detectors, an XGBoost classifier, and an OpenAI-backed coaching system.

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [CSV Format](#csv-format)
- [Bias Detectors](#bias-detectors)
- [AI Features](#ai-features)
- [Sample Datasets](#sample-datasets)

---

## Features

| Feature | Description |
|---|---|
| **Bias Detection** | Four rule-based detectors score traders 0–100 on overtrading, loss aversion, revenge trading, and recency bias |
| **ML Scoring** | XGBoost classifier layer cross-validates and reinforces rule-based scores |
| **Flagged Trade View** | Individual trades highlighted with per-trade evidence explaining why they were flagged |
| **Equity Curve** | Visual account balance progression over time |
| **Danger Hours Heatmap** | 7×24 matrix revealing which days/hours produce the worst trading outcomes |
| **What-If Simulator** | Test hypothetical rule changes (e.g. "no trades within 10 min of a loss") and measure PnL impact |
| **Market Shock Events** | Detects extreme PnL outliers and correlates them with synthetic market shock events |
| **AI Coach Chat** | Multi-turn LangChain/OpenAI chatbot providing personalized coaching based on your trade history |
| **Emotional Check-In** | Pre-session bias awareness quiz that feeds into personalized AI explainers |
| **Practice Mode** | AI-generated scenario questions targeting your two worst biases, with habit-building suggestions |
| **Pulse Workspace** | Real-time risk monitoring strip showing live bias risk level |

---

## Tech Stack

### Backend
- **Python 3.11+** — FastAPI, Uvicorn
- **Pydantic v2** — data validation and schemas
- **XGBoost + scikit-learn** — ML classifier for hybrid bias scoring
- **LangChain + OpenAI** — AI coach chat and practice question generation
- **NumPy / Pandas** — statistical analysis

### Frontend
- **React 19 + TypeScript** — component-based UI
- **Vite** — build tooling and dev server
- **Ant Design** — UI component library
- **TanStack Virtual** — virtualized trade list rendering

---

## CSV Format

Upload a CSV with the following columns (case-insensitive):

```csv
timestamp,asset,side,quantity,entry_price,exit_price,profit_loss,balance
2024-01-15 09:30:00,AAPL,BUY,100,150.00,151.50,150.00,25150.00
2024-01-15 09:45:00,MSFT,SELL,50,380.00,379.00,50.00,25200.00
```

| Column | Description |
|---|---|
| `timestamp` | ISO8601 or `YYYY-MM-DD HH:MM:SS` |
| `asset` | Ticker symbol (AAPL, MSFT, etc.) |
| `side` | `BUY`/`SELL` or `LONG`/`SHORT` |
| `quantity` | Number of shares |
| `entry_price` | Entry price per share |
| `exit_price` | Exit price per share |
| `profit_loss` | Realized PnL for the trade |
| `balance` | Account balance after the trade |

---

## Bias Detectors

Each detector returns a **0–100 score**, a severity (`low` / `med` / `high`), a list of flagged trade IDs, and per-trade evidence strings.

### Overtrading
Detects rapid direction switching, reactive re-entry after outlier PnL events, and burst-hour density using behavioral rate signals — not raw trade counts — so scores are consistent across dataset sizes.

### Loss Aversion / Disposition Bias
Identifies an asymmetry between how long winning vs. losing positions are held, and flags trades where losses are cut too small relative to wins.

### Revenge Trading
Flags rapid re-entry trades placed within minutes of a significant loss, characteristic of emotionally-driven position recovery attempts.

### Recency Bias
Detects changes in position sizing, frequency, or direction following win/loss streaks of 3+ consecutive trades, indicating recent outcomes are over-influencing decisions.

---

## AI Features

> **Requires** `OPENAI_API_KEY` set in `backend/.env`

### AI Coach Chat
Multi-turn conversational coaching powered by LangChain + GPT. The coach is aware of your trade history and bias scores and can answer questions, suggest rule improvements, and explain flagged trades in plain language.

### Emotional Check-In
A short bias-specific questionnaire completed before viewing results. Answers are fed into the AI to generate a personalized context explainer linking your emotional state to your worst-scoring biases.

### Practice Mode
The AI generates four scenario-based multiple-choice questions targeting your top two biases. After the quiz, a habit-building suggestion is served based on your weakest area.

### Bias Context Explainer
Connects flagged trades to generated market shock events and provides a practical takeaway for each detected bias.

---

## Sample Datasets

Four pre-built CSV profiles were used for training(provided by sponsor) but extra data generated from LLMs were also used for training:

| File | Profile |
|---|---|
| `calm_trader.csv` | Well-disciplined, low-bias baseline |
| `overtrader.csv` | High overtrading score profile |
| `loss_averse_trader.csv` | High loss aversion / disposition bias |
| `revenge_trader.csv` | High revenge trading score profile |

---

## License

Built for the National Bank of Canada Capital Markets Hackathon. All rights reserved.
