"""
Generate synthetic training datasets for the XGBoost bias classifier.

Creates four CSV files (one per bias class) in our standard NBH format:
    timestamp, asset, side, quantity, entry_price, exit_price, profit_loss, balance

Usage:
    python -m app.ml.generate_datasets
    python -m app.ml.generate_datasets --trades 3000

The generated CSVs are saved to app/ml/training_data/
"""
from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent / "training_data"

ASSETS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "SPY", "QQQ", "GOOGL"]
ASSET_PRICE = {
    "AAPL": 175.0, "MSFT": 390.0, "NVDA": 860.0,
    "TSLA": 245.0, "AMZN": 195.0, "SPY": 485.0,
    "QQQ": 410.0, "GOOGL": 175.0,
}


def _random_price_move(base_price: float, volatility: float = 0.005) -> float:
    """Return a price that has moved by ±volatility from base."""
    move = np.random.normal(0, base_price * volatility)
    return round(base_price + move, 2)


def _make_timestamp(
    start: datetime,
    trade_index: int,
    mean_gap_minutes: float = 20,
    std_gap_minutes: float = 10,
) -> datetime:
    """Return a market-hours timestamp with random gap from previous trade."""
    gap = max(1, int(np.random.normal(mean_gap_minutes, std_gap_minutes)))
    ts = start + timedelta(minutes=trade_index * gap)
    # Keep within approximate trading hours (9:30 – 16:00 ET)
    hour = ts.hour
    if hour < 9 or (hour == 9 and ts.minute < 30):
        ts = ts.replace(hour=9, minute=30)
    if hour >= 16:
        ts = ts.replace(hour=9, minute=30) + timedelta(days=1)
    return ts


# ===========================================================================
def generate_calm_trader(n_trades: int = 3000, seed: int = 42) -> pd.DataFrame:
    """
    Calm / disciplined trader:
    - Moderate frequency (~5-8 trades/day)
    - ~58% win rate
    - Consistent position sizing
    - Losses closed quickly, winners held somewhat longer
    """
    np.random.seed(seed)
    random.seed(seed)

    rows = []
    balance = 25_000.0
    ts = datetime(2023, 1, 3, 9, 30)

    for i in range(n_trades):
        asset = random.choice(ASSETS)
        base_price = ASSET_PRICE[asset]
        entry_price = _random_price_move(base_price, 0.005)
        side = random.choice(["BUY", "SELL"])
        qty = random.randint(10, 100)

        # ~58% win rate, moderate avg win/loss
        if np.random.random() < 0.58:
            pnl_pct = np.random.uniform(0.002, 0.012)  # +0.2% to +1.2%
        else:
            pnl_pct = -np.random.uniform(0.002, 0.010)  # -0.2% to -1.0%

        pnl = round(qty * entry_price * pnl_pct, 2)
        exit_price = round(entry_price + (pnl / qty), 2)
        balance = round(balance + pnl, 2)

        # Gap ~15-25 min between trades
        gap = int(np.random.normal(20, 8))
        gap = max(2, gap)
        ts += timedelta(minutes=gap)
        if ts.hour >= 16:
            ts = ts.replace(hour=9, minute=30) + timedelta(days=1)
        if ts.weekday() >= 5:
            ts += timedelta(days=(7 - ts.weekday()))

        rows.append({
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "asset": asset, "side": side,
            "quantity": qty, "entry_price": entry_price,
            "exit_price": exit_price,
            "profit_loss": pnl, "balance": balance,
        })

    return pd.DataFrame(rows)


# ===========================================================================
def generate_overtrader(n_trades: int = 3000, seed: int = 101) -> pd.DataFrame:
    """
    Overtrader:
    - High frequency (20-40+ trades/day)
    - Burst windows: 3-7 trades within minutes
    - Lower win rate (~48%) due to over-activity
    - Inconsistent sizing
    """
    np.random.seed(seed)
    random.seed(seed)

    rows = []
    balance = 25_000.0
    ts = datetime(2023, 1, 3, 9, 30)

    i = 0
    while i < n_trades:
        # Randomly trigger a burst (4-8 rapid trades)
        burst_size = np.random.randint(3, 8)
        gap_in_burst = np.random.randint(1, 4)  # 1-3 minutes between burst trades
        gap_after_burst = np.random.randint(5, 20)

        for b in range(burst_size):
            if i >= n_trades:
                break
            asset = random.choice(ASSETS)
            base_price = ASSET_PRICE[asset]
            entry_price = _random_price_move(base_price, 0.006)
            side = random.choice(["BUY", "SELL"])
            qty = random.randint(10, 150)

            if np.random.random() < 0.48:
                pnl_pct = np.random.uniform(0.001, 0.008)
            else:
                pnl_pct = -np.random.uniform(0.001, 0.009)

            pnl = round(qty * entry_price * pnl_pct, 2)
            exit_price = round(entry_price + (pnl / qty), 2)
            balance = round(balance + pnl, 2)

            ts += timedelta(minutes=gap_in_burst)
            if ts.hour >= 16:
                ts = ts.replace(hour=9, minute=30) + timedelta(days=1)
            if ts.weekday() >= 5:
                ts += timedelta(days=(7 - ts.weekday()))

            rows.append({
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "asset": asset, "side": side,
                "quantity": qty, "entry_price": entry_price,
                "exit_price": exit_price,
                "profit_loss": pnl, "balance": balance,
            })
            i += 1

        ts += timedelta(minutes=gap_after_burst)
        if ts.hour >= 16:
            ts = ts.replace(hour=9, minute=30) + timedelta(days=1)
        if ts.weekday() >= 5:
            ts += timedelta(days=(7 - ts.weekday()))

    return pd.DataFrame(rows)


# ===========================================================================
def generate_loss_averse_trader(n_trades: int = 3000, seed: int = 202) -> pd.DataFrame:
    """
    Loss averse trader:
    - Cuts winners quickly (small hold time)
    - Holds losers very long (large hold time = large inter-trade gap after loss)
    - Avg loss >> avg win (risk-reward inverted)
    - ~50% win rate but large losses swamp small wins
    """
    np.random.seed(seed)
    random.seed(seed)

    rows = []
    balance = 25_000.0
    ts = datetime(2023, 1, 3, 9, 30)
    prev_was_loss = False

    for i in range(n_trades):
        asset = random.choice(ASSETS)
        base_price = ASSET_PRICE[asset]
        entry_price = _random_price_move(base_price, 0.005)
        side = random.choice(["BUY", "SELL"])
        qty = random.randint(20, 100)

        is_win = np.random.random() < 0.50

        if is_win:
            # Cut winners short — small gain
            pnl_pct = np.random.uniform(0.001, 0.005)
            gap = int(np.random.normal(8, 3))  # closed quickly
        else:
            # Hold losers long — large loss
            pnl_pct = -np.random.uniform(0.010, 0.030)
            gap = int(np.random.normal(60, 20))  # held a long time

        gap = max(1, gap)
        pnl = round(qty * entry_price * pnl_pct, 2)
        exit_price = round(entry_price + (pnl / qty), 2)
        balance = round(balance + pnl, 2)

        ts += timedelta(minutes=gap)
        if ts.hour >= 16:
            ts = ts.replace(hour=9, minute=30) + timedelta(days=1)
        if ts.weekday() >= 5:
            ts += timedelta(days=(7 - ts.weekday()))

        rows.append({
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "asset": asset, "side": side,
            "quantity": qty, "entry_price": entry_price,
            "exit_price": exit_price,
            "profit_loss": pnl, "balance": balance,
        })
        prev_was_loss = not is_win

    return pd.DataFrame(rows)


# ===========================================================================
def generate_revenge_trader(n_trades: int = 3000, seed: int = 303) -> pd.DataFrame:
    """
    Revenge trader:
    - After a loss, re-enters within 2-10 minutes with 1.5-3× larger position
    - Loss streaks trigger escalating size
    - ~45% win rate overall
    """
    np.random.seed(seed)
    random.seed(seed)

    rows = []
    balance = 25_000.0
    ts = datetime(2023, 1, 3, 9, 30)
    consecutive_losses = 0
    base_qty = 50

    for i in range(n_trades):
        asset = random.choice(ASSETS)
        base_price = ASSET_PRICE[asset]
        entry_price = _random_price_move(base_price, 0.007)
        side = random.choice(["BUY", "SELL"])

        # Scale up size after losses
        size_multiplier = 1.0 + min(consecutive_losses * 0.5, 2.5)
        qty = max(10, int(base_qty * size_multiplier * np.random.uniform(0.8, 1.3)))

        is_win = np.random.random() < 0.45

        if is_win:
            pnl_pct = np.random.uniform(0.003, 0.012)
            gap = int(np.random.normal(25, 10))  # normal gap after win
            consecutive_losses = 0
        else:
            pnl_pct = -np.random.uniform(0.005, 0.015)
            # Quick re-entry after loss
            if consecutive_losses >= 1:
                gap = int(np.random.normal(4, 2))  # revenge re-entry
            else:
                gap = int(np.random.normal(15, 5))
            consecutive_losses += 1

        gap = max(1, gap)
        pnl = round(qty * entry_price * pnl_pct, 2)
        exit_price = round(entry_price + (pnl / qty), 2)
        balance = round(balance + pnl, 2)

        ts += timedelta(minutes=gap)
        if ts.hour >= 16:
            ts = ts.replace(hour=9, minute=30) + timedelta(days=1)
        if ts.weekday() >= 5:
            ts += timedelta(days=(7 - ts.weekday()))

        rows.append({
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "asset": asset, "side": side,
            "quantity": qty, "entry_price": entry_price,
            "exit_price": exit_price,
            "profit_loss": pnl, "balance": balance,
        })

    return pd.DataFrame(rows)


# ===========================================================================
GENERATORS = {
    "calm_trader.csv": generate_calm_trader,
    "overtrader.csv": generate_overtrader,
    "loss_averse_trader.csv": generate_loss_averse_trader,
    "revenge_trader.csv": generate_revenge_trader,
}


def generate_all(n_trades: int = 3000) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, fn in GENERATORS.items():
        out_path = OUTPUT_DIR / filename
        print(f"Generating {filename} ({n_trades} trades)...")
        df = fn(n_trades=n_trades)
        df.to_csv(out_path, index=False)
        print(f"  Saved {len(df)} rows → {out_path}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic trader CSV datasets")
    parser.add_argument(
        "--trades", type=int, default=3000,
        help="Number of trades per dataset (default: 3000)",
    )
    args = parser.parse_args()
    generate_all(n_trades=args.trades)
