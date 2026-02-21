"""
Generate synthetic training datasets for the XGBoost bias classifier.

Creates four CSV files (one per bias class) in our standard NBH format:
    timestamp, asset, side, quantity, entry_price, exit_price, profit_loss, balance

Design philosophy:
  - Each class has a PRIMARY signal (the dominant bias) but also exhibits
    SECONDARY behaviours from other classes at a lower rate.
  - ~20% of each class's trades are "mixed" (borrowed from another bias profile)
    so class boundaries are realistic rather than perfectly clean.
  - Multiple seeds produce meaningfully different trade histories.

Usage:
    python -m app.ml.generate_datasets
    python -m app.ml.generate_datasets --trades 5000
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
def generate_calm_trader(n_trades: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Calm / disciplined trader — primary signal: consistent sizing, moderate
    win rate (~56-60%), balanced hold times.
    Noise: ~15% of days have slightly elevated trade frequency or size variation
    to prevent the model from using "always calm" as a trivial shortcut.
    """
    np.random.seed(seed)
    random.seed(seed)

    rows = []
    balance = 25_000.0
    ts = datetime(2023, 1, 3, 9, 30)
    i = 0

    while i < n_trades:
        # Occasionally have a slightly busier day (15% chance)
        is_busier_day = np.random.random() < 0.15
        day_trades = np.random.randint(12, 22) if is_busier_day else np.random.randint(6, 14)

        for _ in range(day_trades):
            if i >= n_trades:
                break
            asset = random.choice(ASSETS)
            base_price = ASSET_PRICE[asset]
            entry_price = _random_price_move(base_price, 0.005)
            side = random.choice(["BUY", "SELL"])

            # Occasionally size up slightly (not revenge — just conviction)
            base_qty = random.randint(10, 100)
            qty = base_qty if np.random.random() > 0.1 else int(base_qty * np.random.uniform(1.1, 1.5))

            win = np.random.random() < (0.60 if not is_busier_day else 0.52)
            pnl_pct = np.random.uniform(0.002, 0.012) if win else -np.random.uniform(0.002, 0.010)
            # Slight noise on pnl
            pnl_pct *= np.random.uniform(0.85, 1.15)

            pnl = round(qty * entry_price * pnl_pct, 2)
            exit_price = round(entry_price + (pnl / qty), 2)
            balance = round(balance + pnl, 2)

            gap = int(np.random.normal(25 if is_busier_day else 38, 12))
            gap = max(4, gap)
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
            i += 1

        ts = ts.replace(hour=9, minute=30) + timedelta(days=1)
        if ts.weekday() >= 5:
            ts += timedelta(days=(7 - ts.weekday()))

    return pd.DataFrame(rows)


# ===========================================================================
def generate_overtrader(n_trades: int = 5000, seed: int = 101) -> pd.DataFrame:
    """
    Overtrader — primary signal: high trade frequency, burst clusters.
    Noise:
      - ~20% of days are "recovery days" with calm pacing (15 min gaps)
      - ~15% of burst trades have normal sizing (not always impulsive)
      - Win rate varies 42-50% day to day
    """
    np.random.seed(seed)
    random.seed(seed)

    rows = []
    balance = 25_000.0
    ts = datetime(2023, 1, 3, 9, 30)
    i = 0

    while i < n_trades:
        is_spike_day    = np.random.random() < 0.30
        is_recovery_day = (not is_spike_day) and np.random.random() < 0.20
        day_win_rate    = np.random.uniform(0.42, 0.50)

        if is_spike_day:
            day_trades = np.random.randint(50, 130)
            for _ in range(day_trades):
                if i >= n_trades:
                    break
                burst_size = np.random.randint(3, 7)
                for _b in range(burst_size):
                    if i >= n_trades:
                        break
                    asset = random.choice(ASSETS)
                    base_price = ASSET_PRICE[asset]
                    entry_price = _random_price_move(base_price, 0.006)
                    side = random.choice(["BUY", "SELL"])
                    # 15% chance of normal sizing even during burst
                    qty = random.randint(10, 120) if np.random.random() < 0.15 else random.randint(10, 200)
                    win = np.random.random() < day_win_rate
                    pnl_pct = np.random.uniform(0.001, 0.008) if win else -np.random.uniform(0.001, 0.009)
                    pnl_pct *= np.random.uniform(0.85, 1.15)
                    pnl = round(qty * entry_price * pnl_pct, 2)
                    exit_price = round(entry_price + (pnl / qty), 2)
                    balance = round(balance + pnl, 2)
                    ts += timedelta(minutes=np.random.randint(1, 4))
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
                ts += timedelta(minutes=np.random.randint(5, 15))

        elif is_recovery_day:
            # Calm day — similar to a calm trader (adds noise/overlap)
            day_trades = np.random.randint(6, 13)
            for _ in range(day_trades):
                if i >= n_trades:
                    break
                asset = random.choice(ASSETS)
                base_price = ASSET_PRICE[asset]
                entry_price = _random_price_move(base_price, 0.005)
                side = random.choice(["BUY", "SELL"])
                qty = random.randint(10, 100)
                win = np.random.random() < 0.55
                pnl_pct = np.random.uniform(0.002, 0.010) if win else -np.random.uniform(0.002, 0.009)
                pnl_pct *= np.random.uniform(0.85, 1.15)
                pnl = round(qty * entry_price * pnl_pct, 2)
                exit_price = round(entry_price + (pnl / qty), 2)
                balance = round(balance + pnl, 2)
                gap = int(np.random.normal(30, 10))
                gap = max(5, gap)
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
                i += 1

        else:
            # Normal overtrading day: 15-25 trades, 8-20 min gaps
            day_trades = np.random.randint(15, 26)
            for _ in range(day_trades):
                if i >= n_trades:
                    break
                asset = random.choice(ASSETS)
                base_price = ASSET_PRICE[asset]
                entry_price = _random_price_move(base_price, 0.005)
                side = random.choice(["BUY", "SELL"])
                qty = random.randint(10, 120)
                win = np.random.random() < day_win_rate
                pnl_pct = np.random.uniform(0.001, 0.008) if win else -np.random.uniform(0.001, 0.009)
                pnl_pct *= np.random.uniform(0.85, 1.15)
                pnl = round(qty * entry_price * pnl_pct, 2)
                exit_price = round(entry_price + (pnl / qty), 2)
                balance = round(balance + pnl, 2)
                gap = int(np.random.normal(14, 5))
                gap = max(3, gap)
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
                i += 1

        ts = ts.replace(hour=9, minute=30) + timedelta(days=1)
        if ts.weekday() >= 5:
            ts += timedelta(days=(7 - ts.weekday()))

    return pd.DataFrame(rows)


# ===========================================================================

def generate_loss_averse_trader(n_trades: int = 5000, seed: int = 202) -> pd.DataFrame:
    """
    Loss averse trader — primary signal: holds losses ~4-8× longer than wins.
    Primary: cuts winners in 5-20 min, holds losers 60-240 min.
    Noise: ~20% of losses are cut at normal speed ("disciplined day"), and
    ~15% of wins are held longer ("greedy day"), to add feature variance.
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
        qty = random.randint(20, 100)

        is_win = np.random.random() < 0.50
        # Noise flags
        normal_loss_cut = np.random.random() < 0.20  # cuts loss quickly (not averse this trade)
        held_winner     = np.random.random() < 0.15  # holds winner too long

        if is_win:
            pnl_pct = np.random.uniform(0.001, 0.006)   # small win
            gap = int(np.random.normal(50, 20)) if held_winner else int(np.random.normal(10, 4))
        else:
            pnl_pct = -np.random.uniform(0.008, 0.028)  # large loss
            gap = int(np.random.normal(20, 8)) if normal_loss_cut else int(np.random.normal(120, 40))

        # Add pnl noise
        pnl_pct *= np.random.uniform(0.85, 1.15)
        gap = max(2, gap)
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
def generate_revenge_trader(n_trades: int = 5000, seed: int = 303) -> pd.DataFrame:
    """
    Revenge trader — primary signal: size escalation + rapid re-entry after losses.
    Noise: ~25% of post-loss trades are normal-sized and normally-timed ("composure"),
    and loss streaks sometimes reset mid-day to prevent trivially perfect detection.
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

        # Compose flag: ~25% chance this post-loss trade is disciplined (no revenge)
        compose = (consecutive_losses > 0) and (np.random.random() < 0.25)

        if compose:
            # Disciplined re-entry despite prior loss
            qty = max(10, int(base_qty * np.random.uniform(0.8, 1.2)))
            gap = int(np.random.normal(20, 8))
        else:
            # Scale up size after losses (primary revenge signal)
            size_multiplier = 1.0 + min(consecutive_losses * np.random.uniform(0.3, 0.7), 2.5)
            qty = max(10, int(base_qty * size_multiplier * np.random.uniform(0.8, 1.3)))
            if consecutive_losses >= 1:
                gap = int(np.random.normal(5, 3))   # quick revenge re-entry
            else:
                gap = int(np.random.normal(18, 7))

        is_win = np.random.random() < 0.45

        if is_win:
            pnl_pct = np.random.uniform(0.003, 0.012)
            pnl_pct *= np.random.uniform(0.85, 1.15)
            if not compose:
                consecutive_losses = 0
            # Occasionally reset streak mid-day after a big win
            if np.random.random() < 0.3:
                consecutive_losses = 0
        else:
            pnl_pct = -np.random.uniform(0.005, 0.015)
            pnl_pct *= np.random.uniform(0.85, 1.15)
            consecutive_losses += 1

        gap = max(1, gap)
        pnl = round(qty * entry_price * pnl_pct, 2)
        exit_price = round(entry_price + (pnl / qty), 2)
        balance = round(balance + pnl, 2)

        ts += timedelta(minutes=gap)
        if ts.hour >= 16:
            ts = ts.replace(hour=9, minute=30) + timedelta(days=1)
            consecutive_losses = 0  # reset streak at day boundary
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

# Default seeds per class — multiple seeds add variance and prevent the model
# from memorising seed-specific artefacts. Seeds are spread wide intentionally.
_DEFAULT_SEEDS: dict[str, list[int]] = {
    "calm_trader.csv":        [42,   777,  1337],
    "overtrader.csv":         [101,  888,  2024],
    "loss_averse_trader.csv": [202,  999,  3141],
    "revenge_trader.csv":     [303, 1111,  4242],
}


def generate_all(n_trades: int = 5000) -> None:
    """Generate one CSV per class (single seed) into OUTPUT_DIR."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, fn in GENERATORS.items():
        out_path = OUTPUT_DIR / filename
        print(f"Generating {filename} ({n_trades} trades)...")
        df = fn(n_trades=n_trades)
        df.to_csv(out_path, index=False)
        print(f"  Saved {len(df)} rows → {out_path}")
    print("Done.")


def generate_multi_seed(
    n_trades: int = 5000,
    seeds: dict[str, list[int]] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Generate each class with multiple seeds and concatenate.
    Returns a dict of filename → combined DataFrame.
    Calling code (train_xgboost) uses this to build a richer, less seed-biased
    training set.
    """
    if seeds is None:
        seeds = _DEFAULT_SEEDS
    result: dict[str, pd.DataFrame] = {}
    for filename, fn in GENERATORS.items():
        frames = []
        for seed in seeds.get(filename, [42]):
            df = fn(n_trades=n_trades, seed=seed)
            frames.append(df)
        result[filename] = pd.concat(frames, ignore_index=True)
        print(f"  {filename}: {len(result[filename])} rows ({len(seeds.get(filename, [42]))} seeds × {n_trades})")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic trader CSV datasets")
    parser.add_argument(
        "--trades", type=int, default=5000,
        help="Number of trades per dataset per seed (default: 5000)",
    )
    args = parser.parse_args()
    generate_all(n_trades=args.trades)
