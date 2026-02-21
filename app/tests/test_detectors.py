"""Unit tests for bias detectors."""

import pytest
from datetime import datetime, timedelta

from app.models.schemas import NormalizedTrade
from app.detectors.overtrading import detect_overtrading
from app.detectors.loss_aversion import detect_loss_aversion
from app.detectors.revenge import detect_revenge_trading
from app.detectors.recency import detect_recency_bias


@pytest.fixture
def sample_trades():
    """Create sample trades for testing."""
    trades = []
    base_time = datetime(2024, 1, 1, 9, 0, 0)
    balance = 10000
    
    for i in range(20):
        timestamp = base_time + timedelta(minutes=i * 5)
        is_win = i % 2 == 0
        pnl = 100 if is_win else -80
        balance += pnl
        
        trade = NormalizedTrade(
            trade_id=f"TRADE_{i:03d}",
            timestamp=timestamp,
            asset="AAPL",
            side="BUY" if i % 2 == 0 else "SELL",
            quantity=1.0,
            entry_price=150.0,
            exit_price=151.0 if is_win else 149.0,
            profit_loss=pnl,
            balance=balance,
            date=timestamp.strftime("%Y-%m-%d"),
            hour=timestamp.hour,
            weekday=timestamp.weekday(),
            is_win=is_win,
            abs_pnl=abs(pnl),
            size_usd=150.0,
            return_pct=pnl / 150.0,
        )
        trades.append(trade)
    
    return trades


def test_overtrading_detector(sample_trades):
    """Test overtrading detector."""
    result = detect_overtrading(sample_trades)
    
    assert result.score >= 0
    assert result.score <= 100
    assert result.severity in ["low", "med", "high"]
    assert isinstance(result.flagged_trade_ids, list)
    assert isinstance(result.stats, dict)


def test_loss_aversion_detector(sample_trades):
    """Test loss aversion detector."""
    result = detect_loss_aversion(sample_trades)
    
    assert result.score >= 0
    assert result.score <= 100
    assert result.severity in ["low", "med", "high"]
    assert isinstance(result.flagged_trade_ids, list)
    assert "win_rate" in result.stats


def test_revenge_detector(sample_trades):
    """Test revenge trading detector."""
    result = detect_revenge_trading(sample_trades)
    
    assert result.score >= 0
    assert result.score <= 100
    assert result.severity in ["low", "med", "high"]


def test_recency_detector(sample_trades):
    """Test recency bias detector."""
    result = detect_recency_bias(sample_trades)
    
    assert result.score >= 0
    assert result.score <= 100
    assert result.severity in ["low", "med", "high"]
    assert "signal1_flags" in result.stats


def test_empty_trades():
    """Test detectors with empty trades."""
    empty_trades = []
    
    assert detect_overtrading(empty_trades).score == 0
    assert detect_loss_aversion(empty_trades).score == 0
    assert detect_revenge_trading(empty_trades).score == 0
    assert detect_recency_bias(empty_trades).score == 0
