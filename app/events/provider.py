"""Events provider interface and implementations."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from app.models.schemas import EventData


class EventsProvider(ABC):
    """Abstract base class for events providers."""
    
    @abstractmethod
    def get_events(self, start_date: datetime, end_date: datetime) -> list[EventData]:
        """Retrieve events for date range."""
        pass


class StubEventsProvider(EventsProvider):
    """Stub provider that returns empty results."""
    
    def get_events(self, start_date: datetime, end_date: datetime) -> list[EventData]:
        """Return empty list (stub for future integration)."""
        return []


class MarketShockGenerator:
    """Generate market shock events from trades."""
    
    @staticmethod
    def generate_shocks(trades) -> list[EventData]:
        """
        Generate market shock events based on large PnL spikes.
        Useful for demo/offline analysis.
        """
        if not trades:
            return []
        
        shocks = []
        
        # Analyze pnl distribution
        pnls = [abs(t.profit_loss) for t in trades]
        sorted_pnls = sorted(pnls)
        
        # High-magnitude threshold: top 10-percentile
        threshold_idx = int(len(sorted_pnls) * 0.9)
        high_pnl_threshold = sorted_pnls[threshold_idx] if threshold_idx < len(sorted_pnls) else 0
        
        # Find trades exceeding threshold
        for trade in trades:
            if abs(trade.profit_loss) >= high_pnl_threshold:
                shock_event = EventData(
                    timestamp=trade.timestamp,
                    event_type="market_shock",
                    label=f"{'Large Loss' if trade.profit_loss < 0 else 'Large Gain'} Detected",
                    symbols=[trade.asset],
                    pnl_magnitude=trade.profit_loss,
                )
                shocks.append(shock_event)
        
        return shocks
