"""Market shock event generator."""

from datetime import datetime
from app.models.schemas import EventData


def generate_market_shocks_from_trades(trades) -> list[EventData]:
    """
    Generate synthetic market shock events from trade PnL spikes.
    
    Methodology:
    - Identify trades with extreme PnL (top 10%)
    - Treat as "market shocks" for context
    - Label as Large Loss or Large Gain
    """
    if not trades:
        return []
    
    shocks = []
    
    # Compute PnL magnitudes
    pnl_abs = [abs(t.profit_loss) for t in trades]
    if not pnl_abs:
        return []
    
    sorted_abs = sorted(pnl_abs)
    p90 = sorted_abs[int(len(sorted_abs) * 0.9)] if len(sorted_abs) >= 10 else max(sorted_abs)
    
    # Identify shocks
    for trade in trades:
        if abs(trade.profit_loss) >= p90:
            shock = EventData(
                timestamp=trade.timestamp,
                event_type="market_shock",
                label=f"{'Loss' if trade.profit_loss < 0 else 'Gain'} spike: ${abs(trade.profit_loss):.0f}",
                symbols=[trade.asset],
                pnl_magnitude=trade.profit_loss,
            )
            shocks.append(shock)
    
    return shocks
