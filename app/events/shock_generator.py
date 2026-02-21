"""Market shock event generator."""

import numpy as np
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
    
    pnl_abs = np.fromiter((abs(t.profit_loss) for t in trades), dtype=np.float64, count=len(trades))
    if len(pnl_abs) < 10:
        p90 = float(pnl_abs.max()) if len(pnl_abs) > 0 else 0.0
    else:
        p90 = float(np.percentile(pnl_abs, 90))
    
    shocks = [
        EventData(
            timestamp=t.timestamp,
            event_type="market_shock",
            label=f"{'Loss' if t.profit_loss < 0 else 'Gain'} spike: ${abs(t.profit_loss):.0f}",
            symbols=[t.asset],
            pnl_magnitude=t.profit_loss,
        )
        for t in trades
        if abs(t.profit_loss) >= p90
    ]
    return shocks
