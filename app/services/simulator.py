"""What-If Simulator for trading rules."""

from datetime import timedelta
from collections import defaultdict

from app.models.schemas import (
    NormalizedTrade,
    SimulatedTrade,
    SimulationResult,
    SimulationRule,
)


def simulate_with_rules(trades: list[NormalizedTrade], rules: SimulationRule) -> SimulationResult:
    """
    Replay trade stream with rules and compute impact.
    
    Rules:
    - cooldown_after_loss_minutes: Don't trade for X minutes after loss
    - daily_trade_cap: Max trades per day
    - max_position_size_multiplier: Max size as multiple of median
    - stop_after_consecutive_losses: Stop after N consecutive losses
    
    Returns simulation with:
    - baseline vs simulated metrics
    - blocked trades with reasons
    - per-rule impact summary
    """
    if not trades:
        return SimulationResult(
            baseline_pnl=0,
            baseline_trade_count=0,
            simulated_pnl=0,
            simulated_trade_count=0,
            pnl_difference=0,
            blocked_trades=[],
            simulation_log=[],
        )
    
    # Baseline metrics
    baseline_pnl = sum(t.profit_loss for t in trades)
    baseline_trade_count = len(trades)
    
    # Compute baseline size
    sizes = [t.size_usd for t in trades if t.size_usd > 0]
    baseline_size = sorted(sizes)[len(sizes) // 2] if sizes else 0
    max_size = baseline_size * (rules.max_position_size_multiplier or 1.5)
    
    # Simulation state
    simulated_pnl = 0
    simulated_balance = trades[0].balance - trades[0].profit_loss if trades else 0
    simulated_trades = []
    blocked_trades = []
    daily_trade_counts = defaultdict(int)
    last_loss_time = None
    consecutive_losses = 0
    
    rule_impacts = {
        "cooldown": {"skipped": 0, "pnl_impact": 0},
        "daily_cap": {"skipped": 0, "pnl_impact": 0},
        "size_cap": {"skipped": 0, "pnl_impact": 0},
        "loss_streak": {"skipped": 0, "pnl_impact": 0},
    }
    
    # Simulate each trade
    for trade in trades:
        allowed = True
        reason = None
        
        # Check cooldown after loss
        if last_loss_time and rules.cooldown_after_loss_minutes:
            minutes_elapsed = (trade.timestamp - last_loss_time).total_seconds() / 60
            if minutes_elapsed < rules.cooldown_after_loss_minutes:
                allowed = False
                reason = f"Cooldown after loss ({int(minutes_elapsed)}m elapsed, {rules.cooldown_after_loss_minutes}m required)"
                rule_impacts["cooldown"]["skipped"] += 1
                rule_impacts["cooldown"]["pnl_impact"] -= trade.profit_loss
        
        # Check daily trade cap
        if allowed and rules.daily_trade_cap:
            trade_date = trade.date
            if daily_trade_counts[trade_date] >= rules.daily_trade_cap:
                allowed = False
                reason = f"Daily trade cap reached ({daily_trade_counts[trade_date]}/{rules.daily_trade_cap})"
                rule_impacts["daily_cap"]["skipped"] += 1
                rule_impacts["daily_cap"]["pnl_impact"] -= trade.profit_loss
        
        # Check max position size
        if allowed and rules.max_position_size_multiplier:
            if trade.size_usd > max_size:
                allowed = False
                reason = f"Position size exceeds cap ({trade.size_usd:.2f} > {max_size:.2f})"
                rule_impacts["size_cap"]["skipped"] += 1
                rule_impacts["size_cap"]["pnl_impact"] -= trade.profit_loss
        
        # Check consecutive loss stop
        if allowed and rules.stop_after_consecutive_losses:
            if consecutive_losses >= rules.stop_after_consecutive_losses:
                allowed = False
                reason = f"Stop after {consecutive_losses} consecutive losses"
                rule_impacts["loss_streak"]["skipped"] += 1
                rule_impacts["loss_streak"]["pnl_impact"] -= trade.profit_loss
        
        # Execute trade if allowed
        if allowed:
            simulated_pnl += trade.profit_loss
            simulated_balance += trade.profit_loss
            daily_trade_counts[trade.date] += 1
            
            # Update loss tracking
            if trade.is_win:
                consecutive_losses = 0
                last_loss_time = None
            else:
                consecutive_losses += 1
                last_loss_time = trade.timestamp
        else:
            # Track blocked trade
            blocked_trade = SimulatedTrade(
                trade_id=trade.trade_id,
                timestamp=trade.timestamp,
                asset=trade.asset,
                profit_loss=trade.profit_loss,
                balance_before=simulated_balance,
                balance_after=simulated_balance,
                allowed=False,
                reason=reason,
            )
            blocked_trades.append(blocked_trade)
    
    # Compute differences
    pnl_difference = simulated_pnl - baseline_pnl
    simulated_trade_count = len(trades) - len(blocked_trades)
    
    # Create simulation log
    simulation_log = [
        {
            "rule": "cooldown_after_loss",
            "trades_skipped": rule_impacts["cooldown"]["skipped"],
            "pnl_impact": round(rule_impacts["cooldown"]["pnl_impact"], 2),
        },
        {
            "rule": "daily_trade_cap",
            "trades_skipped": rule_impacts["daily_cap"]["skipped"],
            "pnl_impact": round(rule_impacts["daily_cap"]["pnl_impact"], 2),
        },
        {
            "rule": "max_position_size",
            "trades_skipped": rule_impacts["size_cap"]["skipped"],
            "pnl_impact": round(rule_impacts["size_cap"]["pnl_impact"], 2),
        },
        {
            "rule": "stop_after_loss_streak",
            "trades_skipped": rule_impacts["loss_streak"]["skipped"],
            "pnl_impact": round(rule_impacts["loss_streak"]["pnl_impact"], 2),
        },
    ]
    
    return SimulationResult(
        baseline_pnl=round(baseline_pnl, 2),
        baseline_trade_count=baseline_trade_count,
        simulated_pnl=round(simulated_pnl, 2),
        simulated_trade_count=simulated_trade_count,
        pnl_difference=round(pnl_difference, 2),
        blocked_trades=blocked_trades,
        simulation_log=simulation_log,
    )
