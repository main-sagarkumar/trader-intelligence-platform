"""
Define reference schemas for trade transaction and trade-leg data.

These dictionaries document expected raw market activity fields for ingestion,
validation, and future extension toward more granular trade modeling.
"""

TRADE_TRANSACTION_SCHEMA = {
    "trade_id": "str",
    "trader_id": "str",
    "trade_timestamp": "datetime64[ns]",
    "strategy_type": "str",
    "asset_class": "str",
    "underlying": "str",
    "market_regime": "str",

    # Trade-Level Financial Metrics
    "total_pnl": "float",
    "max_drawdown": "float",
    "holding_minutes": "int",
    "leverage_used": "float",
    "margin_used": "float",

    # Behavioral Features
    "stop_loss_used": "int",
    "target_used": "int",
    "overnight_position": "int",
    "expiry_day_trade": "int",
    "emotional_score": "float",
    "revenge_trade_flag": "int",

    # Capital & Outcome Metrics
    "position_size_pct": "float",
    "net_roi_pct": "float",
    "outcome": "str"
}


TRADE_LEGS_SCHEMA = {
    "leg_id": "str",
    "trade_id": "str",
    "trader_id": "str",

    # Instrument Information
    "symbol": "str",
    "instrument_type": "str",   # CE / PE / FUT / EQ
    "strike_price": "float",
    "expiry_date": "datetime64[ns]",

    # Trade Direction
    "side": "str",              # BUY / SELL

    # Position Details
    "quantity": "int",
    "entry_price": "float",
    "exit_price": "float",

    # Greeks & Volatility
    "implied_volatility": "float",
    "delta": "float",
    "gamma": "float",
    "theta": "float",
    "vega": "float",

    # PnL & Execution Metrics
    "leg_pnl": "float",
    "stop_loss_hit": "int",
    "execution_slippage": "float",
    "transaction_cost": "float"
}
