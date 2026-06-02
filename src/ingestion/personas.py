TRADER_PERSONAS = {

    "scalper_option_buyer": {
        "avg_trades_per_day": 15,
        "holding_minutes_range": (5, 60),
        "leverage_range": (3, 7),
        "preferred_strategies": [
            "Long Call",
            "Long Put"
        ],
        "stop_loss_discipline": 0.4,
        "risk_appetite": 0.9,
        "overnight_holding_probability": 0.05,
        "win_probability": 0.40,
        "risk_per_trade_range": (0.05, 0.10)
    },

    "disciplined_option_seller": {
        "avg_trades_per_day": 3,
        "holding_minutes_range": (300, 5000),
        "leverage_range": (1, 3),
        "preferred_strategies": [
            "Iron Condor",
            "Short Straddle"
        ],
        "stop_loss_discipline": 0.9,
        "risk_appetite": 0.4,
        "overnight_holding_probability": 0.9,
        "win_probability": 0.68,
        "risk_per_trade_range": (0.01, 0.03)
    },

    "spread_trader": {
        "avg_trades_per_day": 2,
        "holding_minutes_range": (60, 2000),
        "leverage_range": (1, 4),
        "preferred_strategies": [
            "Bull Call Spread",
            "Bear Put Spread"
        ],
        "stop_loss_discipline": 0.8,
        "risk_appetite": 0.5,
        "overnight_holding_probability": 0.8,
        "win_probability": 0.60,
        "risk_per_trade_range": (0.01, 0.04)
    },

    "expiry_day_gambler": {
        "avg_trades_per_day": 25,
        "holding_minutes_range": (1, 20),
        "leverage_range": (5, 10),
        "preferred_strategies": [
            "Long Call",
            "Long Put"
        ],
        "stop_loss_discipline": 0.2,
        "risk_appetite": 1.0,
        "overnight_holding_probability": 0.0,
        "win_probability": 0.28,
        "risk_per_trade_range": (0.10, 0.40)
    }
}