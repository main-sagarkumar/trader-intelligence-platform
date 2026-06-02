'''
Business definitions generated from clustering analysis.

These definitions are intentionally separated from the ML code so they can
be reused by dashboards, recommendation engines, APIs, and future models.
'''

CLUSTER_MAPPING = {
    0: "Extreme Risk Traders",
    1: "Conservative Income Traders",
    2: "High Activity Traders",
    3: "Structured Strategy Traders",
    4: "Aggressive Speculators"
}

CLUSTER_DESCRIPTIONS = {
    0: (
        "High leverage traders with the lowest win rates and worst ROI. "
        "Exhibit severe risk-taking behavior and frequent capital destruction."
    ),

    1: (
        "Disciplined traders with low leverage, long holding periods, "
        "high win rates, and consistent profitability."
    ),

    2: (
        "Highly active traders with very high trade frequency, short "
        "holding periods, and moderate profitability."
    ),

    3: (
        "Structured traders using systematic approaches with positive ROI, "
        "good win rates, and controlled risk."
    ),

    4: (
        "Risk-seeking traders using high leverage with below-average win rates "
        "and slightly negative profitability."
    )
}

CLUSTER_RECOMMENDATIONS = {

    0: [
        "Reduce leverage usage",
        "Improve position sizing",
        "Focus on capital preservation",
        "Use defined-risk strategies"
    ],

    1: [
        "Explore advanced income strategies",
        "Use portfolio analytics tools",
        "Consider premium strategy products",
        "Continue disciplined risk management"
    ],

    2: [
        "Track trading costs and slippage",
        "Use trade journaling",
        "Focus on trade quality over quantity",
        "Monitor overtrading behavior"
    ],

    3: [
        "Expand strategy portfolio",
        "Use advanced backtesting tools",
        "Explore portfolio optimization",
        "Continue systematic execution"
    ],

    4: [
        "Reduce leverage gradually",
        "Improve risk-reward selection",
        "Adopt stricter stop losses",
        "Focus on consistency over large gains"
    ]
}