from configs.model_config import EARLY_FEATURES

FEATURE_LABELS = {
    "early_total_trades": "Trade Activity",
    "early_avg_pnl": "Average PnL",
    "early_win_rate": "Win Rate",
    "early_avg_holding_minutes": "Holding Duration",
    "early_avg_leverage": "Leverage Usage",
    "early_avg_risk_pct": "Risk Percentage",
    "early_stop_loss_usage_rate": "Stop Loss Usage",
    "early_overnight_position_rate": "Overnight Positions"
}


def get_top_feature_reasons(
    shap_values,
    predicted_cluster,
    top_n=3
):
    contributions = shap_values.values[
        0, :, predicted_cluster
    ]

    feature_scores = list(
        zip(EARLY_FEATURES, contributions)
    )

    feature_scores.sort(
        key=lambda x: abs(x[1]),
        reverse=True
    )

    return feature_scores[:top_n]


def build_detailed_explanations(
    shap_values,
    sample_row,
    predicted_cluster,
    top_n=3
):

    contributions = shap_values.values[
        0, :, predicted_cluster
    ]

    feature_scores = list(
        zip(
            EARLY_FEATURES,
            contributions
        )
    )

    feature_scores.sort(
        key=lambda x: abs(x[1]),
        reverse=True
    )

    explanations = []

    for feature, score in feature_scores[:top_n]:

        feature_value = sample_row[
            feature
        ].iloc[0]

        label = FEATURE_LABELS[
            feature
        ]

        explanations.append({
            "feature": label,
            "value": round(
                float(feature_value),
                2
            ),
            "impact": round(
                float(score),
                4
            )
        })

    return explanations