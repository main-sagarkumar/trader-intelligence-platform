"""
API Schemas
-----------
Pydantic request and response models
for Trader Intelligence API.
"""

from pydantic import BaseModel


# ──────────────────────────────────────────────
# Current Trader Prediction
# ──────────────────────────────────────────────

class CurrentTraderRequest(BaseModel):
    total_trades: int
    avg_pnl: float
    roi_pct: float
    avg_holding_minutes: float
    avg_leverage: float
    win_rate: float


# ──────────────────────────────────────────────
# Future Trader Prediction
# ──────────────────────────────────────────────

class FutureTraderRequest(BaseModel):
    early_total_trades: int
    early_avg_pnl: float
    early_win_rate: float
    early_avg_holding_minutes: float
    early_avg_leverage: float
    early_avg_risk_pct: float
    early_stop_loss_usage_rate: float
    early_overnight_position_rate: float

# ──────────────────────────────────────────────
# Response Models
# ──────────────────────────────────────────────

class CurrentTraderResponse(BaseModel):
    cluster: int
    segment: str
    description: str
    recommendations: list[str]


class FutureTraderResponse(BaseModel):
    cluster: int
    confidence: float
    segment: str
    description: str
    recommendations: list[str]