from pydantic import BaseModel, Field

class ChurnPredictionInput(BaseModel):
    days_since_last_login: int = Field(..., ge=0, description="Inactivity metric")
    transaction_drop_percentage: float = Field(..., ge=0.0, le=1.0, description="Drop in volume over 30 days")
    payment_failure_count: int = Field(..., ge=0, description="Count of failed transaction events")

class ChurnPredictionResponse(BaseModel):
    churn_probability: float
    high_risk_flag: bool
