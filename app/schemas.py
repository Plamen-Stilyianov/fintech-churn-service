from pydantic import BaseModel, Field

class ChurnPredictionInput(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Customer age in years")
    income_bracket: int = Field(..., ge=0, le=3, description="Encoded income tier (0=Low to 3=Very High)")
    active_products: int = Field(..., ge=0, description="Count of active internal financial products")
    app_logins_frequency: int = Field(..., ge=0, description="Monthly mobile app login interactions")
    tx_count: int = Field(..., ge=0, description="Total monthly transactions executed")
    satisfaction_score: int = Field(..., ge=1, le=6, description="Overall satisfaction survey score (1-6 scale)")

class ChurnPredictionResponse(BaseModel):
    churn_probability: float = Field(..., description="Calculated 30-day forecast risk index from 0.0 to 1.0")
    high_risk_flag: bool = Field(..., description="True if score breaches the 0.50 median threshold")
