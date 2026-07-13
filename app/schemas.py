from pydantic import BaseModel, Field

class ChurnPredictionInput(BaseModel):
    step: int = Field(..., ge=0, description="Simulation temporal hour index")
    amount: float = Field(..., ge=0.0, description="Absolute monetary transaction value")
    oldbalanceOrg: float = Field(..., ge=0.0, description="Origin balance prior to execution")
    newbalanceOrig: float = Field(..., ge=0.0, description="Origin balance after execution")

class ChurnPredictionResponse(BaseModel):
    churn_probability: float
    high_risk_flag: bool
