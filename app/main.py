import os
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from app.schemas import ChurnPredictionInput, ChurnPredictionResponse

app = FastAPI(title="Fintech Churn Prediction Service", version="2.0.0")

ARTIFACT_PATH = os.path.join(os.path.dirname(__file__), "artifacts", "churn_model.pkl")
model = None


@app.get("/")
async def root_health_check():
    """
    Root endpoint serving as an automated health monitoring probe
    for load balancers, gateway checks, and container runtimes.
    """
    return {
        "status": "healthy",
        "service": "Fintech Churn Prediction Service",
        "engine_mode": "fallback_rules" if model is None else "production_ml"
    }


@app.on_event("startup")
def load_model_artifacts():
    global model
    if os.path.exists(ARTIFACT_PATH):
        try:
            with open(ARTIFACT_PATH, "rb") as f:
                model = pickle.load(f)
            print("Production PaySim-tuned tree classifier loaded successfully.")
        except Exception as e:
            print(f"Error unpickling serialization weights: {str(e)}")
    else:
        print("Warning: Serialized pkl file missing. Running on fallback rule matrix.")


@app.post("/predict", response_model=ChurnPredictionResponse)
async def evaluate_churn_risk(payload: ChurnPredictionInput):
    try:
        # Construct the execution matrix row matching training pipeline order
        input_data = np.array([[
            payload.step,
            payload.amount,
            payload.oldbalanceOrg,
            payload.newbalanceOrig
        ]])

        if model is not None:
            # Extract prediction probability matrix slice for positive churn target class
            prob = float(model.predict_proba(input_data)[0][1])
        else:
            # Fallback domain logic if model file isn't created yet
            balance_drain = payload.oldbalanceOrg - payload.newbalanceOrig
            if balance_drain > 50000.0 or payload.step > 500:
                prob = 0.88
            else:
                prob = 0.12

        return {
            "churn_probability": round(prob, 4),
            "high_risk_flag": prob >= 0.70
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failure: {str(e)}")
