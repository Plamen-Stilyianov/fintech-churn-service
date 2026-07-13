# Fintech Churn Prediction Service

An end-to-end, production-ready machine learning service designed to predict user churn for digital wallet and fintech platforms. This repository houses the exploratory data structures, feature engineering pipelines, model artifacts, and a containerized FastAPI application ready for cloud deployment.

## 🏗️ System Architecture

The service bridges exploratory data analysis with scalable production inference:

```text
fintech-churn-service/
│
├── notebooks/
│   └── exploration_and_tuning.ipynb  # Feature engineering, EDA, and model tuning
├── app/
│   ├── __init__.py
│   ├── main.py                       # FastAPI application & endpoint routing
│   ├── schemas.py                    # Pydantic data validation types
│   └── artifacts/
│       └── churn_model.pkl           # Trained serialization weights (XGBoost/LightGBM)
├── requirements.txt                  # Strict production environments package lock
└── Dockerfile                        # Lightweight, multi-stage Linux runtime container
```

## 🛠️ Tech Stack & Ecosystem

- **Backend Framework:** FastAPI, Uvicorn, Pydantic (Data validation and type safety)
- **Machine Learning Core:** XGBoost, Scikit-Learn, Pandas (Tabular array modeling)
- **Model Explainability:** SHAP (Shapley Additive exPlanations for feature tracking)
- **DevOps & Portability:** Docker, Python 3.12-slim (Cross-platform Linux execution parity)

## 🚀 Quickstart: Local Development

### 1. Ingest Synthetic Dataset
Before training or executing inference loops, download the synthetic digital wallet transaction stream:
```bash
pip install datasets pandas
python download_dataset.py
```
This stages the transaction log ledger directly inside the `notebooks/` workspace for exploration.

### 2. Stand Up the Application Environment
Install development configurations locally:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: .\venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Launch the Local Service
Spin up the FastAPI server via Uvicorn:
```bash
uvicorn app.main:app --reload --port 8000
```
Navigate to `http://localhost:8000/docs` in your browser to interact with the auto-generated Swagger UI engine.

## 🐳 Container Production Deployment

To package the service natively into a predictable Linux runtime container (matching the local openSUSE Tumbleweed or remote cloud environments):

```bash
# Build the application image
docker build -t fintech-churn-service:latest .

# Run the container bound to host networking interfaces
docker run -p 8000:8000 fintech-churn-service:latest
```

## 📋 Data Feature Schemas

The incoming JSON payloads processed by the `/v1/predict` endpoint evaluate user risk footprints across three primary behavioral aggregates:

| Metric | Type | Validation Constraint | Description |
| :--- | :--- | :--- | :--- |
| `days_since_last_login` | Integer | `>= 0` | Digital interaction inactivity metric |
| `transaction_drop_percentage` | Float | `0.0` to `1.0` | 30-day velocity drop in transactional value |
| `payment_failure_count` | Integer | `>= 0` | Historical frequency of ledger gateway declines |
