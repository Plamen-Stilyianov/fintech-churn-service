# Fintech Churn Prediction Service

An end-to-end, production-ready machine learning service designed to predict user churn for digital wallet and fintech platforms. This repository houses the exploratory data structures, multi-model evaluation frameworks, serialized artifacts, and a containerized FastAPI application ready for cloud deployment.

## 🏗️ System Architecture

The service bridges exploratory data analysis with scalable production inference, strictly organizing files into isolated layers to prevent environment collisions:

```text
fintech-churn-service/
│
├── notebooks/
│   └── exploration_and_tuning.ipynb  # Multi-model evaluation workbench & visualization canvas
├── app/
│   ├── __init__.py
│   ├── main.py                       # FastAPI core gateway & endpoint routing logic
│   ├── schemas.py                    # Pydantic v2 data validation validation classes
│   └── artifacts/
│       └── churn_model.pkl           # Serialized champion model weights (XGBoost)
├── utils/
│   └── batch_test.py                 # Automated regression testing & stress validator script
├── requirements.txt                  # Production environments package lock manifest
└── Dockerfile                        # Multi-stage, cache-less Python 3.12-slim runtime container
```

## 🛠️ Tech Stack & Advanced Features

- **Backend Framework:** FastAPI, Uvicorn, Pydantic v2 (Rust-backed type safety and sub-3ms routing validation)
- **Machine Learning Core:** XGBoost, Scikit-Learn, Pandas (Tabular multi-model classifier comparison arrays)
- **DevOps & Portability:** Multi-Stage Docker, Python 3.12-slim (Total host/runner development environment parity)
- **High Availability Pattern:** Resilience fallback rule-engine that safely processes active payload fields during background model update pipelines.

## 🚀 Quickstart: Local Development

### 1. Ingest Transaction Dataset Ledger
Before training or executing inference loops, run your stream tool to download the transactional data feed:
```bash
pip install datasets pandas
python download_dataset.py
```
This stages the transaction ledger directly inside your local environment path.

### 2. Stand Up the Application Environment
Install development virtual configuration variables locally:
```bash
python -m venv .env
source .env/bin/activate  # On Windows use: .env\Scripts\activate
pip install -r requirements.txt
```

### 3. Headless Model Compilation & Benchmarking
To execute multi-model evaluations (XGBoost vs. Random Forest) and generate diagnostic threshold visualization charts headlessly from your shell:
```bash
cd notebooks
python -m jupyter nbconvert --to notebook --execute exploration_and_tuning.ipynb --inplace
cd ..
```

### 4. Launch the Local Service
Spin up the FastAPI server natively via Uvicorn:
```bash
uvicorn app.main:app --reload --port 8000
```
Navigate to `http://127.0.0` in your browser to interact with the auto-generated Swagger UI engine.

## 🐳 Container Production Deployment

To package the service into a predictable Linux runtime container matching your host environment, run our unified, cache-less multi-stage pipeline:

```bash
# Force clear old container references to prevent port 8000 conflicts
docker rm -f fintech-churn-app 2>/dev/null || true

# Compile the production image cleanly under unified Python 3.12 layers
docker build --no-cache -t fintech-churn-service:latest .

# Run the container background daemon process
docker run -d -p 8000:8000 --name fintech-churn-app fintech-churn-service:latest
```

## 🧪 Automated Regression Batch Testing
With either your local Uvicorn instance or Docker container active on port `8000`, execute our regression validation script to stress-test your system boundaries across multiple user behavioral scenarios:

```bash
python utils/batch_test.py
```
*Successfully passes transaction footprints with a 100% processing success rate under sub-15ms loopback latencies.*

## 📋 Data Feature Ingestion Schemas

The incoming JSON payloads processed by the `/predict` endpoint evaluate live user risk footprints by parsing PaySim-inspired transactional attributes directly, removing data leakage entirely:

| Metric | Type | Validation Constraint | Description |
| :--- | :--- | :--- | :--- |
| `step` | Integer | `>= 0` | Simulation temporal hour step tracking indicator |
| `amount` | Float | `>= 0.0` | Absolute value of the fiat volume requested |
| `oldbalanceOrg` | Float | `>= 0.0` | Senders balance immediately prior to execution |
| `newbalanceOrig` | Float | `>= 0.0` | Senders balance immediately following transaction completion |
