# COFINFAD Fintech Churn Prediction Service

An enterprise-grade, high-availability machine learning microservice container designed to calculate real-time customer attrition risk. The system processes high-signal consumer behavioral profiles from the population-level **COFINFAD (Colombian Fintech Financial Analytics Dataset)** framework and delivers low-latency risk scores via a mathematically calibrated, optimized XGBoost tree classifier.

---

## 🏗️ System Architecture & Repository Layout

The service isolates processing, training, and deployment boundaries to enforce strict environment parity, completely removing browser runtime dependencies for automated tasks:

```text
fintech-churn-service/
│
├── app/
│   ├── artifacts/
│   │   └── churn_model.pkl       # Serialized 0.99 F1-Score XGBoost binary weights
│   ├── __init__.py
│   ├── main.py                   # FastAPI application engine with native NumPy unboxing
│   └── schemas.py                # Strict Pydantic v2 data validation validation classes
│
├── notebooks/
│   ├── fintech_transactions.csv  # Authentic COFINFAD historical aggregated features
│   └── train.py                  # Standalone multi-model tree optimization & class balancer
│
├── utils/
│   └── batch_test.py             # Client regression verification & latency stress testing
│
├── Dockerfile                    # Lightweight multi-stage Python 3.12-slim runtime container
├── docker-compose.yml            # Unified multi-service container orchestration blueprint
└── requirements.txt              # Production environments package lock manifest
```

---

## 🛠️ Tech Stack & Advanced Engineering Features

- **Inference Gateway Framework:** FastAPI, Uvicorn, Pydantic v2 (Rust-backed type safety and sub-3ms routing validation).
- **Machine Learning Core:** XGBoost, Scikit-Learn, Pandas, NumPy (Tabular gradient-boosted decision trees).
- **DevOps & Portability:** Multi-Stage Docker, Docker Compose, Python 3.12-slim (Total host/runner environment parity).
- **High Availability Pattern:** Resilience fallback rule-engine built inside `app/main.py` that gracefully catches artifact defects or permission shifts to serve domain-driven heuristics without interrupting live production traffic.

---

## 🔬 Core Modeling & Class Imbalance Optimizations

During baseline parsing of the continuous historical `churn_probability` field, a severe data skew was isolated. Splitting features on rigid bounds caused models to train on an empty minority group. To guarantee stable decision boundaries across production inference loops, the `notebooks/train.py` pipeline deploys three core optimizations:

1. **Median-Based Class Splitting**: Continuous raw churn metrics are split dynamically against the dataset's calculated statistical median (**0.3540**). This balances target class partitions evenly (~4,870 rows per evaluation category) and stops minority sample dropout.
2. **Gradient Penalty Scaling**: Tree splits are balanced via XGBoost's `scale_pos_weight` ratio and Random Forest's balanced class weight parameters to heavily penalize minority group classification errors.
3. **Calibrated Decision Boundary**: The operational inference threshold is set to a balanced **`0.50`** to match the training target median.

### Model Performance Profiles (Decision Threshold = 0.50)

* **XGBoost Classifier (Production Weight Base)**:
  * **Precision**: 0.99 | **Recall**: 0.99 | **F1-Score**: 0.99
* **Random Forest Classifier**:
  * **Precision**: 0.96 | **Recall**: 0.94 | **F1-Score**: 0.95

---

## 📋 Data Feature Ingestion Contracts

All incoming JSON payloads processed by the `/predict` endpoint route must fulfill strict validation constraints before entering the tree inference gates.

| Variable Name | JSON Data Type | Validation Boundary | Core Definition |
| :--- | :--- | :--- | :--- |
| `age` | Integer | `18` to `100` | Customer age in years |
| `income_bracket` | Integer | `0` to `3` | Encoded income tier (0=Low, 1=Medium, 2=High, 3=Very High) |
| `active_products` | Integer | `>= 0` | Absolute count of active internal financial items adopted |
| `app_logins_frequency` | Integer | `>= 0` | Total monthly mobile application login sessions |
| `tx_count` | Integer | `>= 0` | Absolute count of monthly transactions completed |
| `satisfaction_score` | Integer | `1` to `6` | Overall customer satisfaction survey score (1-6 scale) |

---

## 🚀 Unified Execution & Deployment Routine

### 1. Ingest Features & Compile Model Weights
To run headless model training, map category indexes, fit tree thresholds, and generate your production model binary file natively on your host shell:
```bash
cd notebooks/
python train.py
cd ..
```

### 2. Stand Up Container Infrastructure (Docker Compose)
To clear old background conflicts, compile multi-stage image layers, mount your model artifacts, and boot up your production API background daemon:
```bash
# Bring down the active stack and erase stagnant cached volumes
docker-compose down --volumes 2>/dev/null || true

# Force a clean, no-cache rebuild and launch the background containers
docker-compose up -d --build --force-recreate
```

---

## 🧪 Automated Service Verification

While your microservice container runs securely inside its Docker environment, run the integration test script to stress-test your backend interfaces across multiple customer engagement scenarios:

```bash
python utils/batch_test.py
```

### Expected Green Dashboard Output
```text
⚡ Starting Core COFINFAD Financial Service Inactivity Stress Testing...

🔄 Evaluating Scenario: Scenario 1: Highly Engaged Active Member
   🟢 HTTP 200 OK
   📊 Raw Container Response JSON: {'churn_probability': 0.0036, 'high_risk_flag': False}
   📈 Inferred Score: 0.0036 | 🚨 High Risk Flag: False
-----------------------------------------------------------------
🔄 Evaluating Scenario: Scenario 2: CRITICAL ATTRITION RISK - Low App Engagements & Poor Review Score
   🟢 HTTP 200 OK
   📊 Raw Container Response JSON: {'churn_probability': 0.9919, 'high_risk_flag': True}
   📈 Inferred Score: 0.9919 | 🚨 High Risk Flag: True
-----------------------------------------------------------------
```
