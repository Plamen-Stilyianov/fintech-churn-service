# COFINFAD Fintech Churn Prediction Service

An enterprise-grade, high-availability machine learning microservice container engineered to score real-time customer attrition risk in Latin American financial markets. The system isolates high-signal consumer features from the **COFINFAD (Colombian Fintech Financial Analytics Dataset)** research data store and delivers low-latency risk scores via an optimized gradient-boosted XGBoost tree classifier.

---

## 🏗️ System Architecture & Repository Layout

The service isolates processing, training, and deployment boundaries to ensure strict host parity and support clean containerized workflows:

```text
fintech-churn-service/
├── app/                  # FastAPI Application Layer
│   ├── artifacts/        # Serialized Production Tree Model Binary
│   │   └── churn_model.pkl   # 0.99 F1-Score XGBoost parameters
│   ├── __init__.py
│   ├── main.py           # REST Gateway with defensive matrix unboxing
│   └── schemas.py        # Strict Pydantic v2 entry/exit data validation contracts
├── notebooks/            # Research & Model Compilation Layer
│   ├── fintech_transactions.csv  # Authentic COFINFAD customer_data.csv matrix
│   └── train.py          # Standalone training, median balancing, and export suite
├── utils/                # Integration Verification Layer
│   └── batch_test.py     # Automated sub-15ms loopback regression stress tests
├── Dockerfile            # Multi-stage, cache-less Python 3.12-slim container
├── docker-compose.yml    # Unified multi-service infrastructure blueprint
└── requirements.txt      # Production environment package lock manifest
```

---

## 🔬 Dataset Context & Collection Methodology

The underlying analytics engine relies on population-level observational metrics from **48,723 unique digital wallet customers** of a Colombian fintech company, tracking **3,159,157 individual transaction events** over a 12-month window. 
*   **Data Provenance:** Documented and published at [Mendeley Data (DOI: 10.17632/mhb4zn3258.1)](https://mendeley.com).
*   **Operational Extraction:** Python-based scripts extracted 42 raw transaction-level variables during off-peak hours. Quarterly in-app customer surveys captured user sentiment scores on a rigorous 6-point scale.
*   **Ethical Considerations & Privacy:** Anonymized in strict compliance with Colombian data protection regulations (**Ley 1581 de 2012**). It utilizes non-reversible unique codes, geographic aggregation, and privacy-preserving transformations.

---

## 🧠 Modeling & Class Imbalance Optimizations

During baseline parsing of the continuous historical `churn_probability` field, a severe data skew was isolated. Splitting target features on rigid, arbitrary bounds caused models to train on a functionally empty minority class. To guarantee stable decision boundaries across production inference loops, the `notebooks/train.py` pipeline deploys three core optimizations:

1.  **Median-Based Class Splitting**: Continuous raw churn metrics are split dynamically against the dataset's calculated statistical median (**0.3540**). This balances target class partitions evenly (~4,875 positive vs. 4,870 negative evaluation records) and prevents node dropouts.
2.  **Gradient Penalty Scaling**: Tree splits are balanced via XGBoost's `scale_pos_weight` ratio and Random Forest's balanced class weight parameters to heavily penalize minority group classification errors.
3.  **Calibrated Decision Boundary**: The operational inference threshold is set to a balanced **`0.50`** to match the training target median.

### Multi-Model Performance Comparison (Decision Threshold = 0.50)

*   **XGBoost Classifier (Production Weight Base)**:
    *   **Precision**: 0.99 | **Recall**: 0.99 | **F1-Score**: 0.99
*   **Random Forest Classifier**:
    *   **Precision**: 0.96 | **Recall**: 0.94 | **F1-Score**: 0.95

---

## 📋 Data Feature Ingestion Contracts

All incoming JSON payloads processed by the `/predict` route must fulfill strict validation constraints before entering the tree inference gates. While the raw COFINFAD dataset spans 57 corporate columns, feature selection has isolated the top 6 highest-signal predictors to maintain an ultra-lightweight API surface:

| Field Name | JSON Type | Validation Rule | COFINFAD Variable Map & Definition |
| :--- | :--- | :--- | :--- |
| `age` | Integer | `ge=18`, `le=100` | Var No. 2: Customer's age in years |
| `income_bracket` | Integer | `ge=0`, `le=3` | Var No. 5: Encoded Income Category (0=Low, 1=Med, 2=High, 3=Very High) |
| `active_products` | Integer | `ge=0` | Var No. 17: Number of active internal financial products |
| `app_logins_frequency`| Integer | `ge=0` | Var No. 18: Number of mobile app logins per month |
| `tx_count` | Integer | `ge=0` | Var No. 25: Total number of transactions made by the customer |
| `satisfaction_score` | Integer | `ge=1`, `le=6` | Var No. 33: Overall customer satisfaction score (1-6 scale) |

---

## 🚀 Unified Execution & Deployment Routine

### 1. Execute Headless Model Training
To ingest the core file rows, map string category features, balance the target matrix, and export the binary model files natively on your host environment:
```bash
cd notebooks/
python train.py
cd ..
```

### 2. Stand Up Container Infrastructure (Docker Compose)
To clear old background conflicts, compile your multi-stage Python 3.12 image layers, mount your model artifacts, and boot up your production API background daemon:
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
