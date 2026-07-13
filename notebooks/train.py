import os
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def run_model_training_pipeline():
    print("🔬 COFINFAD Production Engine: Balanced Multi-Model Estimation Loop")
    print("----------------------------------------------------------------------")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, 'fintech_transactions.csv')

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Production source CSV data block missing at: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f'📊 Raw Ingest Target Confirmed. Matrix Record Count: {len(df)}')

    # Clean and map corporate categorical income string fields to uniform numeric keys
    income_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Very High': 3}
    if df['income_bracket'].dtype == 'object':
        df['income_bracket'] = df['income_bracket'].map(income_map).fillna(1).astype(int)

    features = ['age', 'income_bracket', 'active_products', 'app_logins_frequency', 'tx_count', 'satisfaction_score']
    X = df[features]

    # 🔥 FIX 1: Use median-based splitting to correct severe class imbalance issues
    median_prob = df['churn_probability'].median()
    print(f"📊 Dataset Churn Probability Median calculated at: {median_prob:.4f}")
    y = (df['churn_probability'] >= median_prob).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f'✅ Balanced Split Implemented - Train Shape: {X_train.shape}, Test Shape: {X_test.shape}')
    print(f'⚖️ Target Class Distribution in Test Set: \n{y_test.value_counts()}')

    # 🔥 FIX 2: Inject explicit scale weight balancers to stabilize node split partitions
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_weight_ratio = neg_count / max(1, pos_count)

    models = {
        'XGBoost': XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=scale_weight_ratio,
            random_state=42,
            eval_metric='logloss'
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            class_weight='balanced'
        )
    }

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:, 1]

        # 🔥 FIX 3: Lower inference classification threshold down to a balanced 0.50
        preds = (probs >= 0.50).astype(int)
        print(f'\n=== {name} Corrected Performance Profile (Threshold = 0.50) ===')
        print(classification_report(y_test, preds, zero_division=0))

    # Export production artifact
    artifact_dir = os.path.abspath(os.path.join(current_dir, '../app/artifacts'))
    os.makedirs(artifact_dir, exist_ok=True)
    artifact_file_path = os.path.join(artifact_dir, 'churn_model.pkl')

    with open(artifact_file_path, 'wb') as f:
        pickle.dump(models['XGBoost'], f)
    print(f'\n✅ Success! Optimized production tree model weight binaries exported cleanly to: {artifact_file_path}')


if __name__ == "__main__":
    run_model_training_pipeline()
