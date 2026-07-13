# Frozen operational configuration
import json
import os

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🔬 COFINFAD Production Engine: Multi-Model Estimation Loop\n",
    "This notebook reads raw financial log vectors, drops metadata noise, encodes enterprise strings, and fits highly tuned tree architectures headlessly."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import pickle\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from xgboost import XGBClassifier\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import classification_report, roc_curve, auc"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "source": [
    "csv_path = 'fintech_transactions.csv'\n",
    "if not os.path.exists(csv_path):\n",
    "    raise FileNotFoundError('Production source CSV data block missing from workspace path context!')\n",
    "\n",
    "df = pd.read_csv(csv_path)\n",
    "print(f'📊 Raw Ingest Target Confirmed. Matrix Record Count: {len(df)}')\n",
    "\n",
    "# Clean and map corporate categorical income string fields to uniform numeric keys\n",
    "income_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Very High': 3}\n",
    "if df['income_bracket'].dtype == 'object':\n",
    "    df['income_bracket'] = df['income_bracket'].map(income_map).fillna(1).astype(int)\n",
    "\n",
    "# Isolate our selected 6 core predictive behavioral features to preserve minimum payload sizing\n",
    "features = ['age', 'income_bracket', 'active_products', 'app_logins_frequency', 'tx_count', 'satisfaction_score']\n",
    "X = df[features]\n",
    "\n",
    "# Binarize the paper's continuous raw churn probability metric vector to define classification targets\n",
    "y = (df['churn_probability'] >= 0.50).astype(int)\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n",
    "print(f'✅ Supervised Data Boundaries Sealed - Matrix Dimensional Tuning Base: {X_train.shape}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "source": [
    "models = {\n",
    "    'XGBoost': XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric='logloss'),\n",
    "    'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')\n",
    "}\n",
    "\n",
    "results = {} \n",
    "for name, clf in models.items():\n",
    "    clf.fit(X_train, y_train)\n",
    "    probs = clf.predict_proba(X_test)[:, 1]\n",
    "    results[name] = probs\n",
    "    \n",
    "    # Standard evaluation output score profiles\n",
    "    preds = (probs >= 0.70).astype(int)\n",
    "    print(f'\\n=== {name} Real Performance Profile (Threshold = 0.70) ===')\n",
    "    print(classification_report(y_test, preds, zero_division=0))\n",
    "\n",
    "os.makedirs('../app/artifacts', exist_ok=True)\n",
    "with open('../app/artifacts/churn_model.pkl', 'wb') as f:\n",
    "    pickle.dump(models['XGBoost'], f)\n",
    "print('\\n✅ Success! Optimized production tree model weight binaries exported cleanly.')"
   ]
  }
 ],
 "metadata": {
  "language_info": {"name": "python"}
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open("setup_multi_model_notebook.py", "w", encoding="utf-8") as f:
    f.write("# Frozen operational configuration\n")

with open("../notebooks/exploration_and_tuning.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=1)
print("🚀 Target workspace layout successfully mapped to real dataset column features.")
