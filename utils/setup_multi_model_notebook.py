import json
import os

# Updated raw dictionary template containing multi-model evaluation and visual analytics cells
notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🔬 Fintech Churn Engine: Multi-Model Evaluation & Visualization\n",
    "This workbench ingests raw sequential transactional lines, tests alternative machine learning classifiers (XGBoost vs Random Forest), and visualizes threshold metrics."
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
    "from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "csv_path = 'fintech_transactions.csv'\n",
    "if not os.path.exists(csv_path):\n",
    "    raise FileNotFoundError('Dataset missing! Ensure download_dataset.py ran successfully.')\n",
    "\n",
    "df = pd.read_csv(csv_path)\n",
    "features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig']\n",
    "X = df[features]\n",
    "y = (((df['oldbalanceOrg'] - df['newbalanceOrig']) > 15000) | (df['step'] > 48)).astype(int)\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n",
    "print(f'Ingested shapes - Train: {X_train.shape}, Evaluation: {X_test.shape}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "models = {\n",
    "    'XGBoost': XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric='logloss'),\n",
    "    'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')\n",
    "}\n",
    "\n",
    "results = {}\n",
    "for name, clf in models.items():\n",
    "    clf.fit(X_train, y_train)\n",
    "    probs = clf.predict_proba(X_test)[:, 1]\n",
    "    results[name] = probs\n",
    "    \n",
    "    preds = (probs >= 0.70).astype(int)\n",
    "    print(f'\\n=== {name} Performance Profile (Threshold = 0.70) ===')\n",
    "    print(classification_report(y_test, preds, zero_division=0))\n",
    "\n",
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n",
    "for name, probs in results.items():\n",
    "    fpr, tpr, _ = roc_curve(y_test, probs)\n",
    "    ax1.plot(fpr, tpr, label=f'{name} (AUC = {auc(fpr, tpr):.3f})', lw=2)\n",
    "ax1.plot([0, 1], [0, 1], 'k--', lw=1.5)\n",
    "ax1.set_title('ROC-AUC Curve Comparison')\n",
    "ax1.set_xlabel('False Positive Rate')\n",
    "ax1.set_ylabel('True Positive Rate')\n",
    "ax1.legend()\n",
    "ax1.grid(True, linestyle=':')\n",
    "\n",
    "for name, probs in results.items():\n",
    "    precision, recall, _ = precision_recall_curve(y_test, probs)\n",
    "    ax2.plot(recall, precision, label=name, lw=2)\n",
    "ax2.set_title('Precision-Recall Curve Comparison')\n",
    "ax2.set_xlabel('Recall (Sensitivity)')\n",
    "ax2.set_ylabel('Precision')\n",
    "ax2.legend()\n",
    "ax2.grid(True, linestyle=':')\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "with open('../app/artifacts/churn_model.pkl', 'wb') as f:\n",
    "    pickle.dump(models['XGBoost'], f)\n",
    "print('\\n✅ Success! XGBoost exported to artifacts directory for FastAPI usage.')"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

# Ensure directory context is ready
if not os.path.exists('../notebooks'):
    os.makedirs("../notebooks", exist_ok=True)

# Overwrite exploration_and_tuning.ipynb with the advanced analytics structure
with open("../notebooks/exploration_and_tuning.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=1)

print("🚀 Successfully updated and generated exploration_and_tuning.ipynb with visual comparison cells!")
