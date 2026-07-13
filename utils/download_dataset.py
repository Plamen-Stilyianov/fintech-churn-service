import os
import io
import requests
import pandas as pd

# 🔒 Secure Read-Access Token to authenticate your Hugging Face Hub session
HF_ACCESS_TOKEN = "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"

# 🎯 THE SPECIFIC TARGET LINK: Pulls customer_data.csv directly from your dataset repository
RAW_DATA_URL = "https://huggingface.co"


def download_and_process_real_cofinfad():
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_script_dir, ".."))

    notebooks_dir = os.path.join(repo_root, "notebooks")
    os.makedirs(notebooks_dir, exist_ok=True)
    target_csv_path = os.path.join(notebooks_dir, "fintech_transactions.csv")

    print(f"🛰️ Step 1: Querying target repository for file: customer_data.csv...")
    try:
        session = requests.Session()
        session.headers.update({
            "Authorization": f"Bearer {HF_ACCESS_TOKEN}",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
        })

        print(f"   📥 Fetching asset from direct storage link: {RAW_DATA_URL}")
        response = session.get(RAW_DATA_URL, timeout=60)
        response.raise_for_status()

        # Guard rail check: Ensure it is a valid CSV data stream, not the HTML code of a landing page
        if "text/html" in response.headers.get("Content-Type", "") or response.text.strip().startswith(
                "<!doctype html>"):
            raise ValueError("Target repository returned a web interface landing page instead of a raw CSV stream.")

        df = pd.read_csv(io.StringIO(response.text))
        print(f"   📊 Target file loaded successfully! Total customer records found: {len(df)}")

        print("\n🧠 Step 2: Extracting high-signal features and building corporate schemas...")
        features = ['age', 'income_bracket', 'active_products', 'app_logins_frequency', 'tx_count',
                    'satisfaction_score']
        df['Exited'] = (df['churn_probability'] >= 0.50).astype(int)

        if df['income_bracket'].dtype == 'object':
            df['income_bracket'] = df['income_bracket'].map({'Low': 0, 'Medium': 1, 'High': 2, 'Very High': 3}).fillna(
                1)

        final_production_df = df[features + ['Exited']]
        final_production_df.to_csv(target_csv_path, index=False)
        print(f"✅ Success! Processed data file safely staged at: {target_csv_path}")
        print(f"📊 Actual Data Matrix Shape: {final_production_df.shape}")

    except Exception as pipeline_err:
        print(f"💥 Critical Ingestion Failure: {str(pipeline_err)}")
        print(
            "❌ Cannot proceed without authentic data. Ensure your SUSE machine can resolve deep Hugging Face network paths.")


if __name__ == "__main__":
    download_and_process_real_cofinfad()
