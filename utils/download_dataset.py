# download_dataset.py
import os
import subprocess
import sys


def install_and_fetch():
    print("📦 Step 1: Ensuring Hugging Face 'datasets' package is installed...")
    # Safely install the dataset streaming tool inside PyCharm's active venv
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])

    from datasets import load_dataset
    import pandas as pd

    print("🛰️ Step 2: Streaming transactional data logs...")
    # Load the mini transactional dataset stream
    dataset = load_dataset("CiferAI/cifer-fraud-detection-mini-dataset", split="train")
    df = pd.DataFrame(dataset)

    # Save directly into notebooks folder for local Jupyter development
    target_path = os.path.join("../notebooks", "fintech_transactions.csv")
    df.to_csv(target_path, index=False)

    print(f"\n✅ Success! Data safely staged at: {target_path}")
    print(f"📊 Rows ingested: {len(df)} | Columns: {list(df.columns)}")


if __name__ == "__main__":
    install_and_fetch()
