import time
import requests

# Point directly to your active local FastAPI network address
API_URL = "http://127.0.0.1:8000/predict"

# Construct a matrix list representing distinct real-world customer behaviors
BATCH_PAYLOADS = [
    {
        "label": "Scenario 1: Stable Active User (Low Value Transaction)",
        "data": {"step": 12, "amount": 150.00, "oldbalanceOrg": 5400.00, "newbalanceOrig": 5250.00}
    },
    {
        "label": "Scenario 2: Neutral User (Standard Outbound Transfer)",
        "data": {"step": 45, "amount": 3400.00, "oldbalanceOrg": 12500.00, "newbalanceOrig": 9100.00}
    },
    {
        "label": "Scenario 3: CRITICAL TRIGGER - Total Balance Drain (Extreme Churn Risk)",
        "data": {"step": 82, "amount": 95000.00, "oldbalanceOrg": 95000.00, "newbalanceOrig": 0.00}
    },
    {
        "label": "Scenario 4: High Temporal Delay Inactivity Pattern",
        "data": {"step": 612, "amount": 12000.00, "oldbalanceOrg": 45000.00, "newbalanceOrig": 33000.00}
    }
]


def execute_automated_stress_test():
    print("⚡ Starting Automated Fintech Churn Service Batch Evaluation...\n")
    headers = {"Content-Type": "application/json"}

    success_count = 0
    for i, item in enumerate(BATCH_PAYLOADS, 1):
        print(f"🔄 Processing [{i}/{len(BATCH_PAYLOADS)}] | {item['label']}")
        start_time = time.time()

        try:
            # Dispatch the payload synchronously over HTTP POST
            response = requests.post(API_URL, json=item['data'], headers=headers, timeout=5)
            latency = (time.time() - start_time) * 1000

            if response.status_code == 200:
                result = response.json()
                print(f"   🟢 HTTP 200 OK | Latency: {latency:.2f}ms")
                print(f"   📈 Churn Probability Score: {result['churn_probability']:.4f}")
                print(f"   🚨 High Risk Active Flag: {result['high_risk_flag']}")
                success_count += 1
            else:
                print(f"   🔴 HTTP Failure {response.status_code}: {response.text}")

        except Exception as e:
            print(f"   💥 Connection Error: {str(e)}")
        print("-" * 70)

    print(f"\n✅ Stress Test Completed. Success Rate: {success_count}/{len(BATCH_PAYLOADS)} requests processed.")


if __name__ == "__main__":
    # Ensure requests library is installed locally prior to execution
    try:
        execute_automated_stress_test()
    except ImportError:
        print("📦 Library 'requests' missing. Execute: pip install requests")
