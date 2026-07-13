import requests

API_URL = "http://127.0.0.1:8000/predict"

BATCH_PAYLOADS = [
    {
        "label": "Scenario 1: Highly Engaged Active Member",
        "data": {
            "age": 28,
            "income_bracket": 2,
            "active_products": 3,
            "app_logins_frequency": 45,
            "tx_count": 22,
            "satisfaction_score": 5
        }
    },
    {
        "label": "Scenario 2: CRITICAL ATTRITION RISK - Low App Engagements & Poor Review Score",
        "data": {
            "age": 54,
            "income_bracket": 0,
            "active_products": 1,
            "app_logins_frequency": 1,
            "tx_count": 0,
            "satisfaction_score": 1
        }
    }
]


def run_suite():
    print("⚡ Starting Core COFINFAD Financial Service Inactivity Stress Testing...\n")
    for item in BATCH_PAYLOADS:
        print(f"🔄 Evaluating Scenario: {item['label']}")
        try:
            response = requests.post(API_URL, json=item['data'], timeout=5)
            res = response.json()

            print(f"   🟢 HTTP {response.status_code} OK")
            print(f"   📊 Raw Container Response JSON: {res}")

            # 🧠 Defensive Fallback Parsing: Extract features dynamically matching any common key standard
            score = res.get('churn_probability') if res.get('churn_probability') is not None else (
                        res.get('probability') or res.get('churn_score') or 0.0)
            flag = res.get('high_risk_flag') if res.get('high_risk_flag') is not None else (
                        res.get('risk_flag') or False)

            print(f"   📈 Inferred Score: {score} | 🚨 High Risk Flag: {flag}\n" + "-" * 65)
        except Exception as e:
            print(f"   🔴 Parsing Failure Block: {str(e)}\n" + "-" * 65)


if __name__ == "__main__":
    run_suite()
