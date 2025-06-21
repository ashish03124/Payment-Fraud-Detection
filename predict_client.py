import requests
import json

# API endpoint
url = "http://127.0.0.1:5000/predict"

# Input features (based on your model's feature_names.pkl)
payload = {
    "amount": 50000,
    "type_DEPOSIT": 0,
    "type_PAYMENT": 0,
    "type_TRANSFER": 1,
    "type_WITHDRAWAL": 0,
    "balanceErrorOrig": 0,
    "balanceErrorDest": 0,
    "hour": 14,
    "is_night": 0,
    "amount_log": 10.82,
    "amount_zscore": 1.2,
    "is_high_amount": 1,
    "total_errors": 0
}

# Send POST request
response = requests.post(url, json=payload)

# Print response
if response.status_code == 200:
    result = response.json()
    print("✅ Prediction result:")
    print(f"  - Fraud Probability: {result['fraud_probability']:.4f}")
    print(f"  - Is Fraud: {'Yes' if result['is_fraud'] == 1 else 'No'}")
else:
    print(f"❌ Request failed with status code: {response.status_code}")
    print(response.text)
