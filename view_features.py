import joblib

features = joblib.load("models/feature_names.pkl")
print("🧾 Model Feature Names:")
for f in features:
    print("-", f)
