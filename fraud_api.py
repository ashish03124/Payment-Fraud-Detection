from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model & scaler
model = joblib.load("models/fraud_detection_xgb.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/feature_names.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    X = np.array([data[fn] for fn in feature_names]).reshape(1, -1)
    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[0][1]
    prediction = int(prob >= 0.5)
    return jsonify({'fraud_probability': float(prob), 'is_fraud': prediction})

if __name__ == '__main__':
    app.run(debug=True)
