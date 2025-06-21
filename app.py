import streamlit as st
import joblib
import numpy as np

model = joblib.load("models/fraud_detection_xgb.pkl")
scaler = joblib.load("models/scaler.pkl")
features = joblib.load("models/feature_names.pkl")

st.title("Fraud Detection Demo")

inputs = []
for feat in features:
    val = st.number_input(f"{feat}", value=0.0)
    inputs.append(val)

if st.button("Predict"):
    x = np.array(inputs).reshape(1, -1)
    x_scaled = scaler.transform(x)
    prob = model.predict_proba(x_scaled)[0][1]
    st.write(f"Fraud Probability: {prob:.2f}")
    st.success("🚨 Fraud Detected!") if prob > 0.5 else st.info("✅ Not Fraud")
