import streamlit as st
import numpy as np
from joblib import load

# Load trained model
model = load("fraud_model.pkl")

# Page settings
st.set_page_config(page_title="UPI Fraud Detector", layout="centered")

st.title("💳 Real-Time UPI Fraud Detection System")
st.markdown("### AI-Powered Fraud Detection Using Machine Learning")
st.divider()

# User Inputs
time = st.number_input("Transaction Time", min_value=0.0)
amount = st.number_input("Transaction Amount (₹)", min_value=0.0)

# Prediction Button
if st.button("Analyze Transaction"):

    # Create 30 feature input (all zeros)
    input_data = np.zeros((1, 30))

    # Set Time (index 0)
    input_data[0][0] = time

    # Set Amount (index 29)
    input_data[0][29] = amount

    # Get fraud probability
    prediction_proba = model.predict_proba(input_data)[0][1]

    st.write("Fraud Probability:", round(prediction_proba, 4))

    if prediction_proba > 0.30:
        st.error("⚠ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Transaction is Safe")
