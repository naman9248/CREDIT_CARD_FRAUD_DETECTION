import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load model & scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.set_page_config(page_title="Fraud Detection", page_icon="🔍", layout="wide")

st.title("🔍 Credit Card Fraud Detection")
st.write("Enter transaction details below to check if it's fraudulent.")
st.divider()

# --- Amount Input ---
col1, col2 = st.columns(2)
with col1:
    amt = st.number_input("💰 Transaction Amount ($)", min_value=0.0, format="%.2f")
with col2:
    st.info(f"Fraud Threshold: 60% probability")

st.divider()

# --- V1-V28 Inputs ---
st.write("### PCA Features (V1 - V28)")
st.caption("These are anonymized transaction features. Default value is 0.0")

v_features = []
cols = st.columns(4)
for i in range(1, 29):
    with cols[(i-1) % 4]:
        v = st.number_input(f"V{i}", value=0.0, format="%.4f", key=f"v{i}")
        v_features.append(v)

st.divider()

# --- Predict Button ---
if st.button("🔍 Check Transaction", use_container_width=True, type="primary"):

    # Scale only Amount
    amt_scaled = scaler.transform(pd.DataFrame([[amt]], columns=['Amount']))[0][0]

    # Combine features: V1-V28 + scaled Amount
    features = np.array([v_features + [amt_scaled]])

    # Predict
    prob = model.predict_proba(features)[0][1]
    prediction = 1 if prob >= 0.6 else 0

    st.divider()
    if prediction == 1:
        st.error(f"🚨 FRAUDULENT TRANSACTION DETECTED!")
        st.metric("Fraud Probability", f"{round(prob*100, 2)}%")
    else:
        st.success(f"✅ LEGITIMATE TRANSACTION")
        st.metric("Fraud Probability", f"{round(prob*100, 2)}%")