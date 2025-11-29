"""
Simple Fraud Detection Frontend
"""

import streamlit as st
import requests

# Page config
st.set_page_config(page_title="Fraud Detection", page_icon="💳")

st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details to check for fraud")

# API URL
API_URL = "http://localhost:8080"

# Check API health
try:
    health = requests.get(f"{API_URL}/health").json()
    if health.get("model_loaded"):
        st.success("✅ API Connected")
    else:
        st.error("❌ Model not loaded")
except:
    st.error("❌ Cannot connect to API")

st.markdown("---")

# Input form
st.subheader("Transaction Details")

col1, col2 = st.columns(2)

with col1:
    time = st.number_input("Time", value=0.0)
    amount = st.number_input("Amount ($)", value=100.0)

with col2:
    st.write("**PCA Features (V1-V28)**")
    st.caption("Anonymized features")

# Create expandable section for V features
with st.expander("Enter V1-V28 values (click to expand)"):
    v_cols = st.columns(4)
    v_features = {}
    
    for i in range(1, 29):
        col_idx = (i - 1) % 4
        with v_cols[col_idx]:
            v_features[f"V{i}"] = st.number_input(f"V{i}", value=0.0, format="%.6f", key=f"v{i}")

# Predict button
if st.button("🔍 Check for Fraud", type="primary"):
    # Prepare data
    transaction = {
        "Time": time,
        "Amount": amount,
        **v_features
    }
    
    # Call API
    try:
        with st.spinner("Analyzing transaction..."):
            response = requests.post(f"{API_URL}/predict", json=transaction)
            result = response.json()
        
        # Display results
        st.markdown("---")
        st.subheader("Analysis Result")
        
        # Show probability
        prob = result["probability"]
        st.metric("Fraud Probability", f"{prob*100:.4f}%")
        
        # Show verdict
        if result["is_fraud"]:
            st.error("⚠️ **FRAUD DETECTED**")
            st.warning("This transaction has been flagged as potentially fraudulent.")
        else:
            st.success("✅ **LEGITIMATE TRANSACTION**")
            st.info("This transaction appears to be legitimate.")
        
        # Show details
        with st.expander("Technical Details"):
            st.json(result)
    
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Sample data button
st.markdown("---")
if st.button("Load Sample Transaction"):
    st.info("Click 'Check for Fraud' to analyze the sample")