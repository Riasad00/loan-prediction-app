import streamlit as st
import joblib
import numpy as np

pip install --upgrade pip

# Load trained model
model = joblib.load('loan_model.joblib')

st.title("🏦 Loan Approval Prediction App")

st.write("Enter your information below to check loan approval status:")

# Input fields
income = st.number_input("Applicant Income", min_value=0)
loan_amt = st.number_input("Loan Amount", min_value=0)
credit_hist = st.selectbox("Credit History", [1.0, 0.0])

if st.button("Predict"):
    features = np.array([[income, loan_amt, credit_hist]])
    prediction = model.predict(features)
    st.success("✅ Loan Approved!" if prediction[0] == 1 else "❌ Loan Rejected")
