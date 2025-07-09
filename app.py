# File: app.py

import streamlit as st
import pandas as pd
import joblib

# Load the trained pipeline from the file
try:
    pipeline = joblib.load('rf_pipeline.joblib')
except FileNotFoundError:
    st.error("Model file ('rf_pipeline.joblib') not found. Please run 'train_model.py' first.")
    st.stop()

# Set the page title and icon
st.set_page_config(page_title="Customer Churn Predictor", page_icon="🤖")

# Create the title for the app
st.title("🤖 Customer Churn Predictor")
st.write("This app predicts whether a customer is likely to churn. Please enter the customer's details in the sidebar.")

# --- SIDEBAR FOR USER INPUT ---
st.sidebar.header("Customer Details")

def user_input_features():
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    senior_citizen = st.sidebar.selectbox('Senior Citizen', ('No', 'Yes'))
    partner = st.sidebar.selectbox('Partner', ('Yes', 'No'))
    dependents = st.sidebar.selectbox('Dependents', ('No', 'Yes'))
    tenure = st.sidebar.slider('Tenure (months)', 1, 72, 24)
    phone_service = st.sidebar.selectbox('Phone Service', ('No', 'Yes'))
    multiple_lines = st.sidebar.selectbox('Multiple Lines', ('No phone service', 'No', 'Yes'))
    internet_service = st.sidebar.selectbox('Internet Service', ('DSL', 'Fiber optic', 'No'))
    online_security = st.sidebar.selectbox('Online Security', ('No', 'Yes', 'No internet service'))
    online_backup = st.sidebar.selectbox('Online Backup', ('Yes', 'No', 'No internet service'))
    device_protection = st.sidebar.selectbox('Device Protection', ('No', 'Yes', 'No internet service'))
    tech_support = st.sidebar.selectbox('Tech Support', ('No', 'Yes', 'No internet service'))
    streaming_tv = st.sidebar.selectbox('Streaming TV', ('No', 'Yes', 'No internet service'))
    streaming_movies = st.sidebar.selectbox('Streaming Movies', ('No', 'Yes', 'No internet service'))
    contract = st.sidebar.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
    paperless_billing = st.sidebar.selectbox('Paperless Billing', ('Yes', 'No'))
    payment_method = st.sidebar.selectbox('Payment Method', ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))
    monthly_charges = st.sidebar.slider('Monthly Charges ($)', 18.0, 120.0, 70.0, 0.01)
    total_charges = st.sidebar.slider('Total Charges ($)', 18.0, 9000.0, 1400.0, 0.01)

    data = {
        'gender': gender,
        'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display the user input
st.subheader("Customer Input Details:")
st.write(input_df)

# --- PREDICTION ---
if st.button("Predict Churn"):
    prediction = pipeline.predict(input_df)
    prediction_proba = pipeline.predict_proba(input_df)

    st.subheader("Prediction Result")
    
    if prediction[0] == 1:
        st.error(f"This customer is LIKELY to churn with a probability of {prediction_proba[0][1]*100:.2f}%.")
    else:
        st.success(f"This customer is LIKELY to stay with a probability of {prediction_proba[0][0]*100:.2f}%.")