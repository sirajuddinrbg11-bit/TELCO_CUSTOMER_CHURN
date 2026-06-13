import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- 1. Load Model, Scaler, and Label Encoder Mappings ---
try:
    # Load the best overall model (AdaBoost, as identified previously)
    with open('best_overall_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Handle case where model might be saved as a dictionary
    if isinstance(model, dict):
        model = model.get('model', model)
        if isinstance(model, dict):
            st.error("Model file contains a dictionary but no 'model' key found. Please re-save the model correctly.")
            st.stop()
    
    # Load the StandardScaler
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    # Load the LabelEncoder mappings
    with open('label_encoders_maps.pkl', 'rb') as f:
        label_encoders_maps = pickle.load(f)
except FileNotFoundError:
    st.error("Model, scaler, or label encoder mappings not found. Please ensure 'best_overall_model.pkl', 'scaler.pkl', and 'label_encoders_maps.pkl' are in the same directory.")
    st.stop()

# --- 2. Preprocessing Function ---
def preprocess_input(input_df):
    processed_df = input_df.copy()

    # Convert TotalCharges to numeric, handling potential errors
    processed_df['TotalCharges'] = pd.to_numeric(processed_df['TotalCharges'], errors='coerce')
    # For a single input, if TotalCharges becomes NaN, impute with 0. 
    processed_df['TotalCharges'].fillna(0, inplace=True)

    # Apply label encoding using the loaded mappings
    for col, mapping in label_encoders_maps.items():
        if col in processed_df.columns and col != 'Churn': # 'Churn' is the target, not an input feature here
            # Skip SeniorCitizen as it's already numeric (0/1)
            if col == 'SeniorCitizen':
                continue
            # Map input values to their encoded integers. If an unseen value appears, map to -1.
            processed_df[col] = processed_df[col].apply(lambda x: mapping.get(x, -1))

    # Define feature order to match training data (X.columns).
    # This order was observed from the 'X' variable in the kernel state.
    feature_order = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges'
    ]
    
    # Ensure the input dataframe has all expected columns and in the correct order
    for col in feature_order:
        if col not in processed_df.columns:
            # This should ideally not happen if all UI inputs are handled correctly
            processed_df[col] = 0 # Add missing columns with a default value (e.g., 0)

    processed_df = processed_df[feature_order]

    # Scale numerical features using the pre-fitted scaler
    # Ensure we're passing a 2D array
    scaled_data = scaler.transform(processed_df)
    
    return scaled_data

# --- 3. Streamlit UI ---
st.set_page_config(page_title="Telco Churn Predictor", layout="centered")
st.title('Telco Customer Churn Prediction')
st.markdown("Enter customer details below to predict if they will churn.")

# Input widgets within a form
with st.form("churn_prediction_form"):
    st.header("Customer Information")

    col1, col2 = st.columns(2)

    with col1:
        gender_options = list(label_encoders_maps['gender'].keys())
        gender = st.selectbox('Gender', gender_options, key='gender_input')

        senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'], key='senior_citizen_input')

        partner_options = list(label_encoders_maps['Partner'].keys())
        partner = st.selectbox('Partner', partner_options, key='partner_input')

        dependents_options = list(label_encoders_maps['Dependents'].keys())
        dependents = st.selectbox('Dependents', dependents_options, key='dependents_input')

        tenure = st.slider('Tenure (months)', 0, 72, 12, key='tenure_input')

        phone_service_options = list(label_encoders_maps['PhoneService'].keys())
        phone_service = st.selectbox('Phone Service', phone_service_options, key='phone_service_input')

        multiple_lines_options = list(label_encoders_maps['MultipleLines'].keys())
        multiple_lines = st.selectbox('Multiple Lines', multiple_lines_options, key='multiple_lines_input')

        internet_service_options = list(label_encoders_maps['InternetService'].keys())
        internet_service = st.selectbox('Internet Service', internet_service_options, key='internet_service_input')

    with col2:
        online_security_options = list(label_encoders_maps['OnlineSecurity'].keys())
        online_security = st.selectbox('Online Security', online_security_options, key='online_security_input')

        online_backup_options = list(label_encoders_maps['OnlineBackup'].keys())
        online_backup = st.selectbox('Online Backup', online_backup_options, key='online_backup_input')

        device_protection_options = list(label_encoders_maps['DeviceProtection'].keys())
        device_protection = st.selectbox('Device Protection', device_protection_options, key='device_protection_input')

        tech_support_options = list(label_encoders_maps['TechSupport'].keys())
        tech_support = st.selectbox('Tech Support', tech_support_options, key='tech_support_input')

        streaming_tv_options = list(label_encoders_maps['StreamingTV'].keys())
        streaming_tv = st.selectbox('Streaming TV', streaming_tv_options, key='streaming_tv_input')

        streaming_movies_options = list(label_encoders_maps['StreamingMovies'].keys())
        streaming_movies = st.selectbox('Streaming Movies', streaming_movies_options, key='streaming_movies_input')

        contract_options = list(label_encoders_maps['Contract'].keys())
        contract = st.selectbox('Contract', contract_options, key='contract_input')

        paperless_billing_options = list(label_encoders_maps['PaperlessBilling'].keys())
        paperless_billing = st.selectbox('Paperless Billing', paperless_billing_options, key='paperless_billing_input')

        payment_method_options = list(label_encoders_maps['PaymentMethod'].keys())
        payment_method = st.selectbox('Payment Method', payment_method_options, key='payment_method_input')

        monthly_charges = st.number_input('Monthly Charges', min_value=0.0, value=50.0, step=0.1, key='monthly_charges_input')
        total_charges = st.number_input('Total Charges', min_value=0.0, value=100.0, step=0.1, key='total_charges_input')

    submitted = st.form_submit_button("Predict Churn")

    if submitted:
        try:
            # Create a DataFrame from inputs
            input_data = pd.DataFrame({
                'gender': [gender],
                'SeniorCitizen': [1 if senior_citizen == 'Yes' else 0], # Convert 'Yes'/'No' to 1/0
                'Partner': [partner],
                'Dependents': [dependents],
                'tenure': [tenure],
                'PhoneService': [phone_service],
                'MultipleLines': [multiple_lines],
                'InternetService': [internet_service],
                'OnlineSecurity': [online_security],
                'OnlineBackup': [online_backup],
                'DeviceProtection': [device_protection],
                'TechSupport': [tech_support],
                'StreamingTV': [streaming_tv],
                'StreamingMovies': [streaming_movies],
                'Contract': [contract],
                'PaperlessBilling': [paperless_billing],
                'PaymentMethod': [payment_method],
                'MonthlyCharges': [monthly_charges],
                'TotalCharges': [total_charges]
            })

            # Preprocess input data
            processed_input = preprocess_input(input_data)

            # Make prediction
            prediction_encoded = model.predict(processed_input)
            prediction_proba = model.predict_proba(processed_input)

            # Decode the prediction back to 'Yes' or 'No'
            # Find the original labels for 'Churn' based on the stored mapping
            churn_mapping = {v: k for k, v in label_encoders_maps['Churn'].items()}
            churn_result = churn_mapping.get(prediction_encoded[0], 'Unknown')

            st.subheader(f"Prediction: Customer will **{'churn' if churn_result == 'Yes' else 'not churn'}**.")
            st.write(f"Probability of Not Churning: {prediction_proba[0][label_encoders_maps['Churn']['No']]:.2f}")
            st.write(f"Probability of Churning: {prediction_proba[0][label_encoders_maps['Churn']['Yes']]:.2f}")
        
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
