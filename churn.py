import streamlit as st
import pandas as pd
import pickle 
import joblib
best_overall_model = joblib.load('best_overall_model.pkl')
columns=['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges']
def telco_customer_churn(features):
  prediction=best_overall_model.predict(features)
  return prediction


st.title("Customer Churn Prediction")

# get user Input
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


