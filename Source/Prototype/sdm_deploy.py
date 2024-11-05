import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the saved model
model = joblib.load('C:\\Users\\Republic Of Gamers\\OneDrive\\Documents\\GitHub\\TSDN-BoyWithLuv\\Source\\Prototype\\rf_day_stayed.pkl')

# Initialize LabelEncoders for each categorical feature
label_encoders = {
    'Gender': LabelEncoder(),
    'Blood Type': LabelEncoder(),
    'Medical Condition': LabelEncoder(),
    'Insurance Provider': LabelEncoder(),
    'Admission Type': LabelEncoder(),
    'Medication': LabelEncoder(),
    'Test Results': LabelEncoder()
}

# Define options for each categorical input
gender_options = ["Male", "Female"]
medical_condition_options = ["Arthritis", "Diabetes", "Hypertension", "Obesity", "Cancer", "Asthma"]
blood_type_options = ["A", "B", "AB", "O"]
insurance_provider_options = ["Cigna", "Medicare", "UnitedHealthcare", "Blue Cross", "Aetna"]
admission_type_options = ["Emergency", "Urgent", "Elective"]
medication_options = ["Lipitor", "Ibuprofen", "Aspirin", "Paracetamol", "Penicillin"]
test_results_options = ["Abnormal", "Normal", "Inconclusive"]

# Title
st.title("Hospital Stay Duration Prediction")

# Input fields for each feature
age = st.number_input("Age", min_value=0, max_value=120)
gender = st.selectbox("Gender", gender_options)
blood_type = st.selectbox("Blood Type", blood_type_options)
medical_condition = st.selectbox("Medical Condition", medical_condition_options)
insurance_provider = st.selectbox("Insurance Provider", insurance_provider_options)
billing_amount = st.number_input("Billing Amount", min_value=1000.0, max_value=50000.00, format="%.2f")
admission_type = st.selectbox("Admission Type", admission_type_options)
medication = st.selectbox("Medication", medication_options)
test_results = st.selectbox("Test Results", test_results_options)

# Prepare the input for prediction
input_data = {
    "Age": age,
    "Gender": gender,
    "Blood Type": blood_type,
    "Medical Condition": medical_condition,
    "Insurance Provider": insurance_provider,
    "Billing Amount": billing_amount,
    "Admission Type": admission_type,
    "Medication": medication,
    "Test Results": test_results
}

# Convert input_data to a DataFrame
input_df = pd.DataFrame([input_data])

# Encode categorical variables using LabelEncoders
for col in label_encoders:
    input_df[col] = label_encoders[col].fit_transform(input_df[col])

# Ensure feature order matches model training
input_df = input_df[["Age", "Gender", "Blood Type", "Medical Condition", 
                     "Insurance Provider", "Billing Amount", "Admission Type", 
                     "Medication", "Test Results"]]

# Predict button
if st.button("Predict Days Stayed"):
    # Make a prediction
    prediction = model.predict(input_df)
    st.write(f"Predicted Days Stayed: {prediction[0]:.2f}")