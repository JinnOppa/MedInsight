import streamlit as st
import pandas as pd
import numpy as np
import joblib  # For loading the trained model
from sklearn.preprocessing import LabelEncoder

# Load your pre-trained model (assuming the model is saved in a .pkl file)
model = joblib.load('C:\\Users\\Republic Of Gamers\\OneDrive\\Documents\\GitHub\\TSDN-BoyWithLuv\\Source\\Prototype\\surgery_risk\\xgb_surgery_risk.pkl')

# Function to encode categorical columns
def encode_categorical_data(input_data):
    # Define LabelEncoder for each categorical column
    label_encoders = {
        'gender': LabelEncoder(),
        'age_group': LabelEncoder(),
        'smoking_status': LabelEncoder(),
        'e_cigarette_usage': LabelEncoder(),
        'alcohol_consumption_rate': LabelEncoder(),
        'surgery_name': LabelEncoder(),
        'surgery_type': LabelEncoder(),
        'surgical_specialty': LabelEncoder(),
        'anesthesia_type': LabelEncoder(),
        'blood_loss_category': LabelEncoder(),
        'blood_transfusions': LabelEncoder(),
        'stay_duration': LabelEncoder(),
        'room_type': LabelEncoder()
    }

    # Fit and transform each categorical column
    for column, le in label_encoders.items():
        if column in input_data.columns:
            input_data[column] = le.fit_transform(input_data[column])

    return input_data

# Function to make predictions
def predict_preoperative_risk(gender, age_group, smoking_status, e_cigarette_usage, alcohol_consumption_rate, 
                              surgery_name, surgery_type, surgical_specialty, anesthesia_type, surgery_duration, 
                              blood_loss_category, blood_transfusions, stay_duration, room_type, pain_score, 
                              rehab_assessment_score):
    # Prepare the data in the format the model expects
    input_data = pd.DataFrame({
        'gender': [gender],
        'age_group': [age_group],
        'smoking_status': [smoking_status],
        'e_cigarette_usage': [e_cigarette_usage],
        'alcohol_consumption_rate': [alcohol_consumption_rate],
        'surgery_name': [surgery_name],
        'surgery_type': [surgery_type],
        'surgical_specialty': [surgical_specialty],
        'anesthesia_type': [anesthesia_type],
        'surgery_duration': [surgery_duration],
        'blood_loss_category': [blood_loss_category],
        'blood_transfusions': [blood_transfusions],
        'stay_duration': [stay_duration],
        'room_type': [room_type],
        'pain_score': [pain_score],
        'rehab_assessment_score': [rehab_assessment_score]
    })
    
    # Encode categorical data
    input_data = encode_categorical_data(input_data)
    
    # Make prediction
    prediction = model.predict(input_data)
    
    return prediction[0]

# Streamlit UI elements
st.title('Preoperative Risk Prediction')

# User Inputs
gender = st.selectbox('Gender', ['Male', 'Female'])
age_group = st.selectbox('Age Group', ['<20', '20-40', '40-60', '60+'])
smoking_status = st.selectbox('Smoking Status', ['Non-Smoker', 'Smoker'])
e_cigarette_usage = st.selectbox('E-Cigarette Usage', ['Yes', 'No'])
alcohol_consumption_rate = st.selectbox('Alcohol Consumption Rate', ['Low', 'Moderate', 'High', 'None'])
surgery_name = st.selectbox('Surgery Name', ['Cataract Surgery', 'Appendectomy', 'Spinal Fusion', 'Knee Replacement', 
                                              'Gallbladder Removal', 'Breast Cancer Surgery', 'Liver Transplant', 
                                              'Heart Bypass', 'Hip Replacement', 'Hernia Repair'])
surgery_type = st.selectbox('Surgery Type', ['Minor', 'Major'])
surgical_specialty = st.selectbox('Surgical Specialty', ['General', 'Orthopedic', 'Oncology', 'Transplant', 'Cardiothoracic'])
anesthesia_type = st.selectbox('Anesthesia Type', ['General', 'Local', 'Regional'])
surgery_duration = st.number_input('Surgery Duration (in minutes)', min_value=0)
blood_loss_category = st.selectbox('Blood Loss Category', ['Low', 'Medium', 'High'])
blood_transfusions = st.selectbox('Blood Transfusions', ['Yes', 'No'])
stay_duration = st.selectbox('Stay Duration', ['<1 Day', '1-3 Days', '3-7 Days', '>7 Days'])
room_type = st.selectbox('Room Type', ['Standard', 'VIP', 'ICU'])
pain_score = st.slider('Pain Score (1-10)', min_value=1, max_value=10)
rehab_assessment_score = st.slider('Rehabilitation Assessment Score (1-10)', min_value=1, max_value=10)

# Prediction button
if st.button('Predict Preoperative Risk'):
    # Get prediction from the model
    prediction = predict_preoperative_risk(gender, age_group, smoking_status, e_cigarette_usage, alcohol_consumption_rate,
                                           surgery_name, surgery_type, surgical_specialty, anesthesia_type, surgery_duration,
                                           blood_loss_category, blood_transfusions, stay_duration, room_type, pain_score,
                                           rehab_assessment_score)
    
    # Display the corresponding risk level
    if prediction == 0:
        risk_level = "Low Risk Surgery"
    elif prediction == 1:
        risk_level = "Moderate Risk Surgery"
    elif prediction == 2:
        risk_level = "High Risk Surgery"
    elif prediction == 3:
        risk_level = "Very High Risk Surgery"
    else:
        risk_level = "Unknown Risk Level"

    # Show the result
    st.write(f"The predicted preoperative risk class is: {risk_level}")