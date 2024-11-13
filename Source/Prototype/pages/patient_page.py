import streamlit as st
import joblib
import pandas as pd
# import xgboost as xgb
import math

page = st.sidebar.radio("Go to:", ("Home", "Stay Duration"))

# functional_disability_level_mapping = {
#     '<2 mo. follow-up': 1, 
#     'no(M2 and SIP pres)': 2, 
#     'SIP>=30': 3,
#     'adl>=4 (>=5 if sur)': 4, 
#     'Coma or Intub':5 
# }

gender_mapping = {
    'Male': 1, 
    'Female': 2
}

disease_class_mapping = {
    'Cancer': 1,
    'COPD/CHF/Cirrhosis': 2,
    'ARF/MOSF': 3,
    'Coma': 4
}

cancer_status_mapping = {
    'metastatic': 1,
    'no': 0,
    'yes': 2
}

has_dementia_mapping = {
    'False': 0,
    'True': 1
}

has_diabetes_mapping = {
    'False': 0,
    'True': 1
}

has_cancer_mapping = {
    'False': 0,
    'True': 1
}

if page == "Home":
    st.title("Disease Prediction")
    
elif page == "Stay Duration":
    stay_model = joblib.load('Source/Prototype/patient_stay/stay_xgb.joblib')
    # if isinstance(stay_model, xgb.XGBModel):
    #     stay_model.set_params(tree_method="hist")
    
    age = st.number_input('Age', min_value=5, max_value=100, value=20)
    sex = st.selectbox('Sex', ['Male', 'Female'])
    function_level = st.number_input('Functionality Level', min_value=1, max_value=5, value=3)
    diabetes = st.selectbox('Had Diabetes', ['False','True'])
    dementia = st.selectbox('Had Dementia', ['False','True'])
    cancer = st.selectbox('Had Cancer', ['False','True'])
    n_comorbidities = st.number_input('Number Comorbidities', min_value=0, max_value=10, value=1)
    admission_type = st.selectbox('Admission Type',[
        'Cancer', 'COPD/CHF/Cirrhosis',
        'ARF/MOSF', 'Coma'
    ])
    
    input_data = pd.DataFrame({
        'age_years':[age],
        'gender':[sex],
        'functional_disability_level':[function_level],
        'has_diabetes':[diabetes],
        'has_dementia':[dementia],
        'cancer_status':[cancer],
        'num_comorbidities':[n_comorbidities],
        'disease_class':[admission_type],
    })
    
    input_data['gender'] = input_data['gender'].map(gender_mapping)
    input_data['has_diabetes'] = input_data['has_diabetes'].map(has_diabetes_mapping)
    input_data['has_dementia'] = input_data['has_dementia'].map(has_dementia_mapping)
    input_data['cancer_status'] = input_data['cancer_status'].map(has_cancer_mapping)
    input_data['disease_class'] = input_data['disease_class'].map(disease_class_mapping)
    
        # Add a button to make the prediction
    if st.button("Predict Stay Duration"):
        # Make prediction
        prediction = stay_model.predict(input_data)[0]
        # Round up the prediction to the nearest whole number
        rounded_prediction = math.ceil(prediction)
        
        # Display the result
        st.subheader("Predicted Stay Duration")
        st.write(f"The predicted stay duration for this patient is: **{rounded_prediction} days**")