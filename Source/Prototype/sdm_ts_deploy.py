import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load your saved models
with open('C:\\Users\\Republic Of Gamers\\OneDrive\\Documents\\GitHub\\TSDN-BoyWithLuv\\Source\\Prototype\\xgb_elective.pkl', 'rb') as file:
    elective_model = pickle.load(file)

with open('C:\\Users\\Republic Of Gamers\\OneDrive\\Documents\\GitHub\\TSDN-BoyWithLuv\\Source\\Prototype\\xgb_emergency.pkl', 'rb') as file:
    emergency_model = pickle.load(file)

with open('C:\\Users\\Republic Of Gamers\\OneDrive\\Documents\\GitHub\\TSDN-BoyWithLuv\\Source\\Prototype\\xgb_urgent.pkl', 'rb') as file:
    urgent_model = pickle.load(file)

# Function to create input features for predictions
def create_input_features(admission_type, days):
    # Prepare predictions list
    predictions = []

    # Current date
    today = pd.to_datetime("today")

    # Define feature order
    if admission_type == 'Elective':
        feature_names = [
            'Elective_Lag_1', 'Elective_Lag_2', 'Elective_Lag_3', 
            ('Elective', 'Female'), ('Elective', 'Male'), 
            'month', 'day', 'year', 'quarter', 'dayofweek', 'dayofyear'
        ]
    elif admission_type == 'Emergency':
        feature_names = [
            'Emergency_Lag_1', 'Emergency_Lag_2', 'Emergency_Lag_3', 
            ('Emergency', 'Female'), ('Emergency', 'Male'), 
            'month', 'day', 'year', 'quarter', 'dayofweek', 'dayofyear'
        ]
    else:  # Urgent
        feature_names = [
            'Urgent_Lag_1', 'Urgent_Lag_2', 'Urgent_Lag_3', 
            ('Urgent', 'Female'), ('Urgent', 'Male'), 
            'month', 'day', 'year', 'quarter', 'dayofweek', 'dayofyear'
        ]

    for d in range(days):
        future_date = today + pd.Timedelta(days=d)
        
        # Create features with placeholders
        features = {
            'month': future_date.month,
            'day': future_date.day,
            'year': future_date.year,
            'quarter': future_date.quarter,
            'dayofweek': future_date.dayofweek,
            'dayofyear': future_date.dayofyear,
            # Add lag features with placeholder values
            f'{admission_type}_Lag_1': 0,
            f'{admission_type}_Lag_2': 0,
            f'{admission_type}_Lag_3': 0,
            # Admission type specific features with placeholders
            ('{}', 'Female'.format(admission_type)): 0,
            ('{}', 'Male'.format(admission_type)): 0,
        }

        # Create DataFrame in the correct order
        df_input = pd.DataFrame({name: features.get(name, 0) for name in feature_names}, index=[0])
        
        # Make predictions based on selected admission type
        if admission_type == 'Elective':
            prediction = elective_model.predict(df_input)
        elif admission_type == 'Emergency':
            prediction = emergency_model.predict(df_input)
        else:  # Urgent
            prediction = urgent_model.predict(df_input)

        predictions.append((future_date.date(), prediction[0]))

    return predictions

# Streamlit app layout
st.title('Hospital Admission Prediction')
st.header('Predict Future Admissions')

# User input for selecting admission type
admission_type = st.selectbox('Select Admission Type:', ('Elective', 'Emergency', 'Urgent'))

# User input for how many days to predict
days_to_predict = st.number_input('Number of days to predict:', min_value=1, max_value=30)

# Button to predict
if st.button('Predict'):
    st.subheader('Predictions:')
    
    # Get predictions based on the selected admission type
    predictions = create_input_features(admission_type, days_to_predict)
    
    for date, prediction in predictions:
        st.write(f'Prediction for {admission_type} admissions on {date}: {prediction}')