import streamlit as st
import pandas as pd
import pickle
import numpy as np
from datetime import datetime, timedelta

# Load the saved model
with open('C:\\Users\\Republic Of Gamers\\OneDrive\\Documents\\GitHub\\TSDN-BoyWithLuv\\Source\\Prototype\\draft2_elective_rf.pkl', 'rb') as file:
    elective_model = pickle.load(file)

# Function to make predictions
def predict_admissions(days_ahead, model):
    # Initial base features (user can adjust these or get them from other sources)
    base_features = {
        'Female': 50,  # Placeholder values, update as needed
        'Male': 50,
        'lag_1': 95,
        'lag_7': 80,
        'lag_30': 70,
        # 'day': datetime.now().day,
        # 'month': datetime.now().month,
        # 'year': datetime.now().year,
    }

    # Compute rolling mean for 7 days (assuming it's based on `lag_1`, `lag_7`, `lag_30`)
    # If you have actual past data, use it to calculate rolling mean. For now, we can set it as a placeholder
    base_features['rolling_mean_7'] = np.mean([base_features['lag_1'], base_features['lag_7'], base_features['lag_30']])  # Adjust logic based on actual use

    # Construct a DataFrame for input
    input_data = pd.DataFrame([base_features])

    # Adjust the number of days to predict
    future_predictions = []
    for _ in range(days_ahead):
        pred = model.predict(input_data)
        future_predictions.append(pred[0])  # Assumes a single prediction for each day

        # Update the features for the next day (based on the most recent prediction)
        input_data['lag_1'] = pred[0]
        input_data['lag_7'] = input_data['lag_1']  # Example update, use actual logic to calculate this
        input_data['lag_30'] = input_data['lag_1']
        input_data['rolling_mean_7'] = np.mean([input_data['lag_1'], input_data['lag_7'], input_data['lag_30']])
        # input_data['day'] = (datetime.now() + timedelta(days=_ + 1)).day
        # input_data['month'] = (datetime.now() + timedelta(days=_ + 1)).month
        # input_data['year'] = (datetime.now() + timedelta(days=_ + 1)).year

    return future_predictions

# Streamlit UI
st.title("Elective Admission Prediction")

# Input: Number of days to predict
days_ahead = st.slider('Select number of days to predict:', 1, 30, 7)

# Button to make prediction
if st.button('Predict Admissions'):
    predictions = predict_admissions(days_ahead, elective_model)
    
    # Display the predictions
    prediction_dates = [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days_ahead)]
    
    st.write(f"Predictions for the next {days_ahead} days:")
    prediction_df = pd.DataFrame({
        'Date': prediction_dates,
        'Predicted Elective Admissions': predictions
    })
    
    st.write(prediction_df)