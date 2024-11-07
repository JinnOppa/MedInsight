import streamlit as st
import pandas as pd
import pickle
from datetime import timedelta
import numpy as np

# Load saved models for each admission type
with open("C:\\Users\\Republic Of Gamers\\OneDrive\\Documents\\GitHub\\TSDN-BoyWithLuv\\Source\\Prototype\\xgb_elective.pkl", "rb") as file:
    elective_model = pickle.load(file)
with open("C:\\Users\\Republic Of Gamers\\OneDrive\\Documents\\GitHub\\TSDN-BoyWithLuv\\Source\\Prototype\\xgb_emergency.pkl", "rb") as file:
    emergency_model = pickle.load(file)
with open("C:\\Users\\Republic Of Gamers\\OneDrive\\Documents\\GitHub\\TSDN-BoyWithLuv\\Source\\Prototype\\xgb_urgent.pkl", "rb") as file:
    urgent_model = pickle.load(file)

# Function to get column names based on the admission type
def get_feature_columns(admission_type):
    return [
        f"('{admission_type}', 'Female')", 
        f"('{admission_type}', 'Male')", 
        f"{admission_type}_Lag_1", 
        f"{admission_type}_Lag_2", 
        f"{admission_type}_Lag_3", 
        "month", "day", "year", "quarter", "dayofweek", "dayofyear"
    ]

# Set up Streamlit interface
st.title("Hospital Admission Prediction by Type")
st.write("Select an admission type to predict future admissions.")

# Dropdown to select the admission type
admission_type = st.selectbox("Choose Admission Type to Predict:", ["Elective", "Emergency", "Urgent"])

# Load the appropriate model
model = {
    "Elective": elective_model,
    "Emergency": emergency_model,
    "Urgent": urgent_model
}[admission_type]

# Input for number of days to predict
days_to_predict = st.number_input("Enter the number of days to predict:", min_value=1, max_value=30, step=1)

# Button to initiate predictions
if st.button("Predict"):
    # Initialize dummy data with the correct column names for the selected admission type
    feature_columns = get_feature_columns(admission_type)
    data = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)
    
    # Placeholder for predictions
    predictions = []
    
    # Generate predictions for specified days
    for day in range(days_to_predict):
        # Predict next day's admission count
        prediction = model.predict(data)[0]
        predictions.append(prediction)
        
        # Update lag values and shift features for the next prediction
        data[f"{admission_type}_Lag_3"] = data[f"{admission_type}_Lag_2"]
        data[f"{admission_type}_Lag_2"] = data[f"{admission_type}_Lag_1"]
        data[f"{admission_type}_Lag_1"] = prediction
        
        # Increment date features if necessary (e.g., dayofweek, dayofyear, etc.)
    
    # Display predictions as a line chart
    st.write(f"Predicted {admission_type} admissions for the next {days_to_predict} days:")
    st.line_chart(predictions)