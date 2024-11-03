# # HEART ATTACK

# import streamlit as st
# import joblib
# import pandas as pd

# # Load the best Random Forest model
# ha_xgb_model = joblib.load(r'C:\Users\Eugene\Documents\GitHub\TSDN-BoyWithLuv\Source\Prototype\disease_xgb_heart_attack.pkl')

# # Streamlit app title
# st.title('Heart Attack Prediction App')

# # User inputs for the prediction
# age_category = st.selectbox('Age Category', ['Age 18 to 24', 'Age 25 to 29', 'Age 30 to 34', 
#                                                'Age 35 to 39', 'Age 40 to 44', 'Age 45 to 49', 
#                                                'Age 50 to 54', 'Age 55 to 59', 'Age 60 to 64', 
#                                                'Age 65 to 69', 'Age 70 to 74', 'Age 75 to 79', 
#                                                'Age 80 or older'])

# sex = st.selectbox('Sex', ['Male', 'Female'])

# # Additional inputs
# ## daily record
# bmi = st.number_input('BMI', min_value=0.0, max_value=100.0, value=25.0)
# general_health = st.selectbox('General Health', ['Poor', 'Fair', 'Good', 'Very good', 'Excellent'])
# smoker_status = st.selectbox('Smoker Status', ['Never smoked', 'Former smoker', 
#                                                'Current smoker - now smokes some days', 
#                                                'Current smoker - now smokes every day'])
# alcohol_drinkers = st.number_input('Alcohol Drinkers', min_value=0, max_value=10, value=0)
# e_cigarette_usage = st.selectbox('E-Cigarette Usage', ['Never used e-cigarettes in my entire life', 
#                                                        'Not at all (right now)', 
#                                                        'Use them some days', 
#                                                        'Use them every day'])
# ## medical history
# had_stroke = st.selectbox('Had Stroke', [0, 1])
# had_copd = st.selectbox('Had COPD', [0, 1])
# had_diabetes = st.selectbox('Had Diabetes', ['No', 'Yes', 'No, pre-diabetes or borderline diabetes', 
#                                               'Yes, but only during pregnancy (female)'])
# had_kidney_disease = st.selectbox('Had Kidney Disease', [0, 1])
# had_arthritis = st.selectbox('Had Arthritis', [0, 1])
# had_angina = st.selectbox('Had Angina', [0, 1])

# ## screening and vaccination
# pneumo_vax_ever = st.selectbox('PneumoVax Ever', [0, 1])
# chest_scan = st.selectbox('Chest Scan', [0, 1])

# # Creating a DataFrame for the model input
# input_data = pd.DataFrame({
#     'AgeCategory': [age_category],
#     'Sex': [sex],
#     'BMI': [bmi],
#     'GeneralHealth': [general_health],
#     'SmokerStatus': [smoker_status],
#     'AlcoholDrinkers': [alcohol_drinkers],
#     'ECigaretteUsage': [e_cigarette_usage],
#     'HadStroke': [had_stroke],
#     'HadCOPD': [had_copd],
#     'HadDiabetes': [had_diabetes],
#     'HadKidneyDisease': [had_kidney_disease],
#     'HadArthritis': [had_arthritis],
#     'HadAngina': [had_angina],
#     'PneumoVaxEver': [pneumo_vax_ever],
#     'ChestScan': [chest_scan]
# })

# # Mapping the categorical inputs to numerical values
# general_health_mapping = {
#     "Excellent": 5,
#     "Very good": 4,
#     "Good": 3,
#     "Fair": 2,
#     "Poor": 1
# }

# age_category_mapping = {
#     "Age 18 to 24": 1,
#     "Age 25 to 29": 2,
#     "Age 30 to 34": 3,
#     "Age 35 to 39": 4,
#     "Age 40 to 44": 5,
#     "Age 45 to 49": 6,
#     "Age 50 to 54": 7,
#     "Age 55 to 59": 8,
#     "Age 60 to 64": 9,
#     "Age 65 to 69": 10,
#     "Age 70 to 74": 11,
#     "Age 75 to 79": 12,
#     "Age 80 or older": 13
# }

# smoker_status_mapping = {
#     "Never smoked": 0,
#     "Former smoker": 1,
#     "Current smoker - now smokes some days": 2,
#     "Current smoker - now smokes every day": 3
# }

# diabetes_mapping = {
#     "No": 0,
#     "Yes": 1,
#     "No, pre-diabetes or borderline diabetes": 2,
#     "Yes, but only during pregnancy (female)": 3
# }

# e_cigarette_usage_mapping = {
#     "Never used e-cigarettes in my entire life": 0,
#     "Not at all (right now)": 1,
#     "Use them some days": 2,
#     "Use them every day": 3
# }

# # Applying the mappings
# input_data['GeneralHealth'] = input_data['GeneralHealth'].map(general_health_mapping)
# input_data['AgeCategory'] = input_data['AgeCategory'].map(age_category_mapping)
# input_data['SmokerStatus'] = input_data['SmokerStatus'].map(smoker_status_mapping)
# input_data['HadDiabetes'] = input_data['HadDiabetes'].map(diabetes_mapping)
# input_data['ECigaretteUsage'] = input_data['ECigaretteUsage'].map(e_cigarette_usage_mapping)

# # Convert 'Sex' to numerical (assuming Male=0, Female=1)
# input_data['Sex'] = input_data['Sex'].map({'Male': 0, 'Female': 1})

# # Predict using the Random Forest model
# if st.button('Predict Heart Attack'):
#     rf_prediction = ha_xgb_model.predict(input_data)
#     st.write('Heart Attack Prediction:', 'Had Heart Attack' if rf_prediction[0] == 1 else 'No Heart Attack')


# # ANGINA

# import streamlit as st
# import joblib
# import pandas as pd

# # Load the best Random Forest model
# angina_xgb_model = joblib.load(r'C:\Users\Eugene\Documents\GitHub\TSDN-BoyWithLuv\Source\Prototype\disease_xgb_angina.pkl')

# # Streamlit app title
# st.title('Angina Prediction App')

# # User inputs for the prediction
# age_category = st.selectbox('Age Category', ['Age 18 to 24', 'Age 25 to 29', 'Age 30 to 34', 
#                                                'Age 35 to 39', 'Age 40 to 44', 'Age 45 to 49', 
#                                                'Age 50 to 54', 'Age 55 to 59', 'Age 60 to 64', 
#                                                'Age 65 to 69', 'Age 70 to 74', 'Age 75 to 79', 
#                                                'Age 80 or older'])

# sex = st.selectbox('Sex', ['Male', 'Female'])

# # Additional inputs
# ## daily record
# bmi = st.number_input('BMI', min_value=0.0, max_value=100.0, value=25.0)
# general_health = st.selectbox('General Health', ['Poor', 'Fair', 'Good', 'Very good', 'Excellent'])
# smoker_status = st.selectbox('Smoker Status', ['Never smoked', 'Former smoker', 
#                                                'Current smoker - now smokes some days', 
#                                                'Current smoker - now smokes every day'])
# alcohol_drinkers = st.number_input('Alcohol Drinkers', min_value=0, max_value=10, value=0)
# e_cigarette_usage = st.selectbox('E-Cigarette Usage', ['Never used e-cigarettes in my entire life', 
#                                                        'Not at all (right now)', 
#                                                        'Use them some days', 
#                                                        'Use them every day'])
# ## medical history
# had_stroke = st.selectbox('Had Stroke', [0, 1])
# had_copd = st.selectbox('Had COPD', [0, 1])
# had_diabetes = st.selectbox('Had Diabetes', ['No', 'Yes', 'No, pre-diabetes or borderline diabetes', 
#                                               'Yes, but only during pregnancy (female)'])
# had_kidney_disease = st.selectbox('Had Kidney Disease', [0, 1])
# had_arthritis = st.selectbox('Had Arthritis', [0, 1])

# ## screening and vaccination
# pneumo_vax_ever = st.selectbox('PneumoVax Ever', [0, 1])
# chest_scan = st.selectbox('Chest Scan', [0, 1])

# # Creating a DataFrame for the model input
# input_data = pd.DataFrame({
#     'AgeCategory': [age_category],
#     'Sex': [sex],
#     'BMI': [bmi],
#     'GeneralHealth': [general_health],
#     'SmokerStatus': [smoker_status],
#     'AlcoholDrinkers': [alcohol_drinkers],
#     'ECigaretteUsage': [e_cigarette_usage],
#     'HadStroke': [had_stroke],
#     'HadCOPD': [had_copd],
#     'HadDiabetes': [had_diabetes],
#     'HadKidneyDisease': [had_kidney_disease],
#     'HadArthritis': [had_arthritis],
#     'PneumoVaxEver': [pneumo_vax_ever],
#     'ChestScan': [chest_scan]
# })

# # Mapping the categorical inputs to numerical values
# general_health_mapping = {
#     "Excellent": 5,
#     "Very good": 4,
#     "Good": 3,
#     "Fair": 2,
#     "Poor": 1
# }

# age_category_mapping = {
#     "Age 18 to 24": 1,
#     "Age 25 to 29": 2,
#     "Age 30 to 34": 3,
#     "Age 35 to 39": 4,
#     "Age 40 to 44": 5,
#     "Age 45 to 49": 6,
#     "Age 50 to 54": 7,
#     "Age 55 to 59": 8,
#     "Age 60 to 64": 9,
#     "Age 65 to 69": 10,
#     "Age 70 to 74": 11,
#     "Age 75 to 79": 12,
#     "Age 80 or older": 13
# }

# smoker_status_mapping = {
#     "Never smoked": 0,
#     "Former smoker": 1,
#     "Current smoker - now smokes some days": 2,
#     "Current smoker - now smokes every day": 3
# }

# diabetes_mapping = {
#     "No": 0,
#     "Yes": 1,
#     "No, pre-diabetes or borderline diabetes": 2,
#     "Yes, but only during pregnancy (female)": 3
# }

# e_cigarette_usage_mapping = {
#     "Never used e-cigarettes in my entire life": 0,
#     "Not at all (right now)": 1,
#     "Use them some days": 2,
#     "Use them every day": 3
# }

# # Applying the mappings
# input_data['GeneralHealth'] = input_data['GeneralHealth'].map(general_health_mapping)
# input_data['AgeCategory'] = input_data['AgeCategory'].map(age_category_mapping)
# input_data['SmokerStatus'] = input_data['SmokerStatus'].map(smoker_status_mapping)
# input_data['HadDiabetes'] = input_data['HadDiabetes'].map(diabetes_mapping)
# input_data['ECigaretteUsage'] = input_data['ECigaretteUsage'].map(e_cigarette_usage_mapping)

# # Convert 'Sex' to numerical (assuming Male=0, Female=1)
# input_data['Sex'] = input_data['Sex'].map({'Male': 0, 'Female': 1})

# # Predict using the Random Forest model
# if st.button('Predict Angina'):
#     rf_prediction = angina_xgb_model.predict(input_data)
#     st.write('Angina Prediction:', 'Have Angina' if rf_prediction[0] == 1 else "Don't Have Angina")

# STROKE

import streamlit as st
import joblib
import pandas as pd

# Load the best Random Forest model
stroke_xgb_model = joblib.load(r'C:\Users\Eugene\Documents\GitHub\TSDN-BoyWithLuv\Source\Prototype\disease_rf_stroke.pkl')

# Streamlit app title
st.title('Stroke Prediction App')

# User inputs for the prediction
age_category = st.selectbox('Age Category', ['Age 18 to 24', 'Age 25 to 29', 'Age 30 to 34', 
                                               'Age 35 to 39', 'Age 40 to 44', 'Age 45 to 49', 
                                               'Age 50 to 54', 'Age 55 to 59', 'Age 60 to 64', 
                                               'Age 65 to 69', 'Age 70 to 74', 'Age 75 to 79', 
                                               'Age 80 or older'])

sex = st.selectbox('Sex', ['Male', 'Female'])

# Additional inputs
## daily record
bmi = st.number_input('BMI', min_value=0.0, max_value=100.0, value=25.0)
general_health = st.selectbox('General Health', ['Poor', 'Fair', 'Good', 'Very good', 'Excellent'])
smoker_status = st.selectbox('Smoker Status', ['Never smoked', 'Former smoker', 
                                               'Current smoker - now smokes some days', 
                                               'Current smoker - now smokes every day'])
alcohol_drinkers = st.number_input('Alcohol Drinkers', min_value=0, max_value=10, value=0)
e_cigarette_usage = st.selectbox('E-Cigarette Usage', ['Never used e-cigarettes in my entire life', 
                                                       'Not at all (right now)', 
                                                       'Use them some days', 
                                                       'Use them every day'])
## medical history
# Medical history inputs
had_heart_attack = st.selectbox('Had Heart Attack', [0, 1])
had_angina = st.selectbox('Had Angina', [0, 1])
had_copd = st.selectbox('Had COPD', [0, 1])
had_diabetes = st.selectbox('Had Diabetes', ['No', 'Yes', 'No, pre-diabetes or borderline diabetes', 
                                              'Yes, but only during pregnancy (female)'])
had_arthritis = st.selectbox('Had Arthritis', [0, 1])

difficulty_walking = st.selectbox('Had Difficulity to Walk', [0, 1])
difficulty_concentrate = st.selectbox('Had Difficulity to Concentrate', [0, 1])

# Screening and vaccination inputs
pneumo_vax_ever = st.selectbox('PneumoVax Ever', [0, 1])
chest_scan = st.selectbox('Chest Scan', [0, 1])


# Creating a DataFrame for the model input
input_data = pd.DataFrame({
    'AgeCategory': [age_category],
    'Sex': [sex],
    'BMI': [bmi],
    'GeneralHealth': [general_health],
    'SmokerStatus': [smoker_status],
    'AlcoholDrinkers': [alcohol_drinkers],
    'ECigaretteUsage': [e_cigarette_usage],
    'HadHeartAttack': [had_heart_attack],
    'HadAngina': [had_angina],
    'HadCOPD': [had_copd],
    'HadDiabetes': [had_diabetes],
    'DifficultyWalking':[difficulty_walking],
    'HadArthritis': [had_arthritis],
    'DifficultyConcentrating':[difficulty_concentrate],
    'PneumoVaxEver': [pneumo_vax_ever],
    'ChestScan': [chest_scan]
})

# Mapping the categorical inputs to numerical values
general_health_mapping = {
    "Excellent": 5,
    "Very good": 4,
    "Good": 3,
    "Fair": 2,
    "Poor": 1
}

age_category_mapping = {
    "Age 18 to 24": 1,
    "Age 25 to 29": 2,
    "Age 30 to 34": 3,
    "Age 35 to 39": 4,
    "Age 40 to 44": 5,
    "Age 45 to 49": 6,
    "Age 50 to 54": 7,
    "Age 55 to 59": 8,
    "Age 60 to 64": 9,
    "Age 65 to 69": 10,
    "Age 70 to 74": 11,
    "Age 75 to 79": 12,
    "Age 80 or older": 13
}

smoker_status_mapping = {
    "Never smoked": 0,
    "Former smoker": 1,
    "Current smoker - now smokes some days": 2,
    "Current smoker - now smokes every day": 3
}

diabetes_mapping = {
    "No": 0,
    "Yes": 1,
    "No, pre-diabetes or borderline diabetes": 2,
    "Yes, but only during pregnancy (female)": 3
}

e_cigarette_usage_mapping = {
    "Never used e-cigarettes in my entire life": 0,
    "Not at all (right now)": 1,
    "Use them some days": 2,
    "Use them every day": 3
}

# Applying the mappings
input_data['GeneralHealth'] = input_data['GeneralHealth'].map(general_health_mapping)
input_data['AgeCategory'] = input_data['AgeCategory'].map(age_category_mapping)
input_data['SmokerStatus'] = input_data['SmokerStatus'].map(smoker_status_mapping)
input_data['HadDiabetes'] = input_data['HadDiabetes'].map(diabetes_mapping)
input_data['ECigaretteUsage'] = input_data['ECigaretteUsage'].map(e_cigarette_usage_mapping)

# Convert 'Sex' to numerical (assuming Male=0, Female=1)
input_data['Sex'] = input_data['Sex'].map({'Male': 0, 'Female': 1})

# Predict using the Random Forest model
if st.button('Predict Stroke'):
    rf_prediction = stroke_xgb_model.predict(input_data)
    st.write('Stroke Prediction:', 'Have Stroke' if rf_prediction[0] == 1 else "Don't Have Stroke")
