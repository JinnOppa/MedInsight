# import streamlit as st
# import joblib
# import pandas as pd

# ha_model = joblib.load(r'C:\Users\user\OneDrive\Documents\GitHub\TSDN-BoyWithLuv\Source\Prototype\disease_rf_heart_attack.pkl')
# angina_model = joblib.load(r'C:\Users\Eugene\Documents\GitHub\TSDN-BoyWithLuv\Source\Prototype\disease_xgb_angina.pkl')
# stroke_model = joblib.load(r'C:\Users\Eugene\Documents\GitHub\TSDN-BoyWithLuv\Source\Prototype\disease_rf_stroke.pkl')
# dd_model = joblib.load(r'C:\Users\Eugene\Documents\GitHub\TSDN-BoyWithLuv\Source\Prototype\disease_xgb_depressive_disorder.pkl')

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


# # Sidebar for user input
# st.sidebar.title('Main Menu')
# # nyc_logo = "NYC TAXI Logo.svg"
# # st.sidebar.image(nyc_logo, use_column_width=True)
# option = st.sidebar.radio('Options', ['Heart Attack Prediction', 'Angina Prediction', 'Stroke Prediction'])

# selected_disease = st.sidebar.selectbox("Choose a Prediction Model", ["Heart Attack", "Angina", "Stroke"])


# # if option == 'Heart Attack Prediction':
# #     st.title('Heart Attack Prediction')
# #     age_category = st.selectbox('Age Category', ['Age 18 to 24', 'Age 25 to 29', 'Age 30 to 34', 
# #                                                'Age 35 to 39', 'Age 40 to 44', 'Age 45 to 49', 
# #                                                'Age 50 to 54', 'Age 55 to 59', 'Age 60 to 64', 
# #                                                'Age 65 to 69', 'Age 70 to 74', 'Age 75 to 79', 
# #                                                'Age 80 or older'])
# #     sex = st.selectbox('Sex', ['Male', 'Female'])
# #     height = st.number_input('Height', min_value=0.0, max_value=100.0, value=25.0)
# #     weight = st.number_input('Weight', min_value=0.0, max_value=300.0, value=25.0)
# #     general_health = st.selectbox('General Health', ['Poor', 'Fair', 'Good', 'Very good', 'Excellent'])
# #     smoker_status = st.selectbox('Smoker Status', ['Never smoked', 'Former smoker', 
# #                                                 'Current smoker - now smokes some days', 
# #                                                 'Current smoker - now smokes every day'])
# #     alcohol_drinkers = st.number_input('Alcohol Drinkers', min_value=0, max_value=10, value=0)
# #     e_cigarette_usage = st.selectbox('E-Cigarette Usage', ['Never used e-cigarettes in my entire life', 
# #                                                         'Not at all (right now)', 
# #                                                         'Use them some days', 
# #                                                         'Use them every day'])
# #     had_stroke = st.selectbox('Had Stroke', [0, 1])
# #     had_copd = st.selectbox('Had COPD', [0, 1])
# #     had_diabetes = st.selectbox('Had Diabetes', ['No', 'Yes', 'No, pre-diabetes or borderline diabetes', 
# #                                                 'Yes, but only during pregnancy (female)'])
# #     had_kidney_disease = st.selectbox('Had Kidney Disease', [0, 1])
# #     had_arthritis = st.selectbox('Had Arthritis', [0, 1])
# #     had_angina = st.selectbox('Had Angina', [0, 1])
# #     pneumo_vax_ever = st.selectbox('PneumoVax Ever', [0, 1])
# #     chest_scan = st.selectbox('Chest Scan', [0, 1])

# # elif option == 'Angina Prediction':
    