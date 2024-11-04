# disease_prediction.py
import streamlit as st
import joblib
import pandas as pd


st.title("Disease Prediction")
# st.write("## This page will provide options for disease prediction models.")

# Sidebar navigation
page = st.sidebar.radio("Go to:", ("Home", "Heart Attack", "Angina", "Stroke"))

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

sex_mapping = {
    'Male': 0,
    "Female": 1,
}

stroke_mapping = {
    'False': 0,
    'True': 1
}

copd_mapping = {
    'False': 0,
    'True': 1
}

kidney_disease_mapping = {
    'False': 0,
    'True': 1
}

arthritis_mapping = {
    'False': 0,
    'True': 1
}

angina_mapping = {
    'False': 0,
    'True': 1
}

heart_attack_mapping = {
    'False': 0,
    'True': 1
}

pneumo_vax_mapping = {
    'False': 0,
    'True': 1
}

chest_scan_mapping = {
    'False': 0,
    'True': 1
}

difficult_walk_mapping = {
    'False': 0,
    'True': 1
}

if page == "Home":
    st.write("# Welcome to Streamlit!")
    st.markdown(
        """
        Streamlit is an open-source app framework built specifically for
        Machine Learning and Data Science projects.
        **👈 Select a page from the sidebar** to explore the web app features!
        ### Learn More
        - [Streamlit Documentation](https://docs.streamlit.io)
        - [Streamlit Community](https://discuss.streamlit.io)
        """
    )
    
elif page == "Heart Attack":

    # Load the model
    ha_model = joblib.load(r'C:\Users\user\OneDrive\Documents\GitHub\TSDN-BoyWithLuv\Source\Prototype\disease_rf_heart_attack.pkl')

    # Page title and description
    st.title("Heart Attack Risk Prediction")
    st.write("Enter the following information to predict the risk of a heart attack.")

    # Collapsible input sections
    st.subheader("Personal Details")
    with st.expander("Enter Personal Information"):
        age_category = st.selectbox('Age Category', [
            'Age 18 to 24', 'Age 25 to 29', 'Age 30 to 34', 
            'Age 35 to 39', 'Age 40 to 44', 'Age 45 to 49', 
            'Age 50 to 54', 'Age 55 to 59', 'Age 60 to 64', 
            'Age 65 to 69', 'Age 70 to 74', 'Age 75 to 79', 
            'Age 80 or older'
        ])
        sex = st.selectbox('Sex', ['Male', 'Female'])
        height = st.number_input('Height (meters)', min_value=0.0, max_value=2.5, value=1.65)
        weight = st.number_input('Weight (kg)', min_value=0.0, max_value=300.0, value=65.0)

    st.subheader("Health & Lifestyle")
    with st.expander("Enter Health and Lifestyle Details"):
        general_health = st.selectbox('General Health', ['Poor', 'Fair', 'Good', 'Very good', 'Excellent'])
        smoker_status = st.selectbox('Smoker Status', [
            'Never smoked', 'Former smoker', 
            'Current smoker - now smokes some days', 
            'Current smoker - now smokes every day'
        ])
        alcohol_drinkers = st.number_input('Alcohol Consumption (drinks per week)', min_value=0, max_value=50, value=0)
        e_cigarette_usage = st.selectbox('E-Cigarette Usage', [
            'Never used e-cigarettes in my entire life', 
            'Not at all (right now)', 'Use them some days', 
            'Use them every day'
        ])

    st.subheader("Medical History")
    with st.expander("Enter Medical History"):
        had_stroke = st.selectbox('Had Stroke', ['False','True'])
        had_copd = st.selectbox('Had COPD', ['False','True'])
        had_diabetes = st.selectbox('Had Diabetes', [
            'No', 'Yes', 'No, pre-diabetes or borderline diabetes', 
            'Yes, but only during pregnancy (female)'
        ])
        had_kidney_disease = st.selectbox('Had Kidney Disease', ['False','True'])
        had_arthritis = st.selectbox('Had Arthritis', ['False','True'])
        had_angina = st.selectbox('Had Angina', ['False','True'])

    st.subheader("Screening and Vaccination")
    with st.expander("Enter Screening and Vaccination Details"):
        pneumo_vax_ever = st.selectbox('PneumoVax Ever', ['False','True'])
        chest_scan = st.selectbox('Chest Scan', ['False','True'])

    # Create DataFrame for model input
    input_data = pd.DataFrame({
        'AgeCategory': [age_category],
        'Sex': [sex],
        'HeightInMeters': [height],
        'WeightInKilograms': [weight],
        'GeneralHealth': [general_health],
        'SmokerStatus': [smoker_status],
        'AlcoholDrinkers': [alcohol_drinkers],
        'ECigaretteUsage': [e_cigarette_usage],
        'HadStroke': [had_stroke],
        'HadCOPD': [had_copd],
        'HadDiabetes': [had_diabetes],
        'HadKidneyDisease': [had_kidney_disease],
        'HadArthritis': [had_arthritis],
        'HadAngina': [had_angina],
        'PneumoVaxEver': [pneumo_vax_ever],
        'ChestScan': [chest_scan]
    })

    
    input_data['AgeCategory'] = input_data['AgeCategory'].map(age_category_mapping)    
    input_data['Sex'] = input_data['Sex'].map(sex_mapping)
    input_data['GeneralHealth'] = input_data['GeneralHealth'].map(general_health_mapping)
    input_data['SmokerStatus'] = input_data['SmokerStatus'].map(smoker_status_mapping)
    input_data['ECigaretteUsage'] = input_data['ECigaretteUsage'].map(e_cigarette_usage_mapping)
    input_data['HadStroke'] = input_data['HadStroke'].map(stroke_mapping)
    input_data['HadCOPD'] = input_data['HadCOPD'].map(copd_mapping)
    input_data['HadDiabetes'] = input_data['HadDiabetes'].map(diabetes_mapping)
    input_data['HadKidneyDisease'] = input_data['HadKidneyDisease'].map(kidney_disease_mapping)
    input_data['HadArthritis'] = input_data['HadArthritis'].map(arthritis_mapping)
    input_data['HadAngina'] = input_data['HadAngina'].map(angina_mapping)
    input_data['PneumoVaxEver'] = input_data['PneumoVaxEver'].map(pneumo_vax_mapping)
    input_data['ChestScan'] = input_data['ChestScan'].map(chest_scan_mapping)

    # Predict button
    st.markdown("---")
    if st.button("Predict Heart Attack Risk"):
        rf_prediction = ha_model.predict(input_data)
        result = 'High risk of heart attack.' if rf_prediction[0] == 1 else 'Low risk of heart attack.'
        # Display prediction result with custom styling
        if rf_prediction[0] == 1:
            st.error(f"Prediction Result: {result}")
        else:
            st.success(f"Prediction Result: {result}")

elif page == "Angina":
    
    # Load the model
    angina_model = joblib.load(r'C:\Users\user\OneDrive\Documents\GitHub\TSDN-BoyWithLuv\Source\Prototype\disease_xgb_angina.pkl')

    # Page title and description
    st.title("Angina Risk Prediction")
    st.write("Enter the following information to predict the risk of Angina.")
    
    # Collapsible input sections
    st.subheader("Personal Details")
    with st.expander("Enter Personal Information"):
        age_category = st.selectbox('Age Category', [
            'Age 18 to 24', 'Age 25 to 29', 'Age 30 to 34', 
            'Age 35 to 39', 'Age 40 to 44', 'Age 45 to 49', 
            'Age 50 to 54', 'Age 55 to 59', 'Age 60 to 64', 
            'Age 65 to 69', 'Age 70 to 74', 'Age 75 to 79', 
            'Age 80 or older'
        ])
        sex = st.selectbox('Sex', ['Male', 'Female'])
        height = st.number_input('Height (meters)', min_value=0.0, max_value=2.5, value=1.65)
        weight = st.number_input('Weight (kg)', min_value=0.0, max_value=300.0, value=65.0)

    st.subheader("Health & Lifestyle")
    with st.expander("Enter Health and Lifestyle Details"):
        general_health = st.selectbox('General Health', ['Poor', 'Fair', 'Good', 'Very good', 'Excellent'])
        smoker_status = st.selectbox('Smoker Status', [
            'Never smoked', 'Former smoker', 
            'Current smoker - now smokes some days', 
            'Current smoker - now smokes every day'
        ])
        alcohol_drinkers = st.number_input('Alcohol Consumption (drinks per week)', min_value=0, max_value=50, value=0)
        e_cigarette_usage = st.selectbox('E-Cigarette Usage', [
            'Never used e-cigarettes in my entire life', 
            'Not at all (right now)', 'Use them some days', 
            'Use them every day'
        ])
    
    st.subheader("Medical History")
    with st.expander("Enter Medical History"):
        had_stroke = st.selectbox('Had Stroke', ['False','True'])
        had_copd = st.selectbox('Had COPD', ['False','True'])
        had_diabetes = st.selectbox('Had Diabetes', [
            'No', 'Yes', 'No, pre-diabetes or borderline diabetes', 
            'Yes, but only during pregnancy (female)'
        ])
        had_kidney_disease = st.selectbox('Had Kidney Disease', ['False','True'])
        had_arthritis = st.selectbox('Had Arthritis', ['False','True'])
        had_heart_attack = st.selectbox('Had Heart Attack', ['False','True'])

    st.subheader("Screening and Vaccination")
    with st.expander("Enter Screening and Vaccination Details"):
        pneumo_vax_ever = st.selectbox('PneumoVax Ever', ['False','True'])
        chest_scan = st.selectbox('Chest Scan', ['False','True'])


    # Creating a DataFrame for the model input
    input_data = pd.DataFrame({
        'AgeCategory': [age_category],
        'Sex': [sex],
        'HeightInMeters': [height],
        'WeightInKilograms': [weight],
        'GeneralHealth': [general_health],
        'SmokerStatus': [smoker_status],
        'AlcoholDrinkers': [alcohol_drinkers],
        'ECigaretteUsage': [e_cigarette_usage],
        'HadStroke': [had_stroke], 
        'HadCOPD': [had_copd],
        'HadDiabetes': [had_diabetes],
        'HadKidneyDisease': [had_kidney_disease],
        'HadArthritis': [had_arthritis],
        'HadHeartAttack':[had_heart_attack],
        'PneumoVaxEver': [pneumo_vax_ever],
        'ChestScan': [chest_scan]
    })
    
    # Applying the mappings
    input_data['AgeCategory'] = input_data['AgeCategory'].map(age_category_mapping)    
    input_data['Sex'] = input_data['Sex'].map(sex_mapping)
    input_data['GeneralHealth'] = input_data['GeneralHealth'].map(general_health_mapping)
    input_data['SmokerStatus'] = input_data['SmokerStatus'].map(smoker_status_mapping)
    input_data['ECigaretteUsage'] = input_data['ECigaretteUsage'].map(e_cigarette_usage_mapping)
    input_data['HadStroke'] = input_data['HadStroke'].map(stroke_mapping)
    input_data['HadCOPD'] = input_data['HadCOPD'].map(copd_mapping)
    input_data['HadDiabetes'] = input_data['HadDiabetes'].map(diabetes_mapping)
    input_data['HadKidneyDisease'] = input_data['HadKidneyDisease'].map(kidney_disease_mapping)
    input_data['HadArthritis'] = input_data['HadArthritis'].map(arthritis_mapping)
    input_data['HadHeartAttack'] = input_data['HadHeartAttack'].map(heart_attack_mapping)
    input_data['PneumoVaxEver'] = input_data['PneumoVaxEver'].map(pneumo_vax_mapping)
    input_data['ChestScan'] = input_data['ChestScan'].map(chest_scan_mapping)
    #input_data[''] = input_data[''].map()
    
    # Predict button
    st.markdown("---")
    if st.button("Predict Angina Risk"):
        rf_prediction = angina_model.predict(input_data)
        result = 'High risk of having Angina.' if rf_prediction[0] == 1 else 'Low risk of having Angina.'
        # Display prediction result with custom styling
        if rf_prediction[0] == 1:
            st.error(f"Prediction Result: {result}")
        else:
            st.success(f"Prediction Result: {result}")

elif page == "Stroke":
    
    # Load the best Random Forest model
    stroke_model = joblib.load(r'C:\Users\user\OneDrive\Documents\GitHub\TSDN-BoyWithLuv\Source\Prototype\disease_rf_stroke.pkl')
    
    # Page title and description
    st.title("Stroke Risk Prediction")
    st.write("Enter the following information to predict the risk of Stroke.")
    
    # Collapsible input sections
    st.subheader("Personal Details")
    with st.expander("Enter Personal Information"):
        age_category = st.selectbox('Age Category', [
            'Age 18 to 24', 'Age 25 to 29', 'Age 30 to 34', 
            'Age 35 to 39', 'Age 40 to 44', 'Age 45 to 49', 
            'Age 50 to 54', 'Age 55 to 59', 'Age 60 to 64', 
            'Age 65 to 69', 'Age 70 to 74', 'Age 75 to 79', 
            'Age 80 or older'
        ])
        sex = st.selectbox('Sex', ['Male', 'Female'])
        height = st.number_input('Height (meters)', min_value=0.0, max_value=2.5, value=1.65)
        weight = st.number_input('Weight (kg)', min_value=0.0, max_value=300.0, value=65.0)
    
    st.subheader("Health & Lifestyle")
    with st.expander("Enter Health and Lifestyle Details"):
        general_health = st.selectbox('General Health', ['Poor', 'Fair', 'Good', 'Very good', 'Excellent'])
        smoker_status = st.selectbox('Smoker Status', [
            'Never smoked', 'Former smoker', 
            'Current smoker - now smokes some days', 
            'Current smoker - now smokes every day'
        ])
        alcohol_drinkers = st.number_input('Alcohol Consumption (drinks per week)', min_value=0, max_value=50, value=0)
        e_cigarette_usage = st.selectbox('E-Cigarette Usage', [
            'Never used e-cigarettes in my entire life', 
            'Not at all (right now)', 'Use them some days', 
            'Use them every day'
        ])
    
    st.subheader("Medical History")
    with st.expander("Enter Medical History"):
        had_heart_attack = st.selectbox('Had Heart Attack', ['False','True'])
        had_copd = st.selectbox('Had COPD', ['False','True'])
        had_diabetes = st.selectbox('Had Diabetes', ['No', 'Yes', 'No, pre-diabetes or borderline diabetes', 
                                                    'Yes, but only during pregnancy (female)'])
        had_kidney_disease = st.selectbox('Had Kidney Disease', ['False','True'])
        had_arthritis = st.selectbox('Had Arthritis', ['False','True'])
        had_angina = st.selectbox('Had Angina', ['False','True'])
        
    st.subheader("Screening and Vaccination")
    with st.expander("Enter Screening and Vaccination Details"):
        pneumo_vax_ever = st.selectbox('PneumoVax Ever', ['False','True'])
        difficulty_walking = st.selectbox('Had Difficulity to Walk', ['False','True'])
    
    input_data = pd.DataFrame({
        'AgeCategory': [age_category],
        'Sex': [sex],
        'HeightInMeters': [height],
        'WeightInKilograms': [weight],
        'GeneralHealth': [general_health],
        'SmokerStatus': [smoker_status],
        'AlcoholDrinkers': [alcohol_drinkers],
        'ECigaretteUsage': [e_cigarette_usage],
        'HadHeartAttack': [had_heart_attack],
        'HadCOPD': [had_copd],
        'HadDiabetes': [had_diabetes],
        'HadKidneyDisease': [had_kidney_disease],
        'HadArthritis': [had_arthritis],
        'HadAngina': [had_angina],
        'PneumoVaxEver': [pneumo_vax_ever],
        'DifficultyWalking':[difficulty_walking],
    })
    
    input_data['AgeCategory'] = input_data['AgeCategory'].map(age_category_mapping)
    input_data['Sex'] = input_data['Sex'].map(sex_mapping)
    input_data['GeneralHealth'] = input_data['GeneralHealth'].map(general_health_mapping)
    input_data['SmokerStatus'] = input_data['SmokerStatus'].map(smoker_status_mapping)
    input_data['ECigaretteUsage'] = input_data['ECigaretteUsage'].map(e_cigarette_usage_mapping)
    input_data['HadHeartAttack'] = input_data['HadHeartAttack'].map(heart_attack_mapping)
    input_data['HadCOPD'] = input_data['HadCOPD'].map(copd_mapping)
    input_data['HadDiabetes'] = input_data['HadDiabetes'].map(diabetes_mapping)
    input_data['HadKidneyDisease'] = input_data['HadKidneyDisease'].map(kidney_disease_mapping)
    input_data['HadArthritis'] = input_data['HadArthritis'].map(arthritis_mapping)
    input_data['HadAngina'] = input_data['HadAngina'].map(angina_mapping)
    input_data['PneumoVaxEver'] = input_data['PneumoVaxEver'].map(pneumo_vax_mapping)
    input_data['DifficultyWalking'] = input_data['DifficultyWalking'].map(difficult_walk_mapping)
    
    # Predict button
    st.markdown("---")
    if st.button("Predict Stroke Risk"):
        rf_prediction = stroke_model.predict(input_data)
        result = 'High risk of having Stroke.' if rf_prediction[0] == 1 else 'Low risk of having Stroke.'
        # Display prediction result with custom styling
        if rf_prediction[0] == 1:
            st.error(f"Prediction Result: {result}")
        else:
            st.success(f"Prediction Result: {result}")