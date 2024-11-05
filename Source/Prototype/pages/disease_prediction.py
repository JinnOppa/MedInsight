# disease_prediction.py
import streamlit as st
import joblib
import pandas as pd


# st.title("Disease Prediction")
# st.write("## This page will provide options for disease prediction models.")

# Sidebar navigation
page = st.sidebar.radio("Go to:", ("Home", "Heart Attack", "Angina", "Stroke", "Depressive Disorder", "Arthritis"))

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

asthma_mapping = {
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

difficult_concentrate_mapping = {
    'False': 0,
    'True': 1
}

difficult_errand_mapping = {
    'False': 0,
    'True': 1
}

difficult_dressing_mapping = {
    'False': 0,
    'True': 1
}

difficult_blind_vision_mapping = {
    'False': 0,
    'True': 1
}

difficult_deaf_hearing_mapping = {
    'False': 0,
    'True': 1
}

difficult_walk_mapping = {
    'False': 0,
    'True': 1
}

if page == "Home":
    st.title("Disease Prediction")
    tab1, tab2 = st.tabs(["Description", "Machine Learning Model"])
    with tab1:
        st.header("Project Overview")
        
        st.markdown("""
        Welcome to the **Disease Prediction** platform. This application leverages machine learning to assess health data and provide early 
        insights into the potential risk of several diseases. The goal is to enable users to monitor their health metrics and take preventive actions based on predictive insights. The diseases currently supported by this tool include:

        - **Heart Attack**: Predict the likelihood of a heart attack based on personal, lifestyle, and medical history.
        - **Angina**: Assess the risk of angina, a condition marked by chest pain due to reduced blood flow to the heart.
        - **Stroke**: Analyze the potential risk factors associated with a stroke.
        - **Depressive Disorder**: Gauge the likelihood of depressive disorder based on mental health and lifestyle indicators.
        - **Arthritis**: Evaluate the risk of developing arthritis, particularly among individuals with certain lifestyle or genetic factors.

        ### Key Features:
        - **User-friendly Input**: Easily enter your health, lifestyle, and personal details through a guided form.
        - **Disease-specific Models**: Each disease has its own tailored machine learning model, trained on relevant health data to provide accurate risk predictions.
        - **Actionable Insights**: Understand potential health risks and gain insights into preventive measures.

        ### How It Works:
        - **Data Input**: Enter data such as age, health conditions, lifestyle habits, and relevant personal metrics.
        - **Model Prediction**: Our machine learning models analyze your input to predict the likelihood of each disease.
        - **Interpret Results**: Use the risk assessment results to better understand health risks and to take early preventive actions if needed.

        ### Important Note:
        This tool is for informational purposes only and does not replace professional medical advice. Please consult a healthcare provider for 
        personal health concerns or medical advice.
        """)
    with tab2:
        st.header("Model Performance Comparison for Each Disease")

                # Explanation of model choice
        st.markdown("""
        ### Model Choice and Justification
        In this analysis, we used two machine learning models: **XGBoost** and **Random Forest**. These models were chosen due to their robustness in handling large datasets, their ability to capture complex patterns, and their strong performance in classification tasks. 
        
        - **XGBoost (Extreme Gradient Boosting)**: This model is a gradient-boosting algorithm that builds an ensemble of weak decision trees to improve accuracy. It optimizes model performance by minimizing the error using gradient descent techniques and handles both missing data and large-scale datasets efficiently.
        
        - **Random Forest**: This model is an ensemble of multiple decision trees, where each tree is trained on a subset of the data. It works by averaging the predictions of individual trees to reduce overfitting and increase accuracy. Random Forest is particularly effective in handling nonlinear relationships and reduces the risk of high variance in the model predictions.

        The comparison between these models will allow us to identify which model better predicts the likelihood of each disease.

        ### Model Formulas and Structure

        - **XGBoost Structure**: Utilizes a series of decision trees where each tree is added in sequence to correct errors from the previous ones. The final prediction is the sum of the predictions from all trees:
          \n$$ f(x) = \\sum_{i=1}^{K} \\theta_i \\times tree_i(x) $$
          where \( \\theta_i \) is the weight assigned to each tree.
          
        - **Random Forest Structure**: Utilizes multiple decision trees created from random subsets of the data. Each tree provides a classification, and the forest's output is the mode of these classifications:
          \n$$ RF(x) = mode(tree_1(x), tree_2(x), ..., tree_n(x)) $$

        ### Performance Metrics and Their Formulas
        Each model's effectiveness is evaluated using several key metrics:
        
        - **Accuracy**: The percentage of correct predictions out of total predictions:
          \n$$ Accuracy = \\frac{True\\ Positives + True\\ Negatives}{Total\\ Samples} $$

        - **Precision**: The proportion of true positive predictions out of total positive predictions, assessing the model’s specificity:
          \n$$ Precision = \\frac{True\\ Positives}{True\\ Positives + False\\ Positives} $$

        - **Recall**: The proportion of true positives identified by the model out of all actual positives, showing sensitivity:
          \n$$ Recall = \\frac{True\\ Positives}{True\\ Positives + False\\ Negatives} $$

        - **F1-Score**: The harmonic mean of precision and recall, balancing both metrics:
          \n$$ F1-Score = 2 \\times \\frac{Precision \\times Recall}{Precision + Recall} $$

        - **AUC-ROC (Area Under the Receiver Operating Characteristic Curve)**: This metric provides a measure of model discrimination capability, with values closer to 1 indicating better performance.

        ### Insights from Each Metric
        - **Accuracy** provides an overall assessment of the model's performance but can be misleading with imbalanced datasets.
        - **Precision** is useful when the cost of false positives is high, while **Recall** is crucial when false negatives are more costly.
        - **F1-Score** helps balance precision and recall in scenarios where both metrics are important.
        - **AUC-ROC** evaluates the model's ability to distinguish between classes, with a higher score indicating better model performance.

        Below is a comparison of the performance metrics for **XGBoost** and **Random Forest** across five diseases:
        """)
        
        # Separate tabs for each disease
        disease_tabs = st.tabs(["Heart Attack", "Angina", "Stroke", "Depressive Disorder", "Arthritis"])
        
        # Performance data (example values, replace with actual results)
        performance_data = {
            "Heart Attack": {
                "XGBoost Accuracy": 0.92, "XGBoost Precision": 0.91, "XGBoost Recall": 0.90, "XGBoost F1-Score": 0.90, "XGBoost AUC-ROC": 0.93,
                "Random Forest Accuracy": 0.88, "Random Forest Precision": 0.87, "Random Forest Recall": 0.86, "Random Forest F1-Score": 0.86, "Random Forest AUC-ROC": 0.89
            },
            "Angina": {
                "XGBoost Accuracy": 0.89, "XGBoost Precision": 0.85, "XGBoost Recall": 0.86, "XGBoost F1-Score": 0.85, "XGBoost AUC-ROC": 0.91,
                "Random Forest Accuracy": 0.84, "Random Forest Precision": 0.80, "Random Forest Recall": 0.81, "Random Forest F1-Score": 0.80, "Random Forest AUC-ROC": 0.87
            },
            "Stroke": {
                "XGBoost Accuracy": 0.91, "XGBoost Precision": 0.90, "XGBoost Recall": 0.88, "XGBoost F1-Score": 0.89, "XGBoost AUC-ROC": 0.92,
                "Random Forest Accuracy": 0.87, "Random Forest Precision": 0.86, "Random Forest Recall": 0.84, "Random Forest F1-Score": 0.85, "Random Forest AUC-ROC": 0.88
            },
            "Depressive Disorder": {
                "XGBoost Accuracy": 0.88, "XGBoost Precision": 0.87, "XGBoost Recall": 0.84, "XGBoost F1-Score": 0.85, "XGBoost AUC-ROC": 0.89,
                "Random Forest Accuracy": 0.83, "Random Forest Precision": 0.82, "Random Forest Recall": 0.80, "Random Forest F1-Score": 0.81, "Random Forest AUC-ROC": 0.85
            },
            "Arthritis": {
                "XGBoost Accuracy": 0.90, "XGBoost Precision": 0.89, "XGBoost Recall": 0.88, "XGBoost F1-Score": 0.88, "XGBoost AUC-ROC": 0.91,
                "Random Forest Accuracy": 0.85, "Random Forest Precision": 0.83, "Random Forest Recall": 0.82, "Random Forest F1-Score": 0.82, "Random Forest AUC-ROC": 0.86
            }
        }
        
        # Displaying each disease’s performance metrics in its own tab
        for i, disease in enumerate(performance_data.keys()):
            with disease_tabs[i]:
                st.subheader(f"{disease} - Model Performance Comparison")
                # Extracting the model data for this specific disease
                data = performance_data[disease]
                # Creating a DataFrame for better readability
                performance_df = pd.DataFrame({
                    "Model": ["XGBoost", "Random Forest"],
                    "Accuracy": [data["XGBoost Accuracy"], data["Random Forest Accuracy"]],
                    "Precision": [data["XGBoost Precision"], data["Random Forest Precision"]],
                    "Recall": [data["XGBoost Recall"], data["Random Forest Recall"]],
                    "F1-Score": [data["XGBoost F1-Score"], data["Random Forest F1-Score"]],
                    "AUC-ROC": [data["XGBoost AUC-ROC"], data["Random Forest AUC-ROC"]]
                })
                
                # Displaying the table
                st.table(performance_df)
        
    
elif page == "Heart Attack":

    # Load the model
    # ha_model = joblib.load(r'C:\Users\user\OneDrive\Documents\GitHub\TSDN-BoyWithLuv\Source\Prototype\disease_rf_heart_attack.pkl')
    ha_model = joblib.load('Source/Prototype/disease_rf_heart_attack.pkl')

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
        result = 'High risk of Heart Attack.' if rf_prediction[0] == 1 else 'Low risk of Heart Attack.'
        # Display prediction result with custom styling
        if rf_prediction[0] == 1:
            st.error(f"Prediction Result: {result}")
        else:
            st.success(f"Prediction Result: {result}")

elif page == "Angina":
    
    # Load the model
    # angina_model = joblib.load(r'C:\Users\user\OneDrive\Documents\GitHub\TSDN-BoyWithLuv\Source\Prototype\disease_xgb_angina.pkl')
    angina_model = joblib.load('Source/Prototype/disease_xgb_angina.pkl')

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
    # stroke_model = joblib.load(r'C:\Users\user\OneDrive\Documents\GitHub\TSDN-BoyWithLuv\Source\Prototype\disease_rf_stroke.pkl')
    stroke_model = joblib.load('Source/Prototype/disease_xgb_stroke.pkl')

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
            
elif page == "Depressive Disorder":
    # Load the best Random Forest model
    # dd_model = joblib.load(r'C:\Users\user\OneDrive\Documents\GitHub\TSDN-BoyWithLuv\Source\Prototype\disease_xgb_depressive_disorder.pkl')
    dd_model = joblib.load('Source/Prototype/disease_xgb_depressive_disorder.pkl')
    
    # Page title and description
    st.title("Depressive Disorder Risk Prediction")
    st.write("Enter the following information to predict the risk of Depressive Disorder.")

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
        had_stroke  = st.selectbox('Had Stroke', ['False','True'])
        had_copd = st.selectbox('Had COPD', ['False','True'])
        had_asthma = st.selectbox('Had Asthma', ['False','True'])
        had_arthritis = st.selectbox('Had Arthritis', ['False','True'])
    
    st.subheader("Screening and Vaccination")
    with st.expander("Enter Screening and Vaccination Details"):    
        difficulty_concentrating = st.selectbox('DifficultyConcentrating', ['False','True'])
        difficulty_errands = st.selectbox('DifficultyErrands', ['False','True'])
        difficulty_dressing_bathing = st.selectbox('DifficultyDressingBathing', ['False','True'])
        blind_or_vision_difficulty = st.selectbox('BlindOrVisionDifficulty', ['False','True'])
        deaf_or_hard_of_hearing = st.selectbox('DeafOrHardOfHearing', ['False','True'])
        difficulty_walking = st.selectbox('Had Difficulity to Walk', ['False','True'])
        
    input_data = pd.DataFrame({
        'AgeCategory': [age_category],
        'Sex': [sex],
        'GeneralHealth': [general_health],
        'SmokerStatus': [smoker_status],
        'AlcoholDrinkers': [alcohol_drinkers],
        'ECigaretteUsage': [e_cigarette_usage],
        'HadHeartAttack': [had_heart_attack],
        'HadStroke': [had_stroke],
        'HadCOPD': [had_copd],
        'HadAsthma': [had_asthma],
        'HadArthritis': [had_arthritis],
        'DifficultyConcentrating': [difficulty_concentrating],
        'DifficultyErrands': [difficulty_errands],
        'DifficultyDressingBathing': [difficulty_dressing_bathing],
        'BlindOrVisionDifficulty': [blind_or_vision_difficulty],
        'DeafOrHardOfHearing': [deaf_or_hard_of_hearing],
        'DifficultyWalking':[difficulty_walking],
    })
    
    input_data['AgeCategory'] = input_data['AgeCategory'].map(age_category_mapping)
    input_data['Sex'] = input_data['Sex'].map(sex_mapping)
    input_data['GeneralHealth'] = input_data['GeneralHealth'].map(general_health_mapping)
    input_data['SmokerStatus'] = input_data['SmokerStatus'].map(smoker_status_mapping)
    input_data['ECigaretteUsage'] = input_data['ECigaretteUsage'].map(e_cigarette_usage_mapping)
    input_data['HadHeartAttack'] = input_data['HadHeartAttack'].map(heart_attack_mapping)
    input_data['HadStroke'] = input_data['HadStroke'].map(stroke_mapping)
    input_data['HadCOPD'] = input_data['HadCOPD'].map(copd_mapping)
    input_data['HadAsthma'] = input_data['HadAsthma'].map(asthma_mapping)
    input_data['HadArthritis'] = input_data['HadArthritis'].map(arthritis_mapping)
    input_data['DifficultyConcentrating'] = input_data['DifficultyConcentrating'].map(difficult_concentrate_mapping)
    input_data['DifficultyErrands'] = input_data['DifficultyErrands'].map(difficult_errand_mapping)
    input_data['DifficultyDressingBathing'] = input_data['DifficultyDressingBathing'].map(difficult_dressing_mapping)
    input_data['BlindOrVisionDifficulty'] = input_data['BlindOrVisionDifficulty'].map(difficult_blind_vision_mapping)
    input_data['DeafOrHardOfHearing'] = input_data['DeafOrHardOfHearing'].map(difficult_deaf_hearing_mapping)
    input_data['DifficultyWalking'] = input_data['DifficultyWalking'].map(difficult_walk_mapping)
    
    # Predict button
    st.markdown("---")
    if st.button("Predict Depressive Disorder Risk"):
        rf_prediction = dd_model.predict(input_data)
        result = 'High risk of having Depressive Disorder.' if rf_prediction[0] == 1 else 'Low risk of having Depressive Disorder.'
        # Display prediction result with custom styling
        if rf_prediction[0] == 1:
            st.error(f"Prediction Result: {result}")
        else:
            st.success(f"Prediction Result: {result}")
                     
elif page == "Arthritis":
    # Load the best Random Forest model
    # arthritis_model = joblib.load(r'C:\Users\user\OneDrive\Documents\GitHub\TSDN-BoyWithLuv\Source\Prototype\disease_xgb_arthritis.pkl')
    arthritis_model = joblib.load('Source/Prototype/disease_xgb_arthritis.pkl')

    
    # Page title and description
    st.title("Arthritis Risk Prediction")
    st.write("Enter the following information to predict the risk of Arthritis.")
    
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
        had_diabetes = st.selectbox('Had Diabetes', ['No', 'Yes', 'No, pre-diabetes or borderline diabetes', 
                                                    'Yes, but only during pregnancy (female)'])
        had_kidney_disease = st.selectbox('Had Kidney Disease', ['False','True'])

    st.subheader("Screening and Vaccination")
    with st.expander("Enter Screening and Vaccination Details"):
        difficulty_dressing_bathing = st.selectbox('DifficultyDressingBathing', ['False','True'])
        difficulty_errands = st.selectbox('DifficultyErrands', ['False','True'])
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
        'HadDiabetes': [had_diabetes],
        'HadKidneyDisease': [had_kidney_disease],
        'DifficultyDressingBathing': [difficulty_dressing_bathing],
        'DifficultyErrands': [difficulty_errands],
        'DifficultyWalking': [difficulty_walking]
    })
    
    input_data['AgeCategory'] = input_data['AgeCategory'].map(age_category_mapping)
    input_data['Sex'] = input_data['Sex'].map(sex_mapping)
    input_data['GeneralHealth'] = input_data['GeneralHealth'].map(general_health_mapping)
    input_data['SmokerStatus'] = input_data['SmokerStatus'].map(smoker_status_mapping)
    input_data['ECigaretteUsage'] = input_data['ECigaretteUsage'].map(e_cigarette_usage_mapping)
    input_data['HadDiabetes'] = input_data['HadDiabetes'].map(diabetes_mapping)
    input_data['HadKidneyDisease'] = input_data['HadKidneyDisease'].map(kidney_disease_mapping)
    input_data['DifficultyDressingBathing'] = input_data['DifficultyDressingBathing'].map(difficult_dressing_mapping)
    input_data['DifficultyErrands'] = input_data['DifficultyErrands'].map(difficult_errand_mapping)
    input_data['DifficultyWalking'] = input_data['DifficultyWalking'].map(difficult_walk_mapping)

    # Predict button
    st.markdown("---")
    if st.button("Predict Arthritis Risk"):
        rf_prediction = arthritis_model.predict(input_data)
        result = 'High risk of having Arthritis.' if rf_prediction[0] == 1 else 'Low risk of having Arthritis.'
        # Display prediction result with custom styling
        if rf_prediction[0] == 1:
            st.error(f"Prediction Result: {result}")
        else:
            st.success(f"Prediction Result: {result}")