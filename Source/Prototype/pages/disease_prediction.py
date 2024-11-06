# disease_prediction.py
import streamlit as st
import joblib
import pandas as pd


# st.title("Disease Prediction")
# st.write("## This page will provide options for disease prediction models.")

# Sidebar navigation
page = st.sidebar.radio("Go to:", ("Home", "Heart Attack", "Angina", "Stroke", "Depressive Disorder", "Arthritis", "Skin Cancer"))

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

skin_type_mapping = {
    'Type I-II' : 0, 
    'Type IV-VI': 1, 
    'Type III': 2
}
sunlight_exposure_type_mapping = {
    'High': 0, 
    'Low': 1, 
    'Extreme': 2, 
    'Very High': 3, 
    'Moderate': 4
}

if page == "Home":
    st.title("Disease Prediction")
    tab1, tab2 = st.tabs(["Description", "Machine Learning Model"])
    with tab1:
        st.header("Project Overview")
        
        st.write("""
        Welcome to the **Disease Prediction** platform, a comprehensive application designed to leverage the power of machine learning 
        to assess health data and provide early insights into the potential risk of various diseases. This tool aims to empower users—whether they are health professionals, individuals 
        monitoring their health, or those with a general interest in disease prevention—to make informed decisions about their health based on predictive insights.
        """)

        st.write("### Purpose of the Project:")
        st.write("""
        The primary purpose of this platform is to identify and predict the likelihood of several common diseases, helping users understand their health risks 
        through a user-friendly interface. It focuses on several key areas:
        
        - **Awareness**: Increase understanding of personal health risks through data-driven insights.
        - **Prevention**: Enable proactive measures to mitigate risks associated with these diseases.
        - **Empowerment**: Provide users with knowledge and tools to take charge of their health decisions.
        """)
        
        st.write("### Supported Diseases:")
        st.write("""
        The diseases currently supported by this tool include:
        
        - **Heart Attack**: Predict the likelihood of a heart attack based on personal, lifestyle, and medical history.
        - **Angina**: Assess the risk of angina, a condition marked by chest pain due to reduced blood flow to the heart.
        - **Stroke**: Analyze the potential risk factors associated with a stroke, including lifestyle habits and pre-existing conditions.
        - **Depressive Disorder**: Gauge the likelihood of depressive disorder based on mental health indicators and lifestyle choices.
        - **Arthritis**: Evaluate the risk of developing arthritis, particularly among individuals with specific lifestyle or genetic predispositions.
        """)
        st.markdown("---")
        
        # Creating two columns for Key Features and How It Works
        col1, col2 = st.columns(2)

        with col1:
            st.write("### Key Features:")
            st.write("""
            - **User-friendly Input**: Our guided form allows users to easily enter their health, lifestyle, and personal details, making the process straightforward even for those without a medical background.
            - **Disease-specific Models**: Each disease has its own tailored machine learning model, trained on relevant health data to provide accurate risk predictions. This specialization enhances the reliability of the insights.
            - **Actionable Insights**: The platform provides clear and understandable risk assessments, enabling users to recognize potential health risks and to take informed preventive measures.
            - **Educational Resources**: Users can access information about the diseases, risk factors, and lifestyle changes that may help reduce their risk, promoting a better understanding of their health.
            - **High Accuracy Rate**: All machine learning models utilized in this platform achieve at least a **70% accuracy rate**, ensuring users receive reliable predictions regarding their health risks.
            """)

        with col2:
            st.write("### How It Works:")
            st.write("""
            1. **Data Input**: Users enter relevant data, including age, health conditions, lifestyle habits, and personal metrics, through an easy-to-navigate input form.
            2. **Model Prediction**: The application employs advanced machine learning models to analyze the input data and predict the likelihood of each disease. These models utilize extensive datasets to ensure accurate predictions.
            3. **Interpret Results**: Users receive a clear assessment of their health risks, which they can use to make informed decisions about their lifestyle and health management.
            """)
        # st.write("### Key Features:")
        # st.write("""
        # - **User-friendly Input**: Our guided form allows users to easily enter their health, lifestyle, and personal details, making the process straightforward even for those without a medical background.
        # - **Disease-specific Models**: Each disease has its own tailored machine learning model, trained on relevant health data to provide accurate risk predictions. This specialization enhances the reliability of the insights.
        # - **Actionable Insights**: The platform provides clear and understandable risk assessments, enabling users to recognize potential health risks and to take informed preventive measures.
        # - **Educational Resources**: Users can access information about the diseases, risk factors, and lifestyle changes that may help reduce their risk, promoting a better understanding of their health.
        # """)

        # st.write("### How It Works:")
        # st.write("""
        # 1. **Data Input**: Users enter relevant data, including age, health conditions, lifestyle habits, and personal metrics, through an easy-to-navigate input form.
        # 2. **Model Prediction**: The application employs advanced machine learning models to analyze the input data and predict the likelihood of each disease. These models utilize extensive datasets to ensure accurate predictions.
        # 3. **Interpret Results**: Users receive a clear assessment of their health risks, which they can use to make informed decisions about their lifestyle and health management.
        # 4. **Personalized Recommendations**: Based on the assessment results, users may receive suggestions for lifestyle modifications or healthcare follow-ups, further promoting proactive health management.
        # """)
        
        st.markdown("---")
        st.write("### Important Note:")
        st.write("""
        This tool is designed for informational purposes only and should not replace professional medical advice or diagnosis. 
        We strongly encourage users to consult with a qualified healthcare provider for any personal health concerns or medical advice. 
        Understanding your health risks is essential, but it is equally important to have professional guidance tailored to your unique health situation.
        """)
        
        # st.markdown("""
        # Welcome to the **Disease Prediction** platform. This application leverages machine learning to assess health data and provide early 
        # insights into the potential risk of several diseases. The goal is to enable users to monitor their health metrics and take preventive actions based on predictive insights. The diseases currently supported by this tool include:

        # - **Heart Attack**: Predict the likelihood of a heart attack based on personal, lifestyle, and medical history.
        # - **Angina**: Assess the risk of angina, a condition marked by chest pain due to reduced blood flow to the heart.
        # - **Stroke**: Analyze the potential risk factors associated with a stroke.
        # - **Depressive Disorder**: Gauge the likelihood of depressive disorder based on mental health and lifestyle indicators.
        # - **Arthritis**: Evaluate the risk of developing arthritis, particularly among individuals with certain lifestyle or genetic factors.

        # ### Key Features:
        # - **User-friendly Input**: Easily enter your health, lifestyle, and personal details through a guided form.
        # - **Disease-specific Models**: Each disease has its own tailored machine learning model, trained on relevant health data to provide accurate risk predictions.
        # - **Actionable Insights**: Understand potential health risks and gain insights into preventive measures.

        # ### How It Works:
        # - **Data Input**: Enter data such as age, health conditions, lifestyle habits, and relevant personal metrics.
        # - **Model Prediction**: Our machine learning models analyze your input to predict the likelihood of each disease.
        # - **Interpret Results**: Use the risk assessment results to better understand health risks and to take early preventive actions if needed.

        # ### Important Note:
        # This tool is for informational purposes only and does not replace professional medical advice. Please consult a healthcare provider for 
        # personal health concerns or medical advice.
        # """)
        
    with tab2:
        st.header("Model Performance Comparison for Each Disease")
        
        # Explanation of model choice
        st.write("### Model Information")
        st.write("""
        In this analysis, we used two machine learning models: **XGBoost** and **Random Forest**. These models were chosen due to their robustness in handling large datasets, 
        their ability to capture complex patterns, and their strong performance in classification tasks.
        """)
        col1, col2 = st.columns(2)
        with col1:
            st.write("""
                     **XGBoost (Extreme Gradient Boosting)**: 

                     This model is a gradient-boosting algorithm that builds an ensemble of weak decision trees to improve accuracy. It optimizes model performance by minimizing the error using gradient descent techniques and handles both missing data and large-scale datasets efficiently.
                     """)
        with col2:
            st.write("""
                        **Random Forest**: 
                        
                        This model is an ensemble of multiple decision trees, where each tree is trained on a subset of the data. It works by averaging the predictions of individual trees to reduce overfitting and increase accuracy. Random Forest is particularly effective in handling nonlinear relationships and reduces the risk of high variance in the model predictions.
                        """)
        # st.write("""
        # In this analysis, we used two machine learning models: **XGBoost** and **Random Forest**. These models were chosen due to their robustness in handling large datasets, 
        # their ability to capture complex patterns, and their strong performance in classification tasks. 
        
        # **XGBoost (Extreme Gradient Boosting)**: This model is a gradient-boosting algorithm that builds an ensemble of weak decision trees to improve accuracy. 
        # It optimizes model performance by minimizing the error using gradient descent techniques and handles both missing data and large-scale datasets efficiently.
        
        # **Random Forest**: This model is an ensemble of multiple decision trees, where each tree is trained on a subset of the data. It works by averaging the predictions of 
        # individual trees to reduce overfitting and increase accuracy. Random Forest is particularly effective in handling nonlinear relationships and reduces the risk of high 
        # variance in the model predictions.
        # """)
        st.markdown("---")
        st.write("### Model Formulas and Structure")
        # Creating two columns for Key Features and How It Works
        col1, col2 = st.columns(2)
        with col1:
            st.write("**XGBoost Structure:** Utilizes a series of decision trees where each tree is added in sequence to correct errors from the previous ones. The final prediction is the sum of the predictions from all trees:")
            st.latex(r"f(x) = \sum_{i=1}^{K} \theta_i \times \text{tree}_i(x)")
        
        with col2:
            st.write("**Random Forest Structure:** Utilizes multiple decision trees created from random subsets of the data. Each tree provides a classification, and the forest's output is the mode of these classifications:")
            st.latex(r"RF(x) = \text{mode}(\text{tree}_1(x), \text{tree}_2(x), \dots, \text{tree}_n(x))")

        st.markdown("---")
        st.write("### Performance Metrics and Formulas")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Accuracy**: The percentage of correct predictions out of total predictions.")
            st.latex(r"Accuracy = \frac{\text{True Positives} + \text{True Negatives}}{\text{Total Samples}}")
            st.write(" ")
            st.write("**Precision**: The proportion of true positive predictions out of total positive predictions, assessing the model’s specificity.")
            st.latex(r"Precision = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}")

        with col2:
            st.write("**Recall**: The proportion of true positives identified by the model out of all actual positives, showing sensitivity.")
            st.latex(r"Recall = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}")
            st.write(" ")
            st.write("**F1-Score**: The harmonic mean of precision and recall, balancing both metrics.")
            st.latex(r"F1\text{-}Score = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}")
        st.write(" ")
        # Create a new row for the AUC-ROC metric
        st.write("**AUC-ROC (Area Under the Receiver Operating Characteristic Curve)**: This metric provides a measure of model discrimination capability, with values closer to 1 indicating better performance.")
        st.latex(r"AUC\text{-}ROC = \text{Area Under the ROC Curve}")
        
        st.write("""
        ### Metric Functionality
        - **Accuracy** provides an overall assessment of the model's performance but can be misleading with imbalanced datasets.
        - **Precision** is useful when the cost of false positives is high, while **Recall** is crucial when false negatives are more costly.
        - **F1-Score** helps balance precision and recall in scenarios where both metrics are important.
        - **AUC-ROC** evaluates the model's ability to distinguish between classes, with a higher score indicating better model performance.
        """)        
        st.write(" ")
        st.write("### Model's Metric Performance Comparison")
        # Separate tabs for each disease
        disease_tabs = st.tabs(["Heart Attack", "Angina", "Stroke", "Depressive Disorder", "Arthritis"])
        
        # Performance data (example values, replace with actual results)
        performance_data = {
            "Heart Attack": {
                "XGBoost Accuracy": 0.7966294262450293, "XGBoost Precision": 0.80, "XGBoost Recall": 0.80, "XGBoost F1-Score": 0.80, "XGBoost AUC-ROC": 0.8858786967789665,
                "Random Forest Accuracy": 0.7966294262450293, "Random Forest Precision": 0.80, "Random Forest Recall": 0.80, "Random Forest F1-Score": 0.80, "Random Forest AUC-ROC": 0.8860636487215933
            },
            "Angina": {
                "XGBoost Accuracy": 0.8175132546605096, "XGBoost Precision": 0.82, "XGBoost Recall": 0.82, "XGBoost F1-Score": 0.82, "XGBoost AUC-ROC": 0.9006291593821292,
                "Random Forest Accuracy": 0.8173422267829656, "Random Forest Precision": 0.82, "Random Forest Recall": 0.82, "Random Forest F1-Score": 0.82, "Random Forest AUC-ROC": 0.8984425696080401
            },
            "Stroke": {
                "XGBoost Accuracy": 0.7303625377643505, "XGBoost Precision": 0.73, "XGBoost Recall": 0.73, "XGBoost F1-Score": 0.73, "XGBoost AUC-ROC": 0.8107769461685124,
                "Random Forest Accuracy": 0.7313695871097684, "Random Forest Precision": 0.73, "Random Forest Recall": 0.73, "Random Forest F1-Score": 0.73, "Random Forest AUC-ROC": 0.8094244401849822
            },
            "Depressive Disorder": {
                "XGBoost Accuracy": 0.703921468377729, "XGBoost Precision": 0.71, "XGBoost Recall": 0.70, "XGBoost F1-Score": 0.70, "XGBoost AUC-ROC": 0.7745368868768705,
                "Random Forest Accuracy": 0.7004959353750192, "Random Forest Precision": 0.71, "Random Forest Recall": 0.70, "Random Forest F1-Score": 0.70, "Random Forest AUC-ROC": 0.7722841619839349
            },
            "Arthritis": {
                "XGBoost Accuracy": 0.7192471159684275, "XGBoost Precision": 0.72, "XGBoost Recall": 0.72, "XGBoost F1-Score": 0.72, "XGBoost AUC-ROC": 0.7965055602774131,
                "Random Forest Accuracy": 0.7163023679417122, "Random Forest Precision": 0.72, "Random Forest Recall": 0.72, "Random Forest F1-Score": 0.72, "Random Forest AUC-ROC": 0.7946057947266688
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
                
                st.table(performance_df)
                
                if data["XGBoost AUC-ROC"] > data["Random Forest AUC-ROC"]:
                    best_model = "XGBoost"
                elif data["XGBoost AUC-ROC"] < data["Random Forest AUC-ROC"]:
                    best_model = "Random Forest"
                else:  # If AUC-ROC is the same, compare accuracy
                    best_model = "XGBoost" if data["XGBoost Accuracy"] >= data["Random Forest Accuracy"] else "Random Forest"

                st.write(f"**Best Model for {disease} Prediction:** {best_model}")

        
elif page == "Heart Attack":
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
            
elif page == "Skin Cancer":
    sc_model = joblib.load('Source/Prototype/disease_xgb_skin_cancer.pkl')
    
    # Page title and description
    st.title("Skin Cancer Risk Prediction")
    st.write("Enter the following information to predict the risk of Skin Cancer.")

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
        skin = st.selectbox('Skin Type', ['Type I-II', 'Type IV-VI', 'Type III'])
        height = st.number_input('Height (meters)', min_value=0.0, max_value=2.5, value=1.65)
        weight = st.number_input('Weight (kg)', min_value=0.0, max_value=300.0, value=65.0)
        sunlight_exposure = st.selectbox('Sunlight Exposure', ['High', 'Low', 'Extreme', 'Very High', 'Moderate'])
        temperature_exposure = st.number_input('Average Temperature City (°C)', min_value = -20.0 ,max_value= 56.7, value= 25.0)
    
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
        had_copd = st.selectbox('Had COPD', ['False','True'])
    
    st.subheader("Screening and Vaccination")
    with st.expander("Enter Screening and Vaccination Details"):    
        difficulty_errands = st.selectbox('DifficultyErrands', ['False','True'])
        difficulty_dressing_bathing = st.selectbox('DifficultyDressingBathing', ['False','True'])
        
    input_data = pd.DataFrame({
        'AgeCategory': [age_category],
        'Sex': [sex],
        'skin_type': [skin],
        'HeightInMeters':[height],
        'WeightInKilograms':[weight],
        'GeneralHealth': [general_health],
        'SmokerStatus': [smoker_status],
        'AlcoholDrinkers': [alcohol_drinkers],
        'ECigaretteUsage': [e_cigarette_usage],
        'HadCOPD': [had_copd],
        'DifficultyErrands': [difficulty_errands],
        'DifficultyDressingBathing': [difficulty_dressing_bathing],
        'sunlight_exposure_category': [sunlight_exposure],
        'average_environment_temperature': [temperature_exposure]
    })
    
    input_data['AgeCategory'] = input_data['AgeCategory'].map(age_category_mapping)
    input_data['Sex'] = input_data['Sex'].map(sex_mapping)
    input_data['GeneralHealth'] = input_data['GeneralHealth'].map(general_health_mapping)
    input_data['SmokerStatus'] = input_data['SmokerStatus'].map(smoker_status_mapping)
    input_data['ECigaretteUsage'] = input_data['ECigaretteUsage'].map(e_cigarette_usage_mapping)
    input_data['HadCOPD'] = input_data['HadCOPD'].map(copd_mapping)
    input_data['DifficultyErrands'] = input_data['DifficultyErrands'].map(difficult_errand_mapping)
    input_data['DifficultyDressingBathing'] = input_data['DifficultyDressingBathing'].map(difficult_dressing_mapping)
    input_data['skin_type'] = input_data['skin_type'].map(skin_type_mapping)
    input_data['sunlight_exposure_category'] = input_data['sunlight_exposure_category'].map(sunlight_exposure_type_mapping)
    
    # Predict button
    st.markdown("---")
    if st.button("Predict Skin Cancer Risk"):
        rf_prediction = sc_model.predict(input_data)
        result = 'High risk of having Skin Cancer.' if rf_prediction[0] == 1 else 'Low risk of having Skin Cancer.'
        # Display prediction result with custom styling
        if rf_prediction[0] == 1:
            st.error(f"Prediction Result: {result}")
        else:
            st.success(f"Prediction Result: {result}")