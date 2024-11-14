import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the saved model
with open("C:\\Users\\Republic Of Gamers\\OneDrive\\Documents\\GitHub\\TSDN-BoyWithLuv\\Source\\Prototype\\New\\xgb_survival_rate.pkl", "rb") as file:  # Replace 'saved_model.pkl' with your actual model filename
    model = pickle.load(file)

page = st.sidebar.radio("Go to:", ("Home", "Survival Rate"))

if page == "Home":
    st.title("Survival Rate Prediction")
    tab1, tab2 = st.tabs(["Description", "Machine Learning Model"])
    with tab1:
        st.header("Project Overview")
        
        st.write("""
        Welcome to the Survival Rate platform, a comprehensive application designed to leverage 
        the power of machine learning to predict survival rates based on various medical and physiological factors. 
        This platform enables users to input specific details about their health, including vital signs, medical history, and other critical metrics.
        """)

        st.write("### Purpose of the Project:")
        st.write("""
        The primary purpose of this platform is to identify and predict the survival likelihood of a patient that is in critical condition, helping users understand their health risks 
        through a user-friendly interface. It focuses on several key areas:
        
        - **Personalization**: Each prediction provides patients with tailored insights based on their unique health data, offering a clear understanding of individual risk factors.
        - **Early Detection**: By inputting their current health metrics, patients can identify high-risk areas early, enabling them to take proactive steps toward improving their health.
        - **Continuous Health Tracking**: Patients can use the platform periodically to track changes in their survival rate predictions over time, providing motivation for consistent health monitoring and improvement.
        """)
        st.markdown("---")

        col1,col2 = st.columns(2)

        with col1:
            st.write("### Key Features:")
            st.write("""
            - **User-friendly Interface**: Simple interface is provided but some of the details that are needed for prediction might requires further assist or information from medical professionals.
            - **Comprehensive Health Feature Set**: Key input features include comorbidities (diabetes, cancer, dementia), functional disability level, physiological scores, vital signs (blood pressure, heart rate), and lab values (sodium, creatinine), ensuring robust and tailored predictions.
            - **Actionable Insights**: The platform provides clear and understandable risk assessments, enabling users to recognize potential health risks and to take informed preventive measures.
            - **Progress Tracking**: Users can track their survival rate predictions over time, seeing how changes in their health metrics impact their prognosis and receiving encouragement for healthy behavior changes.
            - **High Accuracy Rate**: All machine learning models utilized in this platform achieve at least a **80% accuracy rate**, ensuring users receive reliable predictions regarding their health risks.
            """)
        with col2:
            st.write("### How It Works:")
            st.write("""
            1. **Data Input**: Users enter relevant data, including age, health conditions, lifestyle habits, and personal metrics, through an easy-to-navigate input form.
            2. **Model Prediction**: The application employs advanced machine learning models to analyze the input data and predict the likelihood of survival. These models utilize extensive datasets to ensure accurate predictions.
            3. **Interpret Results**: Users receive a clear assessment of their health risks, which they can use to make informed decisions regarding the patient.
            """)
        
        st.markdown("---")
        st.write("### Important Note:")
        st.write("""
        This tool is designed for informational purposes only and should not replace professional medical advice or diagnosis. 
        We strongly encourage users to consult with a qualified healthcare provider for any personal health concerns or medical advice. 
        Understanding your health risks is essential, but it is equally important to have professional guidance tailored to your unique health situation.
        """)
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
        performance_data = {
            "Survival Rate": {
                "XGBoost Accuracy": 0.81, "XGBoost Precision": 0.82, "XGBoost Recall": 0.81, "XGBoost F1-Score": 0.81, "XGBoost AUC-ROC": 0.873460972017673,
                "Random Forest Accuracy": 0.81, "Random Forest Precision": 0.82, "Random Forest Recall": 0.81, "Random Forest F1-Score": 0.81, "Random Forest AUC-ROC": 0.8727771933515673
            }
        }

        st.subheader("Survival Rate - Model Performance Comparison")
        performance_df = pd.DataFrame({
            "Model": ["XGBoost", "Random Forest"],
            "Accuracy": [performance_data["Survival Rate"]["XGBoost Accuracy"], performance_data["Survival Rate"]["Random Forest Accuracy"]],
            "Precision": [performance_data["Survival Rate"]["XGBoost Precision"], performance_data["Survival Rate"]["Random Forest Precision"]],
            "Recall": [performance_data["Survival Rate"]["XGBoost Recall"], performance_data["Survival Rate"]["Random Forest Recall"]],
            "F1-Score": [performance_data["Survival Rate"]["XGBoost F1-Score"], performance_data["Survival Rate"]["Random Forest F1-Score"]],
            "AUC-ROC": [performance_data["Survival Rate"]["XGBoost AUC-ROC"], performance_data["Survival Rate"]["Random Forest AUC-ROC"]]
        })

        st.table(performance_df)
        st.write("**Best Model for Survival Rate Prediction:** XGBoost")
            
elif page == "Survival Rate":
    # Title and description
    st.title("Survival Rate Prediction")
    st.write("Enter the details below to predict the survival rate based on patient data.")

    # Input fields
    age_years = st.number_input("Age (years)", min_value=0, max_value=120, value=30)
    gender = st.selectbox("Gender", options=["Male", "Female"])
    num_comorbidities = st.number_input("Number of Comorbidities", min_value=0, max_value=10, value=0)
    has_diabetes = st.selectbox("Has Diabetes", options=["Yes", "No"])
    has_dementia = st.selectbox("Has Dementia", options=["Yes", "No"])
    cancer_status = st.selectbox("Cancer Status", options=["Yes", "No"])
    functional_disability_level = st.slider("Functional Disability Level (0-10)", min_value=0, max_value=10, value=5)
    coma_score = st.slider("Coma Score (3-15)", min_value=3, max_value=15, value=10)
    support_physiology_score = st.slider("Support Physiology Score", min_value=0, max_value=20, value=10)
    apache_score = st.slider("APACHE Score", min_value=0, max_value=100, value=50)
    mean_arterial_bp = st.number_input("Mean Arterial Blood Pressure (mmHg)", min_value=0.0, max_value=200.0, value=80.0)
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=0.0, max_value=200.0, value=70.0)
    respiratory_rate = st.number_input("Respiratory Rate (breaths/min)", min_value=0.0, max_value=60.0, value=20.0)
    body_temperature_celsius = st.number_input("Body Temperature (°C)", min_value=30.0, max_value=45.0, value=37.0)
    serum_sodium = st.number_input("Serum Sodium (mmol/L)", min_value=100.0, max_value=200.0, value=140.0)
    serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.0, max_value=20.0, value=1.0)
    do_not_resuscitate_status = st.selectbox("Do Not Resuscitate Status", options=["Yes", "No"])

    # Convert categorical values to numeric encoding if necessary
    gender = 1 if gender == "Male" else 0
    has_diabetes = 1 if has_diabetes == "Yes" else 0
    has_dementia = 1 if has_dementia == "Yes" else 0
    cancer_status = 1 if cancer_status == "Yes" else 0
    do_not_resuscitate_status = 1 if do_not_resuscitate_status == "Yes" else 0

    # Create input DataFrame
    input_data = pd.DataFrame({
        'age_years': [age_years],
        'gender': [gender],
        'num_comorbidities': [num_comorbidities],
        'has_diabetes': [has_diabetes],
        'has_dementia': [has_dementia],
        'cancer_status': [cancer_status],
        'functional_disability_level': [functional_disability_level],
        'coma_score': [coma_score],
        'support_physiology_score': [support_physiology_score],
        'apache_score': [apache_score],
        'mean_arterial_bp': [mean_arterial_bp],
        'heart_rate': [heart_rate],
        'respiratory_rate': [respiratory_rate],
        'body_temperature_celsius': [body_temperature_celsius],
        'serum_sodium': [serum_sodium],
        'serum_creatinine': [serum_creatinine],
        'do_not_resuscitate_status': [do_not_resuscitate_status]
    })

    # Prediction
    if st.button("Predict Survival Rate"):
        prediction = model.predict(input_data)
        survival_rate = prediction[0]
    
        st.subheader("Predicted Survival Rate")
        st.write(f"The predicted survival rate is: {'High' if survival_rate > 0.5 else 'Low'}")
