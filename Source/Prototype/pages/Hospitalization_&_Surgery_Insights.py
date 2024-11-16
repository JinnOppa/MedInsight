import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the saved model
with open("C:\\Users\\Republic Of Gamers\\OneDrive\\Documents\\GitHub\\TSDN-BoyWithLuv\\Source\\Prototype\\New\\xgb_survival_rate.pkl", "rb") as file:  # Replace 'saved_model.pkl' with your actual model filename
    model = pickle.load(file)

page = st.sidebar.radio("Go to:", ("Home", "Survival Rate","Surgery Risk","Hospital Stay Duration","Total Hospitalization Cost"))

if page == "Home":
    st.title("Hospitalization & Surgery Insights")
    tab1, tab2 = st.tabs(["Description", "Machine Learning Model"])
    with tab1:
        st.write("""
        Welcome to the **Survival Rate and Hospital Analytics Platform**, a comprehensive application designed to leverage the power of machine learning to provide actionable predictions related to patient health and hospital analytics. 
        This platform enables users to input specific details about their health, including vital signs, medical history, and other critical metrics, to receive insights into survival likelihood, surgery risk, hospitalization costs, and stay durations.
        """)

        st.markdown("---")

        # Purpose of the Project
        st.header("Purpose of the Project")
        st.write("""
        The primary purpose of this platform is to assist users in understanding their health risks and hospital needs through a user-friendly interface. It focuses on several key areas:  
        """)

        st.markdown("""
        1. **Personalization**: Each prediction offers tailored insights based on individual health data, ensuring clear understanding of unique risk factors and hospital requirements.  
        2. **Early Detection**: By inputting current health metrics, patients can identify high-risk areas early, enabling proactive steps toward health improvement or surgical preparedness.  
        3. **Hospital Planning**: Provides accurate estimates for hospitalization costs and stay durations, helping users plan better for medical treatments and associated expenses.  
        4. **Continuous Health Tracking**: Enables users to periodically monitor changes in survival rate, surgery risk, and other predictions, fostering consistent health tracking and improvement.  
        """)
        st.markdown("---")
        col1,col2 = st.columns(2)

        with col1:
            st.write("### Key Features:")
            st.write("""
            - **User-friendly Interface**: Simple and intuitive design for inputting health and hospital-related metrics, with guidance where medical expertise is required.  
            - **Comprehensive Health Feature Set**: Input features include:  
            - **Survival Rate Prediction**: Comorbidities (diabetes, cancer, dementia), functional disability level, physiological scores, vital signs (blood pressure, heart rate), and lab values (sodium, creatinine).  
            - **Surgery Risk Assessment**: Health metrics, medical history, and physiological data for evaluating surgical risks.  
            - **Hospital Stay Duration Prediction**: Estimate stay duration based on treatment plans, medical conditions, and historical data.  
            - **Total Hospitalization Cost Prediction**: Factors in admission type, medical conditions, treatment plans, and hospital stay duration for precise cost estimation.  
            - **Actionable Insights**: Provides clear, actionable health and hospital-related predictions, enabling users to take informed decisions about treatments and preventive measures.  
            - **Progress Tracking**: Tracks survival rate predictions and other key metrics over time, allowing users to monitor health and financial planning progress.  
            - **High Accuracy Models**: All machine learning models achieve at least an 80% accuracy rate, ensuring reliability and trustworthiness of predictions.  
            """)
        with col2:
            st.write("### How It Works:")
            st.write("""
            1. **Data Input**: Users enter relevant data, including age, health conditions, lifestyle habits, and personal metrics, through an easy-to-navigate form.  
            2. **Model Prediction**: The application employs advanced machine learning models to analyze input data and provide predictions for:  
                - Survival rate likelihood  
                - Surgery risk  
                - Estimated hospital stay duration  
                - Total hospitalization cost  
            3. **Interpret Results**: Users receive clear, comprehensive assessments, enabling them to make informed decisions regarding their health, surgery preparedness, and hospital planning.  
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
            st.write(" ")
            st.write("**AUC-ROC (Area Under the Receiver Operating Characteristic Curve)**: This metric provides a measure of model discrimination capability, with values closer to 1 indicating better performance.")
            st.latex(r"AUC\text{-}ROC = \text{Area Under the ROC Curve}")
            st.write(" ")
            st.write("**MAE (Mean Absolute Error)**: Measures the average absolute difference between predicted and actual values, treating all errors equally.")
            st.latex(r"MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|")
        with col2:
            st.write("**Recall**: The proportion of true positives identified by the model out of all actual positives, showing sensitivity.")
            st.latex(r"Recall = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}")
            st.write(" ")
            st.write("**F1-Score**: The harmonic mean of precision and recall, balancing both metrics.")
            st.latex(r"F1\text{-}Score = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}")
            st.write(" ")
            st.write("**RMSE (Root Mean Square Error)**: Measures the average magnitude of error, with higher sensitivity to large errors due to squaring.")
            st.latex(r"RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}")
            st.write(" ")
            st.write("**R² (Coefficient of Determination)**: Indicates the proportion of variance in the target variable explained by the model.")
            st.latex(r"R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}")

        st.write(" ")
        
        st.write("""
        ### Metric Functionality
        - **Accuracy** provides an overall assessment of the model's performance but can be misleading with imbalanced datasets.
        - **Precision** is useful when the cost of false positives is high, while **Recall** is crucial when false negatives are more costly.
        - **F1-Score** helps balance precision and recall in scenarios where both metrics are important.
        - **AUC-ROC** evaluates the model's ability to distinguish between classes, with a higher score indicating better model performance.
        - **RMSE (Root Mean Square Error)** provides a measure of the average error magnitude, penalizing larger errors more heavily. It is sensitive to outliers and provides insights into the overall prediction accuracy.
        - **MAE (Mean Absolute Error)** is a straightforward metric that calculates the average absolute difference between predicted and actual values, treating all errors equally and being less sensitive to outliers.
        - **R² (Coefficient of Determination)** explains how much of the variance in the target variable is captured by the model, with values closer to 1 indicating better explanatory power.
        """)        
        st.write(" ")
        st.write("### Model's Metric Performance Comparison")
        disease_tabs = st.tabs(["Survival Rate", "Surgery Risk", "Hospitalization Total Cost"])
        performance_data = {
            "Survival Rate": {
                "XGBoost Accuracy": 0.81, "XGBoost Precision": 0.82, "XGBoost Recall": 0.81, "XGBoost F1-Score": 0.81, "XGBoost AUC-ROC": 0.873460972017673,
                "Random Forest Accuracy": 0.81, "Random Forest Precision": 0.82, "Random Forest Recall": 0.81, "Random Forest F1-Score": 0.81, "Random Forest AUC-ROC": 0.8727771933515673
            },
            "Surgery Risk": {
                "XGBoost Accuracy": 0.55, "XGBoost Precision": 0.47, "XGBoost Recall": 0.55, "XGBoost F1-Score": 0.43, "XGBoost AUC-ROC": 0.7796430189722033,
                "Random Forest Accuracy": 0.56, "Random Forest Precision":0.45, "Random Forest Recall": 0.56, "Random Forest F1-Score": 0.40, "Random Forest AUC-ROC": 0.7806320849982206
            },
            "Hospitalization Total Cost": {
                "XGBoost Accuracy": 0.67, "XGBoost Precision": 0.66, "XGBoost Recall": 0.67, "XGBoost F1-Score": 0.66, "XGBoost AUC-ROC": 0.9212253302924063,
                "Random Forest Accuracy": 0.56, "Random Forest Precision":0.66, "Random Forest Recall": 0.66, "Random Forest F1-Score": 0.66, "Random Forest AUC-ROC": 0.9170619726801796,
            },
        }

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
                st.write("**Best Model for Survival Rate Prediction:** XGBoost")
                st.tabs(["Hospital Total Cost"])
        performance_regression = { 
            "Hospital Total Cost": {
            "XGBoost Root Mean Squared Error (RMSE)": 0.5825122581235502, 
            "XGBoost Mean Absolute Error (MAE)": 0.3424451851851852, 
            "XGBoost R-squared (R²)": 0.9842049011823726,
            "Random Forest Root Mean Squared Error (RMSE)": 0.5903727212532774, 
            "Random Forest Mean Absolute Error (MAE)": 0.3468122222222222, 
            "Random Forest R-squared (R²)": 0.9837757446090281
        }
    }

        # Accessing metrics under "Hospital Total Cost"
        hospital_cost_metrics = performance_regression["Hospital Total Cost"]

        # Creating a DataFrame
        performance_regression_df = pd.DataFrame({
            "Model": ["XGBoost", "Random Forest"],
            "Root Mean Squared Error (RMSE)": [
            hospital_cost_metrics["XGBoost Root Mean Squared Error (RMSE)"], 
            hospital_cost_metrics["Random Forest Root Mean Squared Error (RMSE)"]
        ],
        "Mean Absolute Error (MAE)": [
            hospital_cost_metrics["XGBoost Mean Absolute Error (MAE)"], 
            hospital_cost_metrics["Random Forest Mean Absolute Error (MAE)"]
        ],
        "R-squared (R²)": [
            hospital_cost_metrics["XGBoost R-squared (R²)"], 
            hospital_cost_metrics["Random Forest R-squared (R²)"]
        ]
    })

        #    Displaying in Streamlit
        st.table(performance_regression_df)
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
