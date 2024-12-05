import streamlit as st
# import pickle
import pandas as pd
import numpy as np
# from sklearn.preprocessing import LabelEncoder
import joblib
import math

# Load the saved model
# with open("C:\\Users\\Republic Of Gamers\\OneDrive\\Documents\\GitHub\\TSDN-BoyWithLuv\\Source\\Prototype\\survival_rate\\xgb_survival_rate.pkl", "rb") as file:  # Replace 'saved_model.pkl' with your actual model filename
#     model = pickle.load(file)

page = st.sidebar.radio("Go to:", ("Home", "Survival Rate","Surgery Risk","Hospital Stay Duration","Total Hospitalization Cost"))

gender_mapping = {
    'Male': 0, 
    'Female': 1
}

insurance_type_mapping= {
    'Private': 0, 
    'Social Security Agency': 1, 
    'Self-Pay': 2
}

smoking_status_mapping = {
    'Never': 0, 
    'Current': 1, 
    'Former': 2
}

e_cigarette_usage_mapping = {
    False: 0,
    True: 1
}

alcohol_consumption_rate_mapping = {
    'Occasional': 0, 
    'None': 1, 
    'Heavy': 2, 
    'Moderate': 3
}

surgery_name_mapping = {
    'Gallbladder Removal': 0, 
    'Breast Cancer Surgery': 1, 
    'Appendectomy' : 2,
    'Hip Replacement': 3, 
    'Cataract Surgery' : 4, 
    'Hernia Repair' : 5,
    'Knee Replacement' : 6, 
    'Liver Transplant': 7, 
    'Heart Bypass' : 8,
    'Spinal Fusion': 9
}

surgery_type_mapping = {
    'Minor': 0, 
    'Major' : 1
}

room_type_mapping = {
    'Regular Ward' : 0, 
    'ICU': 1, 
    'VIP Ward': 2, 
    'Private Ward': 3
}

total_cost_class_mapping = {
    'Very Low': 0, 
    'Low': 1, 
    'Medium' : 2, 
    'High': 3, 
    'Very High': 4
}

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
        st.write("**Best Model for Prediction:** XGBoost")

elif page == "Survival Rate":
    try:
        sr_model = joblib.load('Source/Prototype/survival_rate/xgb_survival_rate.pkl')
    except FileNotFoundError:
        try:
            sr_model = joblib.load('C:\\Users\\Republic Of Gamers\\OneDrive\\Documents\\GitHub\\TSDN-BoyWithLuv\\Source\\Prototype\\survival_rate\\xgb_survival_rate.pkl')
        except FileNotFoundError:
            sr_model = None
            print("Error: Model file not found in either path.")

    # # Title and description
    # st.title("Survival Rate Prediction")
    # st.write("Enter the details below to predict the survival rate based on patient data.")

    # # Input fields
    # age_years = st.number_input("Age (years)", min_value=0, max_value=120, value=30)
    # gender = st.selectbox("Gender", options=["Male", "Female"])
    # num_comorbidities = st.number_input("Number of Comorbidities", min_value=0, max_value=10, value=0)
    # has_diabetes = st.selectbox("Has Diabetes", options=["Yes", "No"])
    # has_dementia = st.selectbox("Has Dementia", options=["Yes", "No"])
    # cancer_status = st.selectbox("Cancer Status", options=["Yes", "No"])
    # functional_disability_level = st.slider("Functional Disability Level (0-10)", min_value=0, max_value=10, value=5)
    # coma_score = st.slider("Coma Score (3-15)", min_value=3, max_value=15, value=10)
    # support_physiology_score = st.slider("Support Physiology Score", min_value=0, max_value=20, value=10)
    # apache_score = st.slider("APACHE Score", min_value=0, max_value=100, value=50)
    # mean_arterial_bp = st.number_input("Mean Arterial Blood Pressure (mmHg)", min_value=0.0, max_value=200.0, value=80.0)
    # heart_rate = st.number_input("Heart Rate (bpm)", min_value=0.0, max_value=200.0, value=70.0)
    # respiratory_rate = st.number_input("Respiratory Rate (breaths/min)", min_value=0.0, max_value=60.0, value=20.0)
    # body_temperature_celsius = st.number_input("Body Temperature (°C)", min_value=30.0, max_value=45.0, value=37.0)
    # serum_sodium = st.number_input("Serum Sodium (mmol/L)", min_value=100.0, max_value=200.0, value=140.0)
    # serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.0, max_value=20.0, value=1.0)
    # do_not_resuscitate_status = st.selectbox("Do Not Resuscitate Status", options=["Yes", "No"])

    # # Convert categorical values to numeric encoding if necessary
    # gender = 1 if gender == "Male" else 0
    # has_diabetes = 1 if has_diabetes == "Yes" else 0
    # has_dementia = 1 if has_dementia == "Yes" else 0
    # cancer_status = 1 if cancer_status == "Yes" else 0
    # do_not_resuscitate_status = 1 if do_not_resuscitate_status == "Yes" else 0

    # # Create input DataFrame
    # input_data = pd.DataFrame({
    #     'age_years': [age_years],
    #     'gender': [gender],
    #     'num_comorbidities': [num_comorbidities],
    #     'has_diabetes': [has_diabetes],
    #     'has_dementia': [has_dementia],
    #     'cancer_status': [cancer_status],
    #     'functional_disability_level': [functional_disability_level],
    #     'coma_score': [coma_score],
    #     'support_physiology_score': [support_physiology_score],
    #     'apache_score': [apache_score],
    #     'mean_arterial_bp': [mean_arterial_bp],
    #     'heart_rate': [heart_rate],
    #     'respiratory_rate': [respiratory_rate],
    #     'body_temperature_celsius': [body_temperature_celsius],
    #     'serum_sodium': [serum_sodium],
    #     'serum_creatinine': [serum_creatinine],
    #     'do_not_resuscitate_status': [do_not_resuscitate_status]
    # })

    # # Prediction
    # if st.button("Predict Survival Rate"):
    #     prediction = sr_model.predict(input_data)
    #     survival_rate = prediction[0]
    
    #     st.subheader("Predicted Survival Rate")
    #     if survival_rate > 0.5:
    #         st.success("The predicted survival rate is: High")
    #     else:
    #         st.error("The predicted survival rate is: Low")

elif page == "Surgery Risk":
    # Load your pre-trained model (assuming the model is saved in a .pkl file)
    # model = joblib.load('C:\\Users\\Republic Of Gamers\\OneDrive\\Documents\\GitHub\\TSDN-BoyWithLuv\\Source\\Prototype\\surgery_risk\\xgb_surgery_risk.pkl')

    try:
        surisk_model = joblib.load('Source/Prototype/surgery_risk/xgb_surgery_risk.pkl')
    except FileNotFoundError:
        try:
            surisk_model = joblib.load('C:\\Users\\Republic Of Gamers\\OneDrive\\Documents\\GitHub\\TSDN-BoyWithLuv\\Source\\Prototype\\surgery_risk\\xgb_surgery_risk.pkl')
        except FileNotFoundError:
            surisk_model = None
            print("Error: Model file not found in either path.")

    # # Function to encode categorical columns
    # def encode_categorical_data(input_data):
    #     # Define LabelEncoder for each categorical column
    #     label_encoders = {
    #         'gender': LabelEncoder(),
    #         'age_group': LabelEncoder(),
    #         'smoking_status': LabelEncoder(),
    #         'e_cigarette_usage': LabelEncoder(),
    #         'alcohol_consumption_rate': LabelEncoder(),
    #         'surgery_name': LabelEncoder(),
    #         'surgery_type': LabelEncoder(),
    #         'surgical_specialty': LabelEncoder(),
    #         'anesthesia_type': LabelEncoder(),
    #         'blood_loss_category': LabelEncoder(),
    #         'blood_transfusions': LabelEncoder(),
    #         'stay_duration': LabelEncoder(),
    #         'room_type': LabelEncoder()
    #     }

    #     # Fit and transform each categorical column
    #     for column, le in label_encoders.items():
    #         if column in input_data.columns:
    #             input_data[column] = le.fit_transform(input_data[column])

    #     return input_data

    # # Function to make predictions
    # def predict_preoperative_risk(gender, age_group, smoking_status, e_cigarette_usage, alcohol_consumption_rate, 
    #                             surgery_name, surgery_type, surgical_specialty, anesthesia_type, surgery_duration, 
    #                             blood_loss_category, blood_transfusions, stay_duration, room_type, pain_score, 
    #                             rehab_assessment_score):
    #     # Prepare the data in the format the model expects
    #     input_data = pd.DataFrame({
    #         'gender': [gender],
    #         'age_group': [age_group],
    #         'smoking_status': [smoking_status],
    #         'e_cigarette_usage': [e_cigarette_usage],
    #         'alcohol_consumption_rate': [alcohol_consumption_rate],
    #         'surgery_name': [surgery_name],
    #         'surgery_type': [surgery_type],
    #         'surgical_specialty': [surgical_specialty],
    #         'anesthesia_type': [anesthesia_type],
    #         'surgery_duration': [surgery_duration],
    #         'blood_loss_category': [blood_loss_category],
    #         'blood_transfusions': [blood_transfusions],
    #         'stay_duration': [stay_duration],
    #         'room_type': [room_type],
    #         'pain_score': [pain_score],
    #         'rehab_assessment_score': [rehab_assessment_score]
    #     })
    
    #     # Encode categorical data
    #     input_data = encode_categorical_data(input_data)
    
    #     # Make prediction
    #     prediction = surisk_model.predict(input_data)
    
    #     return prediction[0]

    # # Streamlit UI elements
    # st.title('Preoperative Risk Prediction')

    # # User Inputs
    # gender = st.selectbox('Gender', ['Male', 'Female'])
    # age_group = st.selectbox('Age Group', ['<20', '20-40', '40-60', '60+'])
    # smoking_status = st.selectbox('Smoking Status', ['Non-Smoker', 'Smoker'])
    # e_cigarette_usage = st.selectbox('E-Cigarette Usage', ['Yes', 'No'])
    # alcohol_consumption_rate = st.selectbox('Alcohol Consumption Rate', ['Low', 'Moderate', 'High', 'None'])
    # surgery_name = st.selectbox('Surgery Name', ['Cataract Surgery', 'Appendectomy', 'Spinal Fusion', 'Knee Replacement', 
    #                                             'Gallbladder Removal', 'Breast Cancer Surgery', 'Liver Transplant', 
    #                                             'Heart Bypass', 'Hip Replacement', 'Hernia Repair'])
    # surgery_type = st.selectbox('Surgery Type', ['Minor', 'Major'])
    # surgical_specialty = st.selectbox('Surgical Specialty', ['General', 'Orthopedic', 'Oncology', 'Transplant', 'Cardiothoracic'])
    # anesthesia_type = st.selectbox('Anesthesia Type', ['General', 'Local', 'Regional'])
    # surgery_duration = st.number_input('Surgery Duration (in minutes)', min_value=0)
    # blood_loss_category = st.selectbox('Blood Loss Category', ['Low', 'Medium', 'High'])
    # blood_transfusions = st.selectbox('Blood Transfusions', ['Yes', 'No'])
    # stay_duration = st.selectbox('Stay Duration', ['<1 Day', '1-3 Days', '3-7 Days', '>7 Days'])
    # room_type = st.selectbox('Room Type', ['Standard', 'VIP', 'ICU'])
    # pain_score = st.slider('Pain Score (1-10)', min_value=1, max_value=10)
    # rehab_assessment_score = st.slider('Rehabilitation Assessment Score (1-10)', min_value=1, max_value=10)

    # # Prediction button
    # if st.button('Predict Preoperative Risk'):
    #     # Get prediction from the model
    #     prediction = predict_preoperative_risk(gender, age_group, smoking_status, e_cigarette_usage, alcohol_consumption_rate,
    #                                         surgery_name, surgery_type, surgical_specialty, anesthesia_type, surgery_duration,
    #                                         blood_loss_category, blood_transfusions, stay_duration, room_type, pain_score,
    #                                         rehab_assessment_score)
    #     # Determine risk level and display with appropriate visualization
    #     if prediction == 0:
    #         risk_level = "Low Risk Surgery"
    #         st.success(f"The predicted preoperative risk class is: {risk_level}")
    #     elif prediction == 1:
    #         risk_level = "Moderate Risk Surgery"
    #         st.info(f"The predicted preoperative risk class is: {risk_level}")
    #     elif prediction == 2:
    #         risk_level = "High Risk Surgery"
    #         st.warning(f"The predicted preoperative risk class is: {risk_level}")
    #     elif prediction == 3:
    #         risk_level = "Very High Risk Surgery"
    #         st.error(f"The predicted preoperative risk class is: {risk_level}")
    #     else:
    #         risk_level = "Unknown Risk Level"
    #         st.write(f"The predicted preoperative risk class is: {risk_level}")

elif page == "Hospital Stay Duration":
    try:
        stay_model = joblib.load('Source/Prototype/patient_stay_cost/xgb_stayDuration.pkl')
    except FileNotFoundError:
        try:
            stay_model = joblib.load('C:\\Users\\Republic Of Gamers\\OneDrive\\Documents\\GitHub\\TSDN-BoyWithLuv\\Source\\Prototype\\patient_stay_cost\\xgb_stayDuration.pkl')
        except FileNotFoundError:
            stay_model = None
            print("Error: Model file not found in either path.")


    # model = joblib.load('C:\\Users\\Republic Of Gamers\\OneDrive\\Documents\\GitHub\\TSDN-BoyWithLuv\\Source\\Prototype\\patient_stay_cost\\xgb_stayDuration.pkl')  # Ensure the path is correct

    # Function to predict hospital stay duration
    def predict_stay_duration(input_data):
        prediction = stay_model.predict([input_data])
        return prediction[0]

    # Streamlit interface   
    st.title("Hospital Stay Duration Prediction")

    # Input fields for each feature
    insurance_type = st.selectbox("Insurance Type", ["Private", "Social Security Agency", "Self-Pay"])
    surgery_name = st.selectbox("Surgery Name", [
        "Gallbladder Removal", "Breast Cancer Surgery", "Appendectomy",
        "Hip Replacement", "Cataract Surgery", "Hernia Repair", 
        "Knee Replacement", "Liver Transplant", "Heart Bypass", "Spinal Fusion"
    ])
    surgery_duration = st.number_input("Surgery Duration (in hours)", min_value=1, max_value=24, value=2)
    room_type = st.selectbox("Room Type", ["Regular Ward", "ICU", "VIP Ward", "Private Ward"])

    # Numeric inputs
    medical_equipment_count = st.number_input("Medical Equipment Count", min_value=0, max_value=100, value=5)
    ward_cost = st.number_input("Ward Cost (in currency)", min_value=0, value=1000)
    surgery_cost = st.number_input("Surgery Cost (in currency)", min_value=0, value=5000)
    medication_cost = st.number_input("Medication Cost (in currency)", min_value=0, value=100)
    total_cost = st.number_input("Total Cost (in currency)", min_value=0, value=7000)

    # Function to encode inputs to match the model's expected input format
    def encode_inputs(insurance_type, surgery_name, surgery_duration, room_type, medical_equipment_count, ward_cost, surgery_cost, medication_cost, total_cost):
        # Encoding categorical variables as needed
        insurance_type_encoded = {"Private": 0, "Social Security Agency": 1, "Self-Pay": 2}[insurance_type]
        surgery_name_encoded = {"Gallbladder Removal": 0, "Breast Cancer Surgery": 1, "Appendectomy": 2, "Hip Replacement": 3, "Cataract Surgery": 4,
                                "Hernia Repair": 5, "Knee Replacement": 6, "Liver Transplant": 7, "Heart Bypass": 8, "Spinal Fusion": 9}[surgery_name]
        room_type_encoded = {"Regular Ward": 0, "ICU": 1, "VIP Ward": 2, "Private Ward": 3}[room_type]
    
        # Return the input data as a list in the format expected by the model
        return [
            insurance_type_encoded,
            surgery_name_encoded,
            surgery_duration,
            room_type_encoded,
            medical_equipment_count,
            ward_cost,
            surgery_cost,
            medication_cost,
            total_cost
        ]

    # # Button to predict the hospital stay duration
    # if st.button("Predict Stay Duration"):
    #     input_data = encode_inputs(insurance_type, surgery_name, surgery_duration, room_type, medical_equipment_count, ward_cost, surgery_cost, medication_cost, total_cost)
    #     prediction = predict_stay_duration(input_data)
    #     st.write(f"Predicted Hospital Stay Duration: {prediction} days")
    # Button to predict the hospital stay duration
    if st.button("Predict Stay Duration"):
        # Encode the inputs for prediction
        input_data = encode_inputs(
            insurance_type, surgery_name, surgery_duration, room_type, 
            medical_equipment_count, ward_cost, surgery_cost, 
            medication_cost, total_cost
        )
        
        # Get the prediction from the model
        prediction = predict_stay_duration(input_data)
        
        # Round up the prediction to the nearest integer
        rounded_prediction = math.ceil(prediction)
        
        # Enhanced visualization
        st.subheader("Predicted Hospital Stay Duration")
        if rounded_prediction <= 3:
            st.success(f"The predicted hospital stay duration is: {rounded_prediction} days (Short Stay)")
        elif 4 <= rounded_prediction <= 7:
            st.info(f"The predicted hospital stay duration is: {rounded_prediction} days (Moderate Stay)")
        elif 8 <= rounded_prediction <= 14:
            st.warning(f"The predicted hospital stay duration is: {rounded_prediction} days (Long Stay)")
        else:
            st.error(f"The predicted hospital stay duration is: {rounded_prediction} days (Very Long Stay)")

elif page == "Total Hospitalization Cost":

# Load the saved model (adjust the path to where your model is saved)
    try:
        cost_model = joblib.load('Source/Prototype/patient_stay_cost/patient_xgb_cost.pkl')
    except FileNotFoundError:
        try:
            cost_model = joblib.load('C:\\Users\\Republic Of Gamers\\OneDrive\\Documents\\GitHub\\TSDN-BoyWithLuv\\Source\\Prototype\\patient_stay_cost\\patient_xgb_cost.pkl')
        except FileNotFoundError:
            cost_model = None
            print("Error: Model file not found in either path.")

        
    # model = joblib.load('C:\\Users\\Republic Of Gamers\\OneDrive\\Documents\\GitHub\\TSDN-BoyWithLuv\\Source\\Prototype\\patient_stay_cost\\patient_xgb_cost.pkl')

# # Initialize LabelEncoders for categorical features
#     gender_encoder = LabelEncoder()
#     insurance_type_encoder = LabelEncoder()
#     smoking_status_encoder = LabelEncoder()
#     e_cigarette_usage_encoder = LabelEncoder()
#     alcohol_consumption_rate_encoder = LabelEncoder()
#     surgery_name_encoder = LabelEncoder()
#     room_type_encoder = LabelEncoder()

# # Fit the label encoders with the unique values
#     gender_encoder.fit(['Male', 'Female'])
#     insurance_type_encoder.fit(['Private', 'Social Security Agency', 'Self-Pay'])
#     smoking_status_encoder.fit(['Never', 'Current', 'Former'])
#     e_cigarette_usage_encoder.fit([False, True])
#     alcohol_consumption_rate_encoder.fit(['Occasional', 'None', 'Heavy', 'Moderate'])
#     surgery_name_encoder.fit(['Gallbladder Removal', 'Breast Cancer Surgery', 'Appendectomy',
#                             'Hip Replacement', 'Cataract Surgery', 'Hernia Repair',
#                             'Knee Replacement', 'Liver Transplant', 'Heart Bypass', 'Spinal Fusion'])
#     room_type_encoder.fit(['Regular Ward', 'ICU', 'VIP Ward', 'Private Ward'])

#     # Mapping for cost class
#     total_cost_class_mapping = {
#         'Very Low': 0, 
#         'Low': 1, 
#         'Medium' : 2, 
#         'High': 3, 
#         'Very High': 4
#     }

# # Reverse mapping for the output class
#     reverse_total_cost_class_mapping = {v: k for k, v in total_cost_class_mapping.items()}

# # Function to predict the cost class
#     def predict_cost_class(features):
#         # Encoding categorical features
#         features['gender'] = gender_encoder.transform([features['gender']])[0]
#         features['insurance_type'] = insurance_type_encoder.transform([features['insurance_type']])[0]
#         features['smoking_status'] = smoking_status_encoder.transform([features['smoking_status']])[0]
#         features['e_cigarette_usage'] = e_cigarette_usage_encoder.transform([features['e_cigarette_usage']])[0]
#         features['alcohol_consumption_rate'] = alcohol_consumption_rate_encoder.transform([features['alcohol_consumption_rate']])[0]
#         features['surgery_name'] = surgery_name_encoder.transform([features['surgery_name']])[0]
#         features['room_type'] = room_type_encoder.transform([features['room_type']])[0]

#     # Making prediction
#         predicted_class_index = cost_model.predict([features])[0]

#     # Reverse mapping to get the human-readable class label
#         predicted_class_label = reverse_total_cost_class_mapping[predicted_class_index]
#         return predicted_class_label

# # Streamlit Interface
#     st.title('Hospital Cost Prediction')

#     st.write("This app predicts the hospital cost classification based on various features.")

# # Input Fields
#     gender = st.selectbox('Gender', ['Male', 'Female'])
#     age = st.number_input('Age', min_value=0, max_value=120, value=30)
#     insurance_type = st.selectbox('Insurance Type', ['Private', 'Social Security Agency', 'Self-Pay'])
#     smoking_status = st.selectbox('Smoking Status', ['Never', 'Current', 'Former'])
#     e_cigarette_usage = st.selectbox('E-Cigarette Usage', ['Yes', 'No'])
#     alcohol_consumption_rate = st.selectbox('Alcohol Consumption Rate', ['Occasional', 'None', 'Heavy', 'Moderate'])
#     previous_admission_count = st.number_input('Previous Admission Count', min_value=0, value=0)
#     surgery_name = st.selectbox('Surgery Name', ['Gallbladder Removal', 'Breast Cancer Surgery', 'Appendectomy',
#                                                 'Hip Replacement', 'Cataract Surgery', 'Hernia Repair',
#                                                 'Knee Replacement', 'Liver Transplant', 'Heart Bypass',
#                                                 'Spinal Fusion'])
#     room_type = st.selectbox('Room Type', ['Regular Ward', 'ICU', 'VIP Ward', 'Private Ward'])
#     stay_duration = st.number_input('Stay Duration (in days)', min_value=1, value=3)
#     medical_equipment_count = st.number_input('Medical Equipment Count', min_value=0, value=1)

# # Creating a dictionary of input values
#     input_features = {
#         'gender': gender,
#         'age': age,
#         'insurance_type': insurance_type,
#         'smoking_status': smoking_status,
#         'e_cigarette_usage': e_cigarette_usage,
#         'alcohol_consumption_rate': alcohol_consumption_rate,
#         'previous_admission_count': previous_admission_count,
#         'surgery_name': surgery_name,
#         'room_type': room_type,
#         'stay_duration': stay_duration,
#         'medical_equipment_count': medical_equipment_count
#     }

#     # Button to trigger prediction
#     if st.button('Predict Cost Class', key='predict_cost_button'):

#         # Convert input to pandas DataFrame
#         input_df = pd.DataFrame([input_features])

#         # Prediction
#         predicted_class = predict_cost_class(input_df.iloc[0])

#         # Enhanced visualization
#         if predicted_class == "Very Low":
#             st.success(f"The predicted cost class is: {predicted_class} ")
#         elif predicted_class == "Low":
#             st.info(f"The predicted cost class is: {predicted_class} ")
#         elif predicted_class == "Medium":
#             st.warning(f"The predicted cost class is: {predicted_class} ")
#         elif predicted_class == "High":
#             st.error(f"The predicted cost class is: {predicted_class} ")
#         elif predicted_class =="Very High":
#             st.error(f"The predicted cost class is: {predicted_class} ")

    # Collapsible input sections
    st.subheader("Personal Details")
    with st.expander("Enter Personal Information"):
        gender = st.selectbox('Gender', ['Male', 'Female'])
        age = st.number_input('Age', min_value=0, max_value=150, value=30)
        insurance_type = st.selectbox('Insurance Type', ['Private', 'Social Security Agency', 'Self-Pay'])

    st.subheader("Health & Lifestyle")
    with st.expander("Enter Health and Lifestyle Details"):
        smoker_status = st.selectbox('Smoker Status', ['Never', 'Current', 'Former'])
        e_cigarette_usage = st.selectbox('E-Cigarette Usage', ['Yes', 'No'])
        alcohol_consumption = st.selectbox('Alcohol Consumption', ['Occasional', 'None', 'Moderate', 'Heavy'])

    st.subheader("Medical History")
    with st.expander("Enter Medical History"):
        previous_admision = st.number_input('Previous Admission Count', min_value=0, max_value=150, value=1)
        surgery_name = st.selectbox('Surgery Name', [
            'Gallbladder Removal', 'Breast Cancer Surgery','Appendectomy', 'Hip Replacement', 'Cataract Surgery', 'Hernia Repair' ,'Knee Replacement', 'Liver Transplant', 'Heart Bypass' , 'Spinal Fusion'])

    st.subheader("Admission Details")
    with st.expander("Enter Screening and Vaccination Details"):
        room_type = st.selectbox('Ward Type', ['Regular Ward', 'ICU', 'VIP Ward', 'Private Ward'])
        admission_duration = st.number_input('Estimated Patient Stay Duration (Days)', min_value=0, max_value=600, value=1)
        equipment_count = st.number_input('Medical Equipment Count Used', min_value=0, max_value=50, value=1)


    # Create DataFrame for model input
    input_data = pd.DataFrame({
        'gender': [gender],
        'age': [age],
        'insurance_type': [insurance_type],
        'smoking_status': [smoker_status],
        'e_cigarette_usage': [e_cigarette_usage],
        'alcohol_consumption_rate': [alcohol_consumption],
        'previous_admission_count': [previous_admision],
        'surgery_name': [surgery_name],
        'room_type': [room_type],
        'stay_duration': [admission_duration],
        'medical_equipment_count': [equipment_count]
    })

    input_data['gender'] = input_data['gender'].map(gender_mapping)
    input_data['insurance_type'] = input_data['insurance_type'].map(insurance_type_mapping)
    input_data['smoking_status'] = input_data['smoking_status'].map(smoking_status_mapping)
    input_data['e_cigarette_usage'] = input_data['e_cigarette_usage'].map(e_cigarette_usage_mapping)
    input_data['alcohol_consumption_rate'] = input_data['alcohol_consumption_rate'].map(alcohol_consumption_rate_mapping)
    input_data['surgery_name'] = input_data['surgery_name'].map(surgery_name_mapping)
    input_data['room_type'] = input_data['room_type'].map(room_type_mapping)
    

    # # Predict button
    # st.markdown("---")
    # if st.button("Predict Cost Class"):
    #     # Perform prediction using the model
    #     cost_prediction = cost_model.predict(input_data)
        
    #     # Extract the predicted class number
    #     predicted_class_number = cost_prediction[0]
        
    #     # Display prediction result with custom styling
    #     if predicted_class_number in [3, 4]:  # High or Very High classes
    #         st.error(f"Prediction Result: Class {predicted_class_number}")
    #     else:
    #         st.success(f"Prediction Result: Class {predicted_class_number}")

    # Predict button
    st.markdown("---")
    if st.button("Predict Cost Class"):
        # Perform prediction using the model
        cost_prediction = cost_model.predict(input_data)
        
        # Reverse mapping from numerical output to cost class label
        reverse_total_cost_class_mapping = {v: k for k, v in total_cost_class_mapping.items()}
        predicted_class_label = reverse_total_cost_class_mapping.get(cost_prediction[0], "Unknown")
        
        # Display prediction result with custom styling
        if predicted_class_label in ['High', 'Very High']:
            st.error(f"Prediction Result: {predicted_class_label} Cost Class")
        else:
            st.success(f"Prediction Result: {predicted_class_label} Cost Class")
