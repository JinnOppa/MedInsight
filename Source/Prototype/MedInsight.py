# WebApp.py
import streamlit as st

st.set_page_config(page_title="ML Project Web App", layout="wide")

# Title and introductory text
st.title("Welcome to MedInsight👋")
st.header("About Us")

col1, col2 = st.columns(2)
with col1:
    st.image("C:\\Users\\Republic Of Gamers\\OneDrive - Asia Pacific University\\Self Photo\\20241113_001141.jpg", use_column_width=True)
    st.markdown(
        """
        <div style='text-align: center;'>
            <h3>STEVEN YENARDI</h3>
            <p style='font-size:18px;'>An Indonesian Student Pursuing Bachelor Degree at Asia Pacific University of Technology and Innovation, Malaysia.<br>
            Majoring in Computer Science Specializing in Data Analytics.</p>
            <p><a href="https://www.linkedin.com/in/stevenyenardi/" target="_blank" style='color: #0077b5; font-size:16px; text-decoration: none;'>LinkedIn Profile</a></p>
            <p><a href="https://github.com/stevenyenardi" target="_blank" style='color: purple; font-size:16px; text-decoration: none;'>Github Profile</a></p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.image("C:\\Users\\Republic Of Gamers\\OneDrive - Asia Pacific University\\IMG_0156.jpg", use_column_width=True)
    st.markdown(
        """
        <div style='text-align: center;'>
            <h3>EUGENE WINATA</h3>
            <p style='font-size:18px;'>An Indonesian Student Pursuing Bachelor Degree at Asia Pacific University of Technology and Innovation, Malaysia.<br>
            Majoring in Computer Science Specializing in Data Analytics.</p>
            <p><a href="https://www.linkedin.com/in/eugene-winata/" target="_blank" style='color: #0077b5; font-size:16px; text-decoration: none;'>LinkedIn Profile</a></p>
            <p><a href="https://github.com/JinnOppa" target="_blank" style='color: purple; font-size:16px; text-decoration: none;'>Github Profile</a></p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")
st.header('Aim')
st.write('The aim of this project is to develop predictive models using data analytics and data science techniques to forecast hospital-related issues, including predicting the number of patients requiring hospital stays, estimating survival rates, and predicting diseases based on user-inputted patient details.')
st.markdown("---")
st.header('Objectives')
import streamlit as st

# Displaying numbered points and bullet points in markdown
st.markdown("""
1. **Data Collection and Preprocessing**: Gather relevant hospital datasets that include patient demographics, medical conditions, test results, admission types, historical stay durations, surgical risk data, and survival rate information. Clean and preprocess the data by handling missing values, identifying and addressing outliers, and categorizing features (e.g., age, gender, medical conditions, surgical risk factors, and survival status). This will also involve encoding categorical variables and scaling numerical features as necessary.

2. **Predictive Modeling**: Implement machine learning algorithms (e.g., XGBoost and Random Forest) to create models for:
    - **Disease Prediction**: Predict the likelihood of patients developing specific diseases based on their symptoms, medical history, and test results.
    - **Hospital Stay Duration Prediction**: Estimate the number of days a patient is likely to stay in the hospital, using features such as medical conditions, treatment plans, and historical stay data.
    - **Total Cost Prediction**: Predict the total cost of hospitalization, considering factors like admission type, treatment, test results, and duration of stay.
    - **Surgery Risk Prediction**: Assess the potential risks of surgery for patients based on their health metrics, medical history, and current condition.
    - **Survival Rate Prediction**: Predict the likelihood of patient survival based on features such as age, medical conditions, vital signs, comorbidities, and treatment plans.

3. **Model Evaluation**: Evaluate the performance of the predictive models using appropriate metrics:
    - For **classification models** (e.g., disease prediction, surgery risk prediction, survival rate prediction), use metrics such as accuracy, precision, recall, F1-score, and AUC.
    - For **regression models** (e.g., hospital stay duration, total cost prediction), use metrics such as RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), and R².
    - Fine-tune the models using hyperparameter optimization techniques like grid search or random search to improve performance and prediction accuracy.

4. **Deployment**: Deploy the trained models into a user-friendly application (e.g., Streamlit app). This application will allow healthcare providers and users to input patient data (demographics, symptoms, test results, etc.) and receive predictions related to:
    - Disease risk (likelihood of specific diseases).
    - Estimated hospital stay duration.
    - Projected total cost of treatment.
    - Surgery risk assessment.
    - Survival rate prediction (likelihood of patient survival).
""")

st.markdown("---")
st.header('Goal')
st.write("The goal is to provide healthcare professionals and users with reliable, data-driven tools to predict hospital stays, patient survival rates, and potential diseases. This project aims to assist in better resource allocation, improve patient care, enable early disease detection, and enhance decision-making processes in hospital management. Ultimately, this project seeks to contribute to reducing hospital overcrowding, improving patient outcomes, and supporting proactive health management.")


# st.sidebar.success("Select a page from the sidebar.")

# # Sidebar navigation
# page = st.sidebar.radio("Go to:", ("Home", "Disease Prediction", "Operational Insights"))

# if page == "Home":
#     st.write("# Welcome to Streamlit!")
#     st.markdown(
#         """
#         Streamlit is an open-source app framework built specifically for
#         Machine Learning and Data Science projects.
#         **👈 Select a page from the sidebar** to explore the web app features!
#         ### Learn More
#         - [Streamlit Documentation](https://docs.streamlit.io)
#         - [Streamlit Community](https://discuss.streamlit.io)
#         """
#     )
# elif page == "Disease Prediction":
#     # Redirect to the disease prediction page
#     st.experimental_set_query_params(page="disease_prediction")
#     st.experimental_rerun()
# elif page == "Operational Insights":
#     # Redirect to the operational insights page
#     st.experimental_set_query_params(page="operational_insights")
#     st.experimental_rerun()
