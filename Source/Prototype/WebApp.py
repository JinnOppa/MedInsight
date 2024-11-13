# WebApp.py
import streamlit as st

st.set_page_config(page_title="ML Project Web App", layout="wide")

# Title and introductory text
st.title("Welcome to The Web Application👋")
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
    st.image("C:\\Users\\Republic Of Gamers\\OneDrive - Asia Pacific University\\pig.jpg", use_column_width=True)
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

col1, col2, col3 = st.columns(3)
with col1:
    st.header('Aim')
    st.write('The aim of this project is to develop predictive models using data analytics and data science techniques to forecast hospital-related issues, including predicting the number of patients requiring hospital stays, estimating survival rates, and predicting diseases based on user-inputted patient details.')

with col2:
    st.header('Objectives')
    st.write("""
    1.**Data Collection and Preprocessing:** Gather relevant hospital datasets that include patient demographics, medical conditions, test results, admission types, and historical stay durations.
    Clean and preprocess the data by handling missing values, outliers, and categorizing features (e.g., age, gender, medical conditions, etc.).\n
    2.**Predictive Modeling:** Implement machine learning algorithms (e.g., XGBoost and Random Forest) and create models to predict **diseases** based on patient details such as symptoms, medical history, and test results,the number of **hospital stays** for future patients based on several features, as well as the **survival rates** of patients (high or low) based on several features available.\n
    3.**Model Evaluation:** Evaluate the performance of the predictive models using appropriate metrics such as accuracy, precision, recall, F1-score, and AUC for classification models, and RMSE or MAE for regression models and fine-tune the models using hyperparameter optimization techniques to improve model performance and prediction accuracy.\n
    4.**Deployment:** Deploy the trained models into a user-friendly application (e.g., Streamlit app) to allow healthcare providers and users to input new data and receive predictions regarding hospital stay requirements, patient survival probabilities, and potential diseases.
    """)

with col3:
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
