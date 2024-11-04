# WebApp.py
import streamlit as st

st.set_page_config(page_title="ML Project Web App", layout="wide")

# Title and introductory text
st.title("Welcome to the ML Project Web App 👋")


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
