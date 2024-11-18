# MedInsight

## Project Overview
MedInsight is a web application designed to predict disease risks and provide insights into hospitalization outcomes. This platform leverages predictive analytics to forecast disease risks and hospitalization factors for major non-communicable diseases (NCDs) such as heart attack, stroke, angina, depressive disorders, arthritis, and skin cancer. The project also predicts hospitalization-related outcomes, including survival rates, surgery risks, costs, and expected stay durations.

## Introduction
Indonesia is facing significant healthcare challenges, driven by a growing population, increasing prevalence of non-communicable diseases (NCDs), and escalating healthcare costs. The government’s Jaminan Kesehatan Nasional (JKN) program has expanded access to healthcare, but disparities in healthcare quality, medical professional shortages, and rising chronic disease management costs remain key issues.

This project uses predictive analytics to provide actionable insights into these issues, helping healthcare providers make data-driven decisions and optimize patient care. The models predict the likelihood of disease development, survival rates, hospitalization costs, and admission durations, ultimately improving healthcare efficiency and patient outcomes.

## Aim
The aim of the MedInsight project is to leverage machine learning algorithms to predict disease risks and hospitalization outcomes. By developing models to forecast occurrences of diseases, survival rates, surgery risks, hospitalization costs, and admission durations, the project aims to assist healthcare providers in improving patient care, optimizing resource allocation, and addressing the healthcare challenges in Indonesia.

## Objectives
The project objectives are as follows:
1. **Data Collection and Preprocessing**: Gather and preprocess hospital datasets that include patient demographics, medical conditions, test results, and historical data on stay durations, surgical risks, and survival rates.
2. **Predictive Modeling**: Implement machine learning models (XGBoost, Random Forest) to predict:
   - Disease risks (heart attack, stroke, angina, etc.)
   - Hospital stay duration
   - Total hospitalization costs
   - Surgery risk
   - Survival rate
3. **Model Evaluation**: Evaluate model performance using appropriate metrics (accuracy, precision, recall, RMSE, etc.) and optimize using techniques like grid search or random search.
4. **Deployment**: Deploy the models into a user-friendly Streamlit app that allows healthcare providers to input patient data and receive predictions on disease risk, surgery risk, hospital stay duration, and more.

## Goal
The goal is to provide reliable, data-driven tools to healthcare professionals, enabling them to predict hospital stays, patient survival rates, and potential diseases. By improving decision-making, optimizing resource allocation, and enabling early disease detection, the project seeks to enhance patient care and support proactive health management.

## Disease Prediction Model Overview

The platform uses machine learning to predict the likelihood of survival based on various medical and physiological factors. The supported diseases include:

- **Heart Attack**
- **Stroke**
- **Angina**
- **Depressive Disorder**
- **Arthritis**
- **Skin Cancer**

Each disease has its own model tailored to analyze relevant data, providing users with personalized health risk predictions. The models are trained using machine learning algorithms such as XGBoost and Random Forest, ensuring accurate results.

### Key Features:
- **User-friendly input form** for easy data entry.
- **Disease-specific models** that are tailored for each condition.
- **High accuracy** in predictions (70%+).
- **Actionable insights** for disease prevention and health management.
- **Educational resources** about risk factors and prevention.

---

## How It Works

### **Disease Prediction**
1. **Input Data**: Users input health-related information such as age, medical history, and lifestyle factors (e.g., smoking, alcohol consumption).
2. **Model Prediction**: The platform uses machine learning models (XGBoost, Random Forest) to process the input data and predict the likelihood of developing a disease such as heart attack, stroke, or skin cancer.
3. **Risk Assessment**: The model analyzes the data and provides a risk score or prediction, allowing healthcare providers to evaluate the patient's risk for a particular disease. This helps in early detection, prevention, and customized care planning.

### **Hospitalization Insights**
1. **Input Data**: Users enter additional hospitalization-related information, such as prior medical conditions, surgery details, and the patient's current health status.
2. **Model Prediction**: The platform uses machine learning models to forecast critical hospitalization outcomes, including:
   - **Survival Rate**: The likelihood of a patient surviving based on medical history and health parameters.
   - **Surgery Risk**: The potential for complications during surgery, depending on factors such as comorbidities, age, and surgical procedure.
   - **Admission Duration**: The expected length of the hospital stay based on the patient's health condition, disease type, and medical history.
   - **Total Hospitalization Cost**: An estimate of the total hospitalization costs, factoring in surgery, treatment, and room charges.
3. **Outcome Insights**: The platform provides insights into the predicted outcomes for the patient’s hospitalization. This information can help healthcare providers make informed decisions about resource allocation, patient management, and care planning.

---

## Model Performance Comparison

The disease prediction models use two machine learning algorithms: **XGBoost** and **Random Forest**.

- **XGBoost** is a gradient-boosting algorithm known for its robustness and high performance in large datasets.
- **Random Forest** is an ensemble method that reduces overfitting and improves accuracy by averaging predictions from multiple decision trees.

Both models are evaluated using classification metrics (accuracy, precision, recall) for disease prediction and regression metrics (RMSE, MAE) for other prediction tasks such as hospital stay duration and total cost.

## Conclusion
MedInsight aims to provide healthcare professionals with tools to predict patient health outcomes, improve hospital resource allocation, and reduce costs. By leveraging machine learning, the platform empowers users with predictive insights that can lead to better decision-making, early disease detection, and improved patient care.

---

## Installation
To run the MedInsight web application locally on your machine, follow these steps
### 1. **Clone the Repository**
First, clone the repository to your local machine using Git. This will download the project files to your system:
```
git clone https://github.com/JinnOppa/TSDN-BoyWithLuv
```
### 2. **Navigate to the Project Directory**
After cloning the repository, change your working directory to the project folder:
```
cd your-repo-directory
```
### 3.**Install Dependencies**
Install the required Python dependencies from the requirements.txt file. This will ensure that you have all the necessary libraries installed to run the app:
```
pip install -r requirements.txt
```
The dependencies include:
- pandas==2.2.0
- matplotlib==3.7.0
- joblib==1.2.0
- xgboost==1.7.6
- scikit-learn==1.5.1 

```
streamlit run WebApp.py
```
### 4. **Run the Application**
Once the dependencies are installed, start the web app using Streamlit by running the following command:
```
streamlit run MedInsight.py
```

### 5. **Access the Web Application**
After running the above command, Streamlit will launch the app and provide a local URL (usually `http://localhost:8501`). Open this URL in your web browser to start using the application.

---

## Contact Information
For further inquiries or collaboration opportunities, please contact us to :
- **Eugene Winata** at [Email](eugene.winata@gmail.com) | [LinkedIn Profile](https://www.linkedin.com/in/eugene-winata/) | [GitHub Profile](https://github.com/JinnOppa)
- **Steven Yenardi** at
    [Steven's Email](stevenyenardi46@gmail.com) | [LinkedIn Profile](https://www.linkedin.com/in/stevenyenardi/) | [GitHub Profile](https://github.com/stevenyenardi)

---

## MIT License

MIT License

Copyright (c) 2024 Eugene Winata and Steven Yenardi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

