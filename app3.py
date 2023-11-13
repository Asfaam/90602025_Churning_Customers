import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('customer_churn.h5')

# Define the user interface
st.title('Customer Churn Prediction')
st.write('Enter customer data to predict churn')

# Create input fields for user features
Gender = st.selectbox(' Customer's gender:', ['female', 'male'])
SeniorCitizen = st.selectbox(' Customer is a senior citizen:', [0, 1])
Dependents = st.selectbox(' Customer has dependents:', ['yes', 'no'])
PhoneService = st.selectbox(' Customer has phoneservice:', ['yes', 'no'])
MultipleLines = st.selectbox(' Customer has multiplelines:', ['yes', 'no'])
OnlineSecurity = st.selectbox(' Customer has onlinesecurity:', [])
OnlineBackup = st.selectbox(' Customer has onlinebackup:', ['yes', 'no'])
DeviceProtection = st.selectbox(' Customer has deviceprotection:', ['yes', 'no'])
TechSupport = st.selectbox(' Customer has techsupport:', ['yes', 'no'])
StreamingTV = st.selectbox(' Customer has streamingtv:', ['yes', 'no'])
StreamingMovies = st.selectbox(' Customer has streamingmovies:', ['yes', 'no'])
PaperlessBilling = st.selectbox(' Customer has a paperlessbilling:', ['yes', 'no'])

# Convert categorical values to numerical values
Gender = 1 if Gender == 'male' else 0
Dependents = 1 if Dependents == 'yes' else 0
PhoneService = 1 if PhoneService == 'yes' else 0
MultipleLines = 1 if MultipleLines == 'yes' else 0

# Convert "InternetService" categorical value to numerical
InternetService_mapping = {'DSL	': 0, 'Fiber optic': 1, 'no': 2}
InternetService = st.selectbox('Which InternetService does customer use:', list(InternetService_mapping.keys()))
InternetService = InternetService_mapping[InternetService]
    
OnlineSecurity = 1 if OnlineSecurity == 'yes' else 0
OnlineBackup = 1 if OnlineBackup == 'yes' else 0
DeviceProtection = 1 if DeviceProtection == 'yes' else 0
TechSupport = 1 if TechSupport == 'yes' else 0
StreamingTV = 1 if StreamingTV == 'yes' else 0
StreamingMovies = 1 if StreamingMovies == 'yes' else 0
PaperlessBilling = 1 if PaperlessBilling == 'yes' else 0

# Convert "Contract" categorical value to numerical
contract_mapping = {'month-to-month': 0, 'one_year': 1, 'two_year': 2}
Contract = st.selectbox('Customer has a contract:', list(contract_mapping.keys()))
Contract = contract_mapping[Contract]
    

Tenure = st.number_input('Number of months the customer has been with the current telco provider:', min_value=0, max_value=240, value=0)
MonthlyCharges = st.number_input('Monthly charges:', min_value=0, max_value=1000000, value=0)
TotalCharges = Tenure * MonthlyCharges


# Display TotalCharges
    st.write(f'TotalCharges: {TotalCharges}')
    

inputs = [float(Gender), float(SeniorCitizen), float(Dependents), float(PhoneService), float(MultipleLines), float(InternetService), float(OnlineSecurity), float(OnlineBackup), float(DeviceProtection), float(TechSupport), float(StreamingTV), float(StreamingMovies), float(PaperlessBilling), float(Contract), float(Tenure), float(MonthlyCharges), float(TotalCharges)]


# Define the prediction function
def predict_churn(inputs):
    # Preprocess the inputs
    inputs = np.array(inputs).reshape(1, -1)
    # Make the prediction
    prediction = model.predict(inputs)
    return prediction[0]

# Make the prediction
if st.button('Predict Churn'):
    prediction = predict_churn(inputs)
    st.write('Churn Prediction:', prediction)