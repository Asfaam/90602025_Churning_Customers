# Import necessary libraries
import pickle
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

# Load the trained model
filename = 'customer_churn.pkl'
model = pickle.load(open(filename, 'rb'))

def preprocess_input(data):
    # Ensure the columns and their order match the training data
    data = data[['SeniorCitizen', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines',
                 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'TechSupport', 'StreamingTV',
                 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges',
                 'TotalCharges']]

    # Convert categorical columns to numeric using one-hot encoding
    data = pd.get_dummies(data, columns=['SeniorCitizen', 'Dependents', 'PhoneService', 'MultipleLines',
                                         'InternetService', 'OnlineSecurity', 'OnlineBackup', 'TechSupport',
                                         'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                                         'PaymentMethod'])

    # Fill missing values if any
    data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())

    # Perform feature scaling on numeric columns
    scaler = StandardScaler()
    data[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(
        data[['tenure', 'MonthlyCharges', 'TotalCharges']])

    # Ensure the order of columns matches the training data used for feature selection
    data = data[['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract_Month-to-month',
                 'PaperlessBilling_Yes', 'InternetService_Fiber optic', 'TechSupport_No',
                 'OnlineBackup_No', 'DeviceProtection_No', 'MultipleLines_No phone service',
                 'OnlineSecurity_No internet service', 'StreamingTV_No internet service',
                 'StreamingMovies_No internet service', 'SeniorCitizen_No', 'Dependents_No',
                 'PhoneService_No', 'MultipleLines_No', 'InternetService_DSL', 'InternetService_No',
                 'OnlineSecurity_Yes', 'OnlineBackup_Yes', 'DeviceProtection_Yes', 'TechSupport_Yes',
                 'StreamingTV_Yes', 'StreamingMovies_Yes', 'Contract_One year', 'Contract_Two year',
                 'PaperlessBilling_No', 'PaymentMethod_Bank transfer (automatic)',
                 'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
                 'PaymentMethod_Mailed check']]

    return data

# Streamlit web application
import streamlit as st

def main():
    st.title('Telecom Customer Churn Prediction App')
    st.markdown("""
        :dart: This Streamlit app predicts customer churn in a fictional telecommunication use case.
        The application supports both online prediction and batch data prediction.
    """)

    # Sidebar for selecting prediction mode
    add_selectbox = st.sidebar.selectbox("How would you like to predict?", ("Online", "Batch"))

    if add_selectbox == "Online":
        st.info("Input data below")

        # Collect input data from user
        # (Insert Streamlit input components based on your original code)

        # Preprocess the input for prediction
        input_data = preprocess_input(data)
        prediction = model.predict(input_data)

        # Display prediction result
        if st.button('Predict'):
            if prediction == 1:
                st.warning('Yes, the customer will terminate the service.')
            else:
                st.success('No, the customer is happy with Telco Services.')

    else:
        st.subheader("Dataset upload")
        uploaded_file = st.file_uploader("Choose a file")

        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            preprocess_df = preprocess(data, "Batch")

            if st.button('Predict'):
                # Batch prediction for uploaded dataset
                predictions = model.predict(preprocess_df)
                prediction_df = pd.DataFrame(predictions, columns=["Predictions"])
                prediction_df = prediction_df.replace({1: 'Churn', 0: 'No Churn'})

                st.subheader('Prediction')
                st.write(prediction_df)

if __name__ == '__main__':
    main()
