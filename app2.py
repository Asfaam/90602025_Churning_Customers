import streamlit as st
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the pre-trained MLP model
model = load_model('customer_churn.keras')
model.summary()

def predict_churn(features):
    try:
        # Preprocess the input data (you may need to adjust this based on your preprocessing during training)
        churn_dataset = pd.read_csv('CustomerChurn_dataset.csv')

        # Renaming column: gender to Gender
        churn_dataset.rename(columns={'gender':'Gender', }, inplace=True)

        # Renaming column: tenure to Tenure
        churn_dataset.rename(columns={'tenure':'Tenure', }, inplace=True)

        # Convert the data type of the 'TotalCharges' column from object to numeric
        churn_dataset['TotalCharges'] = churn_dataset['TotalCharges'].apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna()

        # Convert 'No internet service' to 'No' for the below-mentioned columns
        cols = ['OnlineBackup','StreamingMovies','DeviceProtection',
                'TechSupport','OnlineSecurity','StreamingTV']
        for i in cols:
            churn_dataset[i]  = churn_dataset[i].replace({'No internet service' : 'No'})

        # Convert 'No phone service' to 'No' for the below-mentioned column
        churn_dataset['MultipleLines'] = churn_dataset['MultipleLines'].replace({'No phone service' : 'No'})

        # Drop 'customerID' column since it is not useful for churn prediction as the feature is used for identification of customers.
        selected_features = churn_dataset.drop(['customerID'], axis = 1)

        # Perform Label encoding on the categorical columns columns
        label_encoder = LabelEncoder()
        selected_features['Gender'] = label_encoder.fit_transform(selected_features['Gender'])
        selected_features['Dependents'] = label_encoder.fit_transform(selected_features['Dependents'])
        selected_features['Partner'] = label_encoder.fit_transform(selected_features['Partner'])
        selected_features['PhoneService'] = label_encoder.fit_transform(selected_features['PhoneService'])
        selected_features['MultipleLines'] = label_encoder.fit_transform(selected_features['MultipleLines'])
        selected_features['InternetService'] = label_encoder.fit_transform(selected_features['InternetService'])
        selected_features['OnlineSecurity'] = label_encoder.fit_transform(selected_features['OnlineSecurity'])
        selected_features['OnlineBackup'] = label_encoder.fit_transform(selected_features['OnlineBackup'])
        selected_features['DeviceProtection'] = label_encoder.fit_transform(selected_features['DeviceProtection'])
        selected_features['TechSupport'] = label_encoder.fit_transform(selected_features['TechSupport'])
        selected_features['StreamingTV'] = label_encoder.fit_transform(selected_features['StreamingTV'])
        selected_features['StreamingMovies'] = label_encoder.fit_transform(selected_features['StreamingMovies'])
        selected_features['Contract'] = label_encoder.fit_transform(selected_features['Contract'])
        selected_features['PaperlessBilling'] = label_encoder.fit_transform(selected_features['PaperlessBilling'])
        selected_features['PaymentMethod'] = label_encoder.fit_transform(selected_features['PaymentMethod'])
        selected_features['Churn'] = label_encoder.fit_transform(selected_features['Churn'])

        # Impute missing values for 'TotalCharges' column
        selected_features['TotalCharges'] = selected_features['TotalCharges'].fillna(selected_features['TotalCharges'].median())

        # Perform Feature Scaling on 'tenure', 'MonthlyCharges' in order to bring them on the same scale
        standardScaler = StandardScaler()
        columns_for_scaling = ['SeniorCitizen','Tenure', 'MonthlyCharges', 'TotalCharges']

        # Apply the feature scaling operation on the dataset using fit_transform() method
        selected_features[columns_for_scaling] = standardScaler.fit_transform(selected_features[columns_for_scaling])

        final_selected_features = ['SeniorCitizen', 'Dependents', 'Tenure', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges']
        y = selected_features['Churn']
        X = selected_features[final_selected_features]

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        input_data = np.array(features).reshape(1, -1)

        # Make predictions
        prediction = model.predict(input_data)
        churn_probability = prediction[0, 0]

        return churn_probability
    except Exception as e:
        return str(e)

def main():
    st.title("Customer Churn Prediction")

    # Create input fields for user features
    seniorcitizen = st.selectbox(' Customer is a senior citizen:', [0, 1])
        
    dependents = st.selectbox(' Customer has dependents:', ['yes', 'no'])
    phoneservice = st.selectbox(' Customer has phoneservice:', ['yes', 'no'])
    multiplelines = st.selectbox(' Customer has multiplelines:', ['yes', 'no'])
        
    onlinesecurity = st.selectbox(' Customer has onlinesecurity:', ['yes', 'no'])
    onlinebackup = st.selectbox(' Customer has onlinebackup:', ['yes', 'no'])
    deviceprotection = st.selectbox(' Customer has deviceprotection:', ['yes', 'no'])
    techsupport = st.selectbox(' Customer has techsupport:', ['yes', 'no'])
    streamingtv = st.selectbox(' Customer has streamingtv:', ['yes', 'no'])
    streamingmovies = st.selectbox(' Customer has streamingmovies:', ['yes', 'no'])
    contract = st.selectbox(' Customer has a contract:', ['month-to-month', 'one_year', 'two_year'])
    paperlessbilling = st.selectbox(' Customer has a paperlessbilling:', ['yes', 'no'])
        
    tenure = st.number_input('Number of months the customer has been with the current telco provider:', min_value=0, max_value=240, value=0)
    monthlycharges = st.number_input('Monthly charges:', min_value=0, max_value=240, value=0)
    totalcharges = tenure * monthlycharges   

    # Make prediction when the user clicks the button
    if st.button("Predict"):
        features = [float(feature1), float(feature2)]  # Add more features as needed
        churn_probability = predict_churn(features)

        # Display the prediction result
        st.success(f"Churn Probability: {churn_probability}")

if __name__ == "__main__":
    main()
