import streamlit as st
import keras
import utils
import pickle
import warnings
import numpy as np
import pandas as pd
from math import sqrt
import seaborn as sns
import tensorflow as tf
from keras import layers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import plotly.express as px ##  For visualization
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt ##  For visualization
from tensorflow.keras.layers import Input, Dense, Dropout
from sklearn.metrics import accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE ## For oversampling
from sklearn.feature_selection import RFECV ## For feature selection
from sklearn.model_selection import StratifiedKFold ## For feature selection
from sklearn.linear_model import LogisticRegression ## For feature selection
warnings.simplefilter(action='ignore', category=Warning) ## Suppress warnings
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,roc_curve, recall_score,
                                  classification_report, f1_score, precision_recall_fscore_support)

# Define the custom objects with the layer names
custom_objects = {
    'Dense': tf.keras.layers.Dense,
    'InputLayer': tf.keras.layers.InputLayer,
    'Dropout': tf.keras.layers.Dropout,
}

# Load the model with custom objects
model = load_model('customer_churn.h5', custom_objects=custom_objects, compile=True)

# Recompile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the summary of the loaded model
print(model.summary())

def predict_churn(features):
    try:
        churn_dataset = pd.read_csv('CustomerChurn_dataset.csv')

        ## Renaming column: tenure to Tenure
        churn_dataset.rename(columns={'tenure':'Tenure', }, inplace=True)

        ## Convert the data type of the 'TotalCharges' column from object to numeric
        churn_dataset['TotalCharges'] = churn_dataset['TotalCharges'].apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna()

        ## Convert 'No internet service' to 'No' for the below mentioned columns
        cols = ['OnlineBackup','StreamingMovies','DeviceProtection',
                'TechSupport','OnlineSecurity','StreamingTV']
        for i in cols :
            churn_dataset[i]  = churn_dataset[i].replace({'No internet service' : 'No'})

        ## Convert 'No phone service' to 'No' for the below mentioned column
        churn_dataset['MultipleLines'] = churn_dataset['MultipleLines'].replace({'No phone service' : 'No'})

        ## Drop 'customerID' column since it is not useful for churn prediction as the feature is used for the identification of customers.
        selected_features = churn_dataset.drop(['customerID'], axis=1)

        
        ## Perform Label encoding on the categorical columns columns
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

        """### Feature Selection"""

        ## Select the independent (X) and dependent (y) variables from the selected_features dataset
        y = selected_features['Churn']
        X = selected_features.drop(['Churn'], axis=1)


        ## Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        ## Feature selection to improve model building
        log = LogisticRegression()
        rfecv = RFECV(estimator=log, cv=StratifiedKFold(10, random_state=50, shuffle=True), scoring="accuracy")
        rfecv.fit(X, y)

        ## Recursive Feature Elimination (REF)
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'])
        plt.grid()
        plt.xticks(range(1, X.shape[1] + 1))
        plt.xlabel("Number of Selected Features")
        plt.ylabel("CV Score")
        plt.title("Recursive Feature Elimination (RFE)")
        plt.show()

        print("The optimal number of features: {}".format(rfecv.n_features_))

        ## Saving dataframe with optimal features
        X_rfe = X.iloc[:, rfecv.support_]

        ## Overview of the optimal features in comparison with the initial dataframe
        print("\nInitial dimension of X: {}".format(X.shape))
        print("\nInitial X column list:", X.columns.tolist())
        print("\nDimension of X considering only the optimal features: {}".format(X_rfe.shape))
        print("\nColumn list of X considering only the optimal features:", X_rfe.columns.tolist())

        """### **Training**"""

        ## The dataset containing optimal features
        final_selected_features = X_rfe.columns.tolist()

        ## Select the independent (X) and dependent (y) variables
        X = selected_features[final_selected_features]
        y = selected_features['Churn']

        ## Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



        ## Perform Feature Scaling on X_train and X_test
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.fit_transform(X_test)

       
        oversample = SMOTE(k_neighbors=5)
        X_smote, y_smote = oversample.fit_resample(X_train, y_train)
        X_train, y_train = X_smote, y_smote

        X_train.shape, y_test.shape, y_train.shape, X_test.shape

        input_data = np.array(features).reshape(1, -1)
       
        # Make predictions
        prediction = model.predict(input_data)
        churn_probability = prediction[0, 0]

        return churn_probability
    except Exception as e:
        return str(e)



def main():
    st.title("Telecom Customer Churn Prediction App")
    st.markdown("""
        :dart: This app predicts customer churn in a fictional telecommunication use case.
    """)
    st.write("Enter customer details to predict customer churn.")

    # Create input fields for user features
    SeniorCitizen = st.selectbox(' Customer is a senior citizen:', [0, 1])

    Dependents = st.selectbox(' Customer has dependents:', ['yes', 'no'])
    PhoneService = st.selectbox(' Customer has phoneservice:', ['yes', 'no'])
    MultipleLines = st.selectbox(' Customer has multiplelines:', ['yes', 'no'])
    OnlineSecurity = st.selectbox(' Customer has onlinesecurity:', ['yes', 'no'])
    OnlineBackup = st.selectbox(' Customer has onlinebackup:', ['yes', 'no'])
    DeviceProtection = st.selectbox(' Customer has deviceprotection:', ['yes', 'no'])
    TechSupport = st.selectbox(' Customer has techsupport:', ['yes', 'no'])
    StreamingTV = st.selectbox(' Customer has streamingtv:', ['yes', 'no'])
    StreamingMovies = st.selectbox(' Customer has streamingmovies:', ['yes', 'no'])
    PaperlessBilling = st.selectbox(' Customer has a paperlessbilling:', ['yes', 'no'])

    # Convert categorical values to numerical values
    Dependents = 1 if Dependents == 'yes' else 0
    PhoneService = 1 if PhoneService == 'yes' else 0
    MultipleLines = 1 if MultipleLines == 'yes' else 0
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
    

    Tenure = st.number_input(
        'Number of months the customer has been with the current telco provider:', min_value=0, max_value=240, value=0)
    MonthlyCharges = st.number_input('Monthly charges:', min_value=0, max_value=1000000, value=0)
    TotalCharges = Tenure * MonthlyCharges

    # Display TotalCharges
    st.write(f'TotalCharges: {TotalCharges}')
    

    # Make prediction when the user clicks the button
    if st.button("Predict Customer Churn"):
        features = [float(SeniorCitizen), float(Dependents), float(PhoneService), float(MultipleLines), float(OnlineSecurity), float(OnlineBackup), float(DeviceProtection), float(TechSupport),       float(StreamingTV), float(StreamingMovies), float(Contract), float(PaperlessBilling), float(MonthlyCharges), float(TotalCharges), float(Tenure)]
       
        churn_probability = predict_churn(features)
       
        # Display the prediction result
        st.success(f"Churn Probability: {churn_probability}")

        # Display a message based on the churn probability
        #if churn_probability > 0.5:
            #st.warning("High churn probability. Customer is likely to churn.")
        #else:
            #st.info("Low churn probability. Customer is less likely to churn.")
        


if __name__ == "__main__":
    main()
