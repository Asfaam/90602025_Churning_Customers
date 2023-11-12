import streamlit as st
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import warnings
import numpy as np
import pandas as pd
from math import sqrt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import plotly.express as px  ## For visualization
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt  ## For visualization
from tensorflow.keras.layers import Input, Dense, Dropout
from sklearn.metrics import accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE  ## For oversampling
from sklearn.feature_selection import RFECV  ## For feature selection
from sklearn.model_selection import StratifiedKFold  ## For feature selection
from sklearn.linear_model import LogisticRegression  ## For feature selection
warnings.simplefilter(action='ignore', category=Warning)  ## Suppress warnings
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc, roc_curve, recall_score,
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
        # Preprocess the input data (you may need to adjust this based on your preprocessing during training)
        churn_dataset = pd.read_csv('CustomerChurn_dataset.csv')

        # Renaming column: gender to Gender
        churn_dataset.rename(columns={'gender': 'Gender', }, inplace=True)

        # Renaming column: tenure to Tenure
        churn_dataset.rename(columns={'tenure': 'Tenure', }, inplace=True)

        # Convert the data type of the 'TotalCharges' column from object to numeric
        churn_dataset['TotalCharges'] = churn_dataset['TotalCharges'].apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna()

        # Convert 'No internet service' to 'No' for the below-mentioned columns
        cols = ['OnlineBackup', 'StreamingMovies', 'DeviceProtection',
                'TechSupport', 'OnlineSecurity', 'StreamingTV']
        for i in cols:
            churn_dataset[i] = churn_dataset[i].replace({'No internet service': 'No'})

        # Convert 'No phone service' to 'No' for the below-mentioned column
        churn_dataset['MultipleLines'] = churn_dataset['MultipleLines'].replace({'No phone service': 'No'})

        # Drop 'customerID' column since it is not useful for churn prediction as the feature is used for identification of customers.
        selected_features = churn_dataset.drop(['customerID'], axis=1)

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
        selected_features['TotalCharges'] = selected_features['TotalCharges'].fillna(
            selected_features['TotalCharges'].median())

        # Perform Feature Scaling on 'tenure', 'MonthlyCharges' in order to bring them on the same scale
        standardScaler = StandardScaler()
        columns_for_scaling = ['SeniorCitizen', 'Tenure', 'MonthlyCharges', 'TotalCharges']

        # Apply the feature scaling operation on the dataset using fit_transform() method
        selected_features[columns_for_scaling] = standardScaler.fit_transform(selected_features[columns_for_scaling])

        final_selected_features = ['SeniorCitizen', 'Dependents', 'Tenure', 'PhoneService', 'MultipleLines',
                                   'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                   'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'MonthlyCharges',
                                   'TotalCharges']
        y = selected_features['Churn']
        X = selected_features[final_selected_features]

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        oversample = SMOTE(k_neighbors=5)
        X_smote, y_smote = oversample.fit_resample(X_train, y_train)
        X_train, y_train = X_smote, y_smote

        ## Define the architecture of the neural network using the Functional API
        input_layer = Input(shape=(X_train.shape[1],))
        hidden_layer_1 = Dense(64, activation='relu')(input_layer)
        dropout_1 = Dropout(0.5)(hidden_layer_1)
        hidden_layer_2 = Dense(32, activation='relu')(dropout_1)
        output_layer = Dense(1, activation='sigmoid')(hidden_layer_2)

        ## Create the model
        model = Model(inputs=input_layer, outputs=output_layer)

        ## Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        ## Train the model
        history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

        ## Display the model summary
        print(model.summary())

        ## Evaluate the model
        y_pred = model.predict(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int)

        ## Calculate accuracy and AUC score
        accuracy = accuracy_score(y_test, y_pred_binary)
        auc_score = roc_auc_score(y_test, y_pred)

        print(f"\nAccuracy: {accuracy}")
        print(f"\nAUC Score: {auc_score}")

        ## Evaluate the model's accuracy and calculate the AUC value.
        y_pred = model.predict(X_test)
        predictions = [np.round(value) for value in y_pred]

        ## Evaluate predictions
        accuracy = accuracy_score(y_test, predictions)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))

        ## Calculate the AUC
        fpr, tpr, thresholds = roc_curve(y_test, predictions)
        roc_auc = auc(fpr, tpr)

        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, label='AUC = %0.4f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([-0.001, 1])
        plt.ylim([0, 1.001])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

        ## Calculating the confidence factor
        cofidence_factor = 2.58 * sqrt((accuracy * (1 - accuracy)) / y_test.shape[0])
        cofidence_factor

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
    Contract = st.selectbox(' Customer has a contract:', ['month-to-month', 'one_year', 'two_year'])
    PaperlessBilling = st.selectbox(' Customer has a paperlessbilling:', ['yes', 'no'])

    Tenure = st.number_input(
        'Number of months the customer has been with the current telco provider:', min_value=0, max_value=240, value=0)
    MonthlyCharges = st.number_input('Monthly charges:', min_value=0, max_value=240, value=0)
    TotalCharges = Tenure * MonthlyCharges

    # Make prediction when the user clicks the button
    if st.button("Predict"):
        features = [float(SeniorCitizen), float(Dependents), float(PhoneService), float(MultipleLines), float(OnlineSecurity), float(OnlineBackup), float(DeviceProtection), float(TechSupport),       float(StreamingTV), float(StreamingMovies), float(Contract), float(PaperlessBilling), float(MonthlyCharges), float(TotalCharges), float(Tenure)]  
        churn_probability = predict_churn(features)

        # Display the prediction result
        st.success(f"Churn Probability: {churn_probability}")


if __name__ == "__main__":
    main()
