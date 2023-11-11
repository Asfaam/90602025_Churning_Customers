import pickle
import streamlit as st
import pandas as pd
from PIL import Image

# Load the trained model
model = pickle.load(open('customer_churn.pkl', 'rb'))
 
def main():
    image = Image.open('images/icone.png') 
    image2 = Image.open('images/image.png')
    st.image(image,use_column_width=False) 
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))
    st.sidebar.info('This app is created to predict Customer Churn')
    st.sidebar.image(image2)
    st.title("Predicting Customer Churn")
    if add_selectbox == 'Online':
        
        seniorcitizen= st.selectbox(' Customer is a senior citizen:', [0, 1])
        
        dependents = st.selectbox(' Customer has  dependents:', ['yes', 'no'])
        phoneservice = st.selectbox(' Customer has phoneservice:', ['yes', 'no'])
        multiplelines = st.selectbox(' Customer has multiplelines:', ['yes', 'no'])
        
        onlinesecurity= st.selectbox(' Customer has onlinesecurity:', ['yes', 'no'])
        onlinebackup = st.selectbox(' Customer has onlinebackup:', ['yes', 'no'])
        deviceprotection = st.selectbox(' Customer has deviceprotection:', ['yes', 'no'])
        techsupport = st.selectbox(' Customer has techsupport:', ['yes', 'no'])
        streamingtv = st.selectbox(' Customer has streamingtv:', ['yes', 'no'])
        streamingmovies = st.selectbox(' Customer has streamingmovies:', ['yes', 'no'])
        contract= st.selectbox(' Customer has a contract:', ['month-to-month', 'one_year', 'two_year'])
        paperlessbilling = st.selectbox(' Customer has a paperlessbilling:', ['yes', 'no'])
        
        tenure = st.number_input('Number of months the customer has been with the current telco provider :', min_value=0, max_value=240, value=0)
        monthlycharges= st.number_input('Monthly charges :', min_value=0, max_value=240, value=0)
        totalcharges = tenure*monthlycharges
        output= ""
        output_prob = ""
        input_dict={
                "seniorcitizen": seniorcitizen,
               
                "dependents": dependents,
                "phoneservice": phoneservice,
                "multiplelines": multiplelines,
                
                "onlinesecurity": onlinesecurity,
                "onlinebackup": onlinebackup,
                "deviceprotection": deviceprotection,
                "techsupport": techsupport,
                "streamingtv": streamingtv,
                "streamingmovies": streamingmovies,
                "contract": contract,
                "paperlessbilling": paperlessbilling,
                
                "tenure": tenure,
                "monthlycharges": monthlycharges,
                "totalcharges": totalcharges
            }
        if st.button("Predict"):
            
            X = [input_dict]
            y_pred = model.predict_proba(X)[0, 1]
            churn = y_pred >= 0.5
            output_prob = float(y_pred)
            output = bool(churn)
  
        st.success('Churn: {0}, Risk Score: {1}'.format(output, output_prob))
 
    if add_selectbox == 'Batch':
 
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
        if file_upload is not None:
            data = pd.read_csv(file_upload)
            X = [data]
            y_pred = model.predict_proba(X)[0, 1]
            churn = y_pred >= 0.5
            churn = bool(churn)
            st.write(churn)
 
 
if __name__ == '__main__':
    main()