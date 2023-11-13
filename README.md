-- Telecom Customer Churn Prediction Model --

## Introduction

Customer attrition is one of the biggest expenditures of any organization. Customer churn otherwise known as customer attrition or customer turnover is the percentage of customers that stopped using your company's product or service within a specified timeframe. For instance, if you began the year with 500 customers but later ended with 480 customers, the percentage of customers that left would be 4%. If we could figure out why a customer leaves and when they leave with reasonable accuracy, it would immensely help the organization to strategize their retention initiatives manifold.

** Overview **
This project aims to predict customer churn in a fictional telecommunication use case using machine learning. The predictive model is based on a Multi-Layer Perceptron (MLP) trained on customer data to predict whether a customer is likely to terminate their service or not. 

Link to my model deploymnet video: [https://youtu.be/7sVhixurWCY]

## Model Features

- **Input Features:**
  - The model takes into account various customer demographic data.
  - Payment information and service subscriptions (e.g., online security, streaming services) are considered.

- **Target Variable:**
  - The target variable is binary, representing whether a customer churned (1) or not (0).

- **Training Technique:**
  - The model is trained using the Functional API of TensorFlow.

## Model Evaluation

- **Performance Metrics:**
  - The model's accuracy and Area Under the Curve (AUC) score are evaluated on a test dataset.

- **Oversampling:**
  - The training dataset is oversampled using SMOTE to address class imbalance.

## Code Structure

- The project includes codes for data exploration, preprocessing, model training, and evaluation.

## Usage

- The model can be use for predicting customer churn in various scenarios by providing relevant customer data.

## Dependencies
- Python
- TensorFlow
- Scikit-learn
- Pandas
- NumPy
- Seaborn
- Matplotlib



Conclusion:
In conclusion, this project successfully developed a churn prediction model using a Multi-Layer Perceptron model (Functional API). By applying feature selection, and model evaluation, the performance of the model was improved compared to the baseline. It is important to note that this project represents a specific approach to churn prediction. Further improvements and experimentation, such as using more advanced techniques in deep learning could potentially enhance the model's performance.


## Acknowledgments
- This project was developed as part of Ashesi University's coursework in Introduction to AI.


## Author/
- Faisal Alidu

---

