This repository contains all my machine learning exercises and small projects.

***Learning Outcomes*** 
* learn how to preprocess data for modeling.
* Skill to know when to use the appropriate models.
* Pandas,Numpy and matplotib skills 
* Develop an appreciation for what is involved in Learning models from data
* Understand a wide variety of learning algorithms
* Understand how to evaluate models generated from data
* Apply the algorithms to a real problem, optimize the models learned and report on the expected accuracy that can be achieved by applying the models

# Classification -- Predicting Customer Churn

Introduction

Customer attrition is one of the biggest expenditures of any organization. Customer churn otherwise known as customer attrition or customer turnover is the percentage of customers that stopped using your company's product or service within a specified timeframe.
For instance, if you began the year with 500 customers but later ended with 480 customers, the percentage of customers that left would be 4%. If we could figure out why a customer leaves and when they leave with reasonable accuracy, it would immensely help the organization to strategize their retention initiatives manifold.

In this project, the aim is to find the likelihood of a customer leaving the organization, the key indicators of churn as well as the retention strategies that can be implemented to avert this problem.

Data Understanding

The data for this project is in a csv format. The following describes the columns present in the data.

Gender -- Whether the customer is a male or a female

SeniorCitizen -- Whether a customer is a senior citizen or not

Partner -- Whether the customer has a partner or not (Yes, No)

Dependents -- Whether the customer has dependents or not (Yes, No)

Tenure -- Number of months the customer has stayed with the company

Phone Service -- Whether the customer has a phone service or not (Yes, No)

MultipleLines -- Whether the customer has multiple lines or not

InternetService -- Customer's internet service provider (DSL, Fiber Optic, No)

OnlineSecurity -- Whether the customer has online security or not (Yes, No, No Internet)

OnlineBackup -- Whether the customer has online backup or not (Yes, No, No Internet)

DeviceProtection -- Whether the customer has device protection or not (Yes, No, No internet service)

TechSupport -- Whether the customer has tech support or not (Yes, No, No internet)

StreamingTV -- Whether the customer has streaming TV or not (Yes, No, No internet service)

StreamingMovies -- Whether the customer has streaming movies or not (Yes, No, No Internet service)

Contract -- The contract term of the customer (Month-to-Month, One year, Two year)

PaperlessBilling -- Whether the customer has paperless billing or not (Yes, No)

Payment Method -- The customer's payment method (Electronic check, mailed check, Bank transfer(automatic), Credit card(automatic))

MonthlyCharges -- The amount charged to the customer monthly

TotalCharges -- The total amount charged to the customer

Churn -- Whether the customer churned or not (Yes or No)

Instructions

The task is to understand the data and prepare it for model building. The analysis or methods should incorporate the following steps.

Hypothesis formation and Data Processing - Importing the relevant libraries and modules, Cleaning of Data, Check data types, Encoding Data labels etc.

Data Evaluation -- Perform bivariate and multivariate analysis, EDA

Useful resources [ Exploratory Data Analysis: Univariate, Bivariate, and Multivariate Analysis , Univariate, Bivariate and Multivariate Analysis , Exploratory Data Analysis (EDA) Using Python]

Build & Select Model -- Train Model on dataset and select the best performing model.

Evaluate your chosen Model.

Model Improvement.

Future Predictions.

Key Insights and Conclusion.

Upon completion of the project, I'll write a blog post on my thought process on medium, LinkedIn, personal blog, or any other suitable blogging site.

Conclusion:
In conclusion, this project successfully developed a churn prediction model using logistic regression. By applying feature selection, hyperparameter tuning, and model evaluation, the performance of the model was improved compared to the baseline. The final logistic regression model achieved an F1-score of 72.4% on the test set.

The model demonstrated the ability to distinguish between churned and non-churned customers with satisfactory discriminatory power. The selected features, such as gender, device protection, senior citizenship, contract type, payment method, and tenure, played significant roles in predicting customer churn.

It's important to note that this project represents a specific approach to churn prediction using logistic regression. Further improvements and experimentation, such as feature engineering, trying different classification algorithms, or using more advanced techniques like deep learning, could potentially enhance the model's performance.



# Author
Faisal Alidu