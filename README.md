# Customer_Churn_Prediction_Using_MachineLearning

**Project Overview**

This project aims to predict customer churn in the telecom industry using machine learning models. The dataset consists of various customer-related attributes, and the goal is to identify customers who are likely to leave.

**Technologies Used**

Python

Pandas

NumPy

Scikit-learn

XGBoost

Seaborn

Matplotlib

Imbalanced-learn (SMOTE)

Pickle

---

**Dataset**

The dataset (WA_Fn-UseC_-Telco-Customer-Churn.csv) contains information about telecom customers, including their demographics, account details, and usage patterns.

---

**Steps Involved**

**1. Data Loading & Preprocessing**

Load the dataset from a CSV file.

Drop irrelevant columns like customerID.

Handle missing values.

Convert categorical features to numerical using Label Encoding.

Balance the dataset using SMOTE to address class imbalance.

**2. Model Training**

Split the dataset into training and testing sets.

Train multiple classification models:

Decision Tree Classifier

Random Forest Classifier

XGBoost Classifier

Perform hyperparameter tuning.

**3. Model Evaluation**

Evaluate models based on:

Accuracy Score

Confusion Matrix

Classification Report

**4. Model Deployment**

Save the trained model using Pickle for future use.

---

**Installation & Usage**
1.Clone the repository:

git clone https://github.com/yourusername/Customer_Churn_Prediction.git

cd Customer_Churn_Prediction

2.Install required dependencies:

pip install -r requirements.txt

3.Run the Jupyter Notebook to train and evaluate the model.

4.Load the saved model and use it for predictions.

---

**Results**

The trained model helps in identifying potential churners with high accuracy. The best model is selected based on evaluation metrics.


