import pandas as pd
from pydantic_settings import BaseSettings
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)

data = pd.read_csv('/home/carrington/Desktop/dataScience/Financial_inclusion_dataset.csv')

'''print(data.head(5))

print(data.info())

report = ProfileReport(data, minimal=True)
report.to_file("/home/carrington/Desktop/dataScience/financeAfrika_profile.html")

print(data.isnull().sum())

print(data.duplicated().sum())'''

encoder = LabelEncoder()
#data = data.sample(n = 10000, random_state=42)
#cols = ['country','uniqueid','bank_account','location_type','cellphone_access','gender_of_respondent','relationship_with_head','marital_status','education_level','job_type'])
# Apply label encoding to each categorical column
for col in data.select_dtypes(include=['object']).columns:
    data[col] = encoder.fit_transform(data[col])

z_scores = stats.zscore(data)
data = data[(z_scores < 3).all(axis=1)]
#print(data.shape)

# Feature selection and preprocessing (consider feature engineering)
data = data.sample(n=2500, random_state=42)
X = data.drop('uniqueid', axis=1)
y = data['uniqueid']  # "uniqueid" is the target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (optional)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train your machine learning model
model = RandomForestClassifier()  # Replace with your chosen model
model.fit(X_train, y_train)

features = ['job_type', 'age_of_respondent', 'location_type']
# Title and header
st.title("Financial Inclusion Prediction")
st.header("Predict your likelihood of using a bank account")

user_input = {}
for feature in features:
    user_input[feature] = st.number_input(f"{feature}")

scaler = StandardScaler()

# Prediction button
if st.button("Predict Bank Account Usage"):

    # Create a DataFrame from user input
    user_data = pd.DataFrame([user_input])

    # Preprocess user input (e.g., scaling)
    user_data_scaled = scaler.transform(user_data)

    # Make prediction
    prediction = model.predict(user_data_scaled)

    # Display prediction result
    if prediction[0]:
        st.success("You are likely to have a bank account")
    else:
        st.error("You are less likely to have a bank account")
