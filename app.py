import streamlit as st
import numpy as np
import pandas as pd
import joblib 
from sklearn.preprocessing import LabelEncoder

# Load the saved Random Forest model
model = joblib.load('best_rf_model.pkl')

st.title("Stroke Prediction")

st.write("Enter the input details below:")


# Collect user input
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
age = st.number_input("Age", min_value=0)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
ever_married = st.selectbox("Ever Married", ["Yes", "No"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Average Glucose Level")
bmi = st.number_input("BMI")
smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

# Initialize a label encoder object
label_encoders = {}

# Loop through each categorical column and encode
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # Save the encoder for potential inverse transformations later

# List of numerical columns to scale
numerical_cols = ['age', 'avg_glucose_level', 'bmi']

# Initialize the scaler
scaler = StandardScaler()

# Apply the scaler to the numerical features
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# Split the dataset into training and validation sets (80% training, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply ADASYN to balance the classes
adasyn = ADASYN(random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X, y)
# Balance the data
for train_idx, test_idx in skf.split(X_resampled, y_resampled):
    X_train, X_test = X_resampled.iloc[train_idx], X_resampled.iloc[test_idx]
    y_train, y_test = y_resampled.iloc[train_idx], y_resampled.iloc[test_idx]


# Convert input into DataFrame (make sure it matches training features)
input_data = pd.DataFrame({
    'gender': [gender],
    'age': [age],
    'hypertension': [hypertension],
    'heart_disease': [heart_disease],
    'ever_married': [ever_married],
    'work_type': [work_type],
    'Residence_type': [residence_type],
    'avg_glucose_level': [avg_glucose_level],
    'bmi': [bmi],
    'smoking_status': [smoking_status]
})


# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]  # Probability of stroke

    st.write(f"### Prediction: {'Stroke' if prediction[0] == 1 else 'No Stroke'}")
    st.write(f"### Probability of Stroke: {probability:.2%}")
