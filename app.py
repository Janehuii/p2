import streamlit as st
import numpy as np
import pandas as pd
import joblib 

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
