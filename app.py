# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
pip install scikit-learn==1.5.1

# Load saved pipeline
model = joblib.load('best_rf_model.pkl')

st.title("Stroke Prediction App")

# User inputs
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
age = st.number_input("Age", min_value=0)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
ever_married = st.selectbox("Ever Married", ["Yes", "No"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0)
bmi = st.number_input("BMI", min_value=0.0)
smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

gender_map = {"Female": 0, "Male": 1, "Other": 2}
ever_married_map = {"No": 0, "Yes": 1}
work_type_map = {
    "Private": 2,
    "Self-employed": 3,
    "Govt_job": 0,
    "children": 4,
    "Never_worked": 1
}
residence_map = {"Urban": 1, "Rural": 0}
smoking_status_map = {
    "never smoked": 2,
    "formerly smoked": 1,
    "smokes": 3,
    "Unknown": 0
}

input_data = pd.DataFrame({
    'gender': [gender_map[gender]],
    'age': [age],
    'hypertension': [hypertension],
    'heart_disease': [heart_disease],
    'ever_married': [ever_married_map[ever_married]],
    'work_type': [work_type_map[work_type]],  
    'Residence_type': [residence_map[residence_type]],
    'avg_glucose_level': [avg_glucose_level],
    'bmi': [bmi],
    'smoking_status': [smoking_status_map[smoking_status]]
})

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    st.write(f"### Prediction: {'Stroke' if prediction == 1 else 'No Stroke'}")
    st.write(f"### Probability of Stroke: {probability:.2%}")

