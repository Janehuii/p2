# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load saved pipeline
model = joblib.load('rfmodel.pkl')
# Set Streamlit page config
st.set_page_config(
    page_title="Stroke Prediction App",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://i.imgur.com/f4dXw5U.png');  /* Replace with your cartoon image URL */
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
        padding: 10px;
    }
    .css-1d391kg {
        background-color: rgba(255, 255, 255, 0.85); /* Semi-transparent card */
        border-radius: 12px;
        padding: 20px;
    }
    .stButton>button {
        background-color: #ff6f61;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 1.5em;
    }
    .stButton>button:hover {
        background-color: #e53935;
    }
    .prediction-text {
        font-size: 24px;
        font-weight: 700;
        color: #d32f2f;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("Stroke Prediction App")

# User inputs
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
age = st.number_input("Age", min_value=0)
ever_married = st.selectbox("Ever Married", ["Yes", "No"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
bmi_input = st.number_input("BMI", min_value=0.0)
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Average Glucose Level (mg/dL)", min_value=0.0)
smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])
hypertension = st.selectbox("Hypertension", ["No", "Yes"])
heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])

gender_map = {"Female": 0, "Male": 1, "Other": 2}
ever_married_map = {"No": 0, "Yes": 1}
work_type_map = {"Govt_job": 0, "Never_worked": 1,"Private": 2,"Self-employed": 3,"children": 4,}
residence_map = {"Rural": 0,"Urban": 1}
smoking_status_map = { "Unknown": 0,  "formerly smoked": 1, "never smoked": 2,"smokes": 3}
hypertension = 1 if hypertension == "Yes" else 0
heart_disease = 1 if heart_disease == "Yes" else 0
bmi= 0

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
    
    result = "🟥 High Risk of Stroke" if prediction == 1 else "🟩 Low Risk of Stroke"
    st.subheader(f"Prediction: {result}")


