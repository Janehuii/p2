# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import requests

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
    body {
        background-color: rgba(173, 216, 230, 0.5); 
    }
    .stApp {
        background-color: rgba(173, 216, 230, 0.5);
    }
    .css-1d391kg {
        background-color: #ffffff; /* Card color */
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
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
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"])
bmi_input = st.number_input("BMI", min_value=0.0)
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Average Glucose Level (mmol/L)", min_value=0.0)
smoking_status = st.selectbox("Smoking Status", ["Never smoked", "Formerly smoked", "Smokes", "Unknown"])
hypertension = st.selectbox("Hypertension", ["No", "Yes"])
heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])

gender_map = {"Other": 0, "Male": 1, "Female": 2}
ever_married_map = {"No": 0, "Yes": 1}
work_type_map = {"Govt_job": 0, "Never_worked": 1,"Private": 2,"Self-employed": 3,"Children": 4,}
residence_map = {"Rural": 0,"Urban": 1}
smoking_status_map = { "Unknown": 0,  "Formerly smoked": 1, "Never smoked": 2,"Smokes": 3}
bmi=0
avg_glucose_level = avg_glucose_level * 18.018
hypertension = 1 if hypertension == "Yes" else 0
heart_disease = 1 if heart_disease == "Yes" else 0


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

    if prediction == 1:
        # Play alert sound for high risk
        st.markdown(
            """
            <audio autoplay>
                <source src="https://cdn.pixabay.com/download/audio/2025/05/23/audio_67ba0ac874.mp3?filename=alerte-346112.mp3" type="audio/mpeg">
            </audio>
            """,
            unsafe_allow_html=True
        )

        # Create two columns
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(
                "<p style='color:red; font-size:24px; font-weight:bold;'>🔴 Prediction: High Risk of Stroke</p>",
                unsafe_allow_html=True
            )
        with col2:
            st.image(
                "https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExZnZ0Z2F1OHNueWozZHh3YW9vbTNiZTdjY2txeGhwZWMwaDBweHh1MSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/Mjk31gLRH3uQnKCsGX/giphy.gif",
                use_container_width=True
            )
    else:
        # Low risk output
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(
                "<p style='color:green; font-size:24px; font-weight:bold;'>🟢 Prediction: Low Risk of Stroke</p>",
                unsafe_allow_html=True
            )
        with col2:
            st.image(
                "https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExZmg0b2twZXQzend6MDNzODI4dnUycGN1dGpwNXZ4NjY4dWl3MXg1cSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/LpiVeIRgrqVsZJpM5H/giphy.gif",
                use_container_width=True
            )
