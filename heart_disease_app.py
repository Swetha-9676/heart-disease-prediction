import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("heart_disease.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("❤️ Heart Disease Prediction System")
st.write("Enter patient medical details")

# ====== PASTE HERE ======
age = st.number_input("Age", min_value=1, max_value=120, value=30)

cp = st.selectbox("Chest Pain Type", [1, 2, 3, 4])

trestbps = st.number_input("Resting Blood Pressure", value=120)
chol = st.number_input("Serum Cholesterol", value=200)

fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])

restecg = st.selectbox("Resting ECG", [0, 1, 2])

thalach = st.number_input("Maximum Heart Rate Achieved", value=150)

exang = st.selectbox("Exercise Induced Angina", [0, 1])

oldpeak = st.number_input("Oldpeak (ST Depression)", value=1.0)

slope = st.selectbox("ST Slope", [1, 2, 3])
# ====== STOP HERE ======

# Prediction
if st.button("Predict"):
    input_data = np.array([[age, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak, slope]])

    input_data = scaler.transform(input_data)  # 🔥 THIS LINE FIXES EVERYTHING

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease Detected")

