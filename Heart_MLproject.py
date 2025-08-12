import streamlit as st
import pandas as pd
import joblib

# Load models and scalers
model = joblib.load("LR_Heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("Columns.pkl")

st.title("Heart Disease prediction by Khushi🌟")
st.markdown("Provide the following details")

# User inputs
age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("SEX", ['M', 'F'])
chest_pain = st.selectbox("Chest pain type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting blood pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 300, 200)
fasting_bs = st.selectbox("Fasting blood sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise induced angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# Predict button
if st.button("Predict"):
    # Create raw input dictionary
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'OldPeak': oldpeak,
        'Sex': sex,
        'ChestPainType' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1,
    }

    # Create DataFrame
    input_df = pd.DataFrame([raw_input])

    # Add missing columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Arrange columns in expected order
    input_df = input_df[expected_columns]

    # Scale input and predict
    scale_input = scaler.transform(input_df)
    prediction = model.predict(scale_input)[0]

    # Output result
    if prediction == 1:
        st.error("High Risk of heart disease")
    else:
        st.success("Low risk of heart disease")
