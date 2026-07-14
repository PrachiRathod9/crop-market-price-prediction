import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Crop Market Price Prediction",
    page_icon="🌾",
    layout="centered"
)

st.title("🌾 Crop Market Price Prediction")
st.write("Predict the estimated market price of crops using Machine Learning.")

# Check if required model files exist
required_files = [
    "model.pkl",
    "label_encoder.pkl",
    "state_encoder.pkl",
    "scaler.pkl"
]

missing = [f for f in required_files if not os.path.exists(f)]

if missing:
    st.error("Missing model files:")
    st.write(missing)
    st.stop()

# Load models
model = joblib.load("model.pkl")
crop_encoder = joblib.load("label_encoder.pkl")
state_encoder = joblib.load("state_encoder.pkl")
scaler = joblib.load("scaler.pkl")

# User Inputs
crop = st.selectbox(
    "Select Crop",
    crop_encoder.classes_
)

state = st.selectbox(
    "Select State",
    state_encoder.classes_
)

month = st.slider(
    "Month",
    1,
    12,
    1
)

rainfall = st.number_input(
    "Rainfall (mm)",
    min_value=0.0,
    max_value=500.0,
    value=120.0
)

temperature = st.number_input(
    "Temperature (°C)",
    min_value=0.0,
    max_value=50.0,
    value=25.0
)

demand = st.slider(
    "Demand Index",
    0,
    100,
    70
)

if st.button("Predict Price"):

    crop_value = crop_encoder.transform([crop])[0]
    state_value = state_encoder.transform([state])[0]

    input_data = pd.DataFrame(
        [[
            crop_value,
            state_value,
            month,
            rainfall,
            temperature,
            demand
        ]],
        columns=[
            "Crop",
            "State",
            "Month",
            "Rainfall",
            "Temperature",
            "Demand"
        ]
    )

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    st.success(f"Estimated Market Price: ₹ {prediction:.2f} per Quintal")
