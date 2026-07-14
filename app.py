import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Crop Market Price Prediction",layout="centered")

st.title("🌾 Crop Market Price Prediction")

model = joblib.load("model.pkl")
crop_encoder = joblib.load("label_encoder.pkl")
state_encoder = joblib.load("state_encoder.pkl")
scaler = joblib.load("scaler.pkl")

crop = st.selectbox("Select Crop",crop_encoder.classes_)

state = st.selectbox("Select State",state_encoder.classes_)

month = st.slider("Month",1,12,1)

rainfall = st.number_input("Rainfall (mm)",50.0,300.0,120.0)

temperature = st.number_input("Temperature (°C)",10.0,45.0,25.0)

demand = st.slider("Demand Index",50,100,70)

if st.button("Predict Price"):

    crop_value = crop_encoder.transform([crop])[0]
    state_value = state_encoder.transform([state])[0]

    data = pd.DataFrame([[crop_value,state_value,month,rainfall,temperature,demand]],
                        columns=["Crop","State","Month","Rainfall","Temperature","Demand"])

    data = scaler.transform(data)

    prediction = model.predict(data)[0]

    st.success(f"Estimated Market Price : ₹ {prediction:.2f} per Quintal")
