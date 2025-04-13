import streamlit as st
import pandas as pd
import pickle

# Load model, scaler, and encoder
with open("D:\Ev Charging Time Prediction\models\ev_charging_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("D:\Ev Charging Time Prediction\models\scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("D:\Ev Charging Time Prediction\models\encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

st.title("EV Charging Time Prediction & Optimization")

# User Inputs
battery_capacity = st.number_input("Battery Capacity (kWh)", min_value=30, max_value=100, step=1)
current_battery = st.number_input("Current Battery Level (%)", min_value=0, max_value=100, step=1)
charging_power = st.number_input("Charging Power (kW)", min_value=5, max_value=350, step=5)
temperature = st.number_input("Temperature (°C)", min_value=-10, max_value=50, step=1)
charging_type = st.selectbox("Charging Station Type", ['AC_Slow', 'DC_Fast'])

if st.button("Predict Charging Time"):
    input_data = pd.DataFrame([[battery_capacity, current_battery, charging_power, temperature, charging_type]],
                              columns=['battery_capacity', 'current_battery_level', 'charging_power', 'temperature', 'charging_station_type'])
    encoded_type = encoder.transform([[charging_type]])
    input_data = input_data.drop(columns=['charging_station_type']).join(pd.DataFrame(encoded_type, columns=encoder.get_feature_names_out(['charging_station_type'])))
    input_data.iloc[:, :-1] = scaler.transform(input_data.iloc[:, :-1])
    
    prediction = model.predict(input_data)
    st.success(f"Predicted Charging Time: {prediction[0]:.2f} minutes")
