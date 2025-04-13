import pandas as pd
import pickle
import sys

# Load model, scaler, and encoder
with open("D:\Ev Charging Time Prediction\models\ev_charging_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("D:\Ev Charging Time Prediction\models\scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("D:\Ev Charging Time Prediction\models\encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Predict function
def predict_charging_time(battery_capacity, current_battery, charging_power, temperature, charging_station_type):
    input_data = pd.DataFrame({
        'battery_capacity': [battery_capacity],
        'current_battery_level': [current_battery],
        'charging_power': [charging_power],
        'temperature': [temperature]
    })

    # Encode categorical variable
    encoded_type = encoder.transform([[charging_station_type]])
    encoded_df = pd.DataFrame(encoded_type, columns=encoder.get_feature_names_out(['charging_station_type']))
    input_data = input_data.join(encoded_df)

    # Scale numerical features
    input_data[["battery_capacity", "current_battery_level", "charging_power", "temperature"]] = scaler.transform(
        input_data[["battery_capacity", "current_battery_level", "charging_power", "temperature"]])

    # Predict
    prediction = model.predict(input_data)
    return prediction[0]

if __name__ == "__main__":
    battery_capacity = float(sys.argv[1])
    current_battery = float(sys.argv[2])
    charging_power = float(sys.argv[3])
    temperature = float(sys.argv[4])
    charging_station_type = sys.argv[5]
    
    result = predict_charging_time(battery_capacity, current_battery, charging_power, temperature, charging_station_type)
    print(f"Predicted Charging Time: {result:.2f} minutes")
