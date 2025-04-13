import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle

# Load dataset
df = pd.read_csv("D:\Ev Charging Time Prediction\dataset\ev_charging_data.csv")

# Feature selection
features = ['battery_capacity', 'current_battery_level', 'charging_power', 'temperature', 'charging_station_type']
target = 'charging_time_minutes'

# One-hot encode categorical features
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_features = encoder.fit_transform(df[['charging_station_type']])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['charging_station_type']))
df = df.drop(columns=['charging_station_type']).join(encoded_df)

# Scaling numerical features
scaler = StandardScaler()
df[features[:-1]] = scaler.fit_transform(df[features[:-1]])

# # Save preprocessing models
with open("D:\Ev Charging Time Prediction\models\scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("D:\Ev Charging Time Prediction\models\encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

# Save preprocessed data
df.to_csv("dataset/preprocessed_data.csv", index=False)
