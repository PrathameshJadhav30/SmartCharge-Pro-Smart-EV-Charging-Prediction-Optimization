from flask import Flask, request, render_template, jsonify
import pandas as pd
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Load model, scaler, and encoder
with open("D:/Ev Charging Time Prediction/models/ev_charging_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("D:/Ev Charging Time Prediction/models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("D:/Ev Charging Time Prediction/models/encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Define the home page
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# Define the prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user input from the request
        battery_capacity = float(request.form.get("battery_capacity"))
        current_battery = float(request.form.get("current_battery"))
        charging_power = float(request.form.get("charging_power"))
        temperature = float(request.form.get("temperature"))
        charging_type = request.form.get("charging_type")

        # Create a DataFrame with user input
        input_data = pd.DataFrame([[battery_capacity, current_battery, charging_power, temperature, charging_type]],
                                  columns=['battery_capacity', 'current_battery_level', 'charging_power', 'temperature', 'charging_station_type'])

        # One-hot encode the 'charging_station_type' column
        encoded_type = encoder.transform(input_data[['charging_station_type']])
        encoded_df = pd.DataFrame(encoded_type, columns=encoder.get_feature_names_out(['charging_station_type']))

        # Drop the original 'charging_station_type' and join the encoded columns
        input_data = input_data.drop(columns=['charging_station_type']).join(encoded_df)

        # Scale the numerical features
        input_data.iloc[:, :-1] = scaler.transform(input_data.iloc[:, :-1])

        # Make the prediction
        prediction = model.predict(input_data)

        # Convert the prediction to a standard Python float (this ensures it's JSON serializable)
        prediction_value = float(prediction[0])

        # Add optimization suggestions
        optimization_suggestions = get_optimization_suggestions(battery_capacity, current_battery, charging_power, temperature, charging_type, prediction_value)

        return render_template("result.html", prediction=round(prediction_value, 2), suggestions=optimization_suggestions)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

def get_optimization_suggestions(battery_capacity, current_battery, charging_power, temperature, charging_type, predicted_time):
    suggestions = []

    # Optimization Tips Based on Predicted Charging Time (in minutes)
    if predicted_time < 15:
        suggestions.append("Charging time is quick, but make sure your battery is not being overcharged.")
        if charging_power < 50:
            suggestions.append("Consider using a DC Fast Charger to further reduce charging time.")
        if charging_type == 'AC_Slow':
            suggestions.append("AC Slow chargers are fine but consider upgrading to DC for quicker charging.")
    elif 15 <= predicted_time < 45:
        suggestions.append("You can optimize the charging time by considering the time of day. Charge during off-peak hours to reduce power grid congestion.")
        if charging_power < 100:
            suggestions.append("Increase charging power if possible, but ensure the charger supports it.")
        if temperature < 5 or temperature > 35:
            suggestions.append("Charging is slower in extreme temperatures. Try to charge within a more moderate temperature range.")
        if charging_type == 'AC_Slow':
            suggestions.append("Switch to DC Fast Charger for reducing charging time in urgent situations.")
    elif 45 <= predicted_time < 75:
        suggestions.append("You may want to charge overnight or during extended periods when you're not in a rush.")
        if charging_power < 50:
            suggestions.append("Switching to a DC Fast Charger could drastically reduce your charging time.")
        if charging_type == 'AC_Slow':
            suggestions.append("AC Slow charging can work fine for regular use but is not ideal when you need to save time.")
        if temperature < 5 or temperature > 35:
            suggestions.append("Extreme temperatures are affecting charging efficiency. Try to charge in optimal temperature conditions (between 20-25°C).")
    elif predicted_time >= 75:
        suggestions.append("For very long charging times, plan ahead and charge during the night or when you're not in a rush.")
        if charging_power < 100:
            suggestions.append("Consider upgrading to a higher-power charger to speed up the charging process.")
        if current_battery < 20:
            suggestions.append("Low battery levels will naturally take longer to charge. Consider starting charging earlier in the day to avoid delays.")
        if charging_type == 'AC_Slow':
            suggestions.append("AC Slow charging is too slow for such high charging times. Switch to DC Fast Charging if you're in a hurry.")
        if temperature < 5 or temperature > 35:
            suggestions.append("Charging will be significantly slower in extreme temperatures. Charge in moderate temperature conditions for better efficiency.")

    if charging_power < 50:
        suggestions.append("Consider using a DC Fast Charger for faster charging.")
    if current_battery < 20:
        suggestions.append("Charging from low battery levels may take longer. Charge early when possible.")
    if charging_type == 'AC_Slow':
        suggestions.append("AC Slow chargers are better for long-term, non-time-sensitive charging.")
    if temperature < 5 or temperature > 35:
        suggestions.append("Charging efficiency is lower in extreme temperatures. Avoid charging in very hot or cold conditions.")

    return suggestions

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
