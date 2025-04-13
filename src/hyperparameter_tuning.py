import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor

# Load preprocessed data
df = pd.read_csv("D:\Ev Charging Time Prediction\dataset\preprocessed_data.csv")
X = df.drop(columns=['charging_time_minutes'])
y = df['charging_time_minutes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model and hyperparameters
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 6, 8]
}

grid_search = GridSearchCV(XGBRegressor(random_state=42), param_grid, cv=3, scoring='r2', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# Save optimized model
with open("D:\Ev Charging Time Prediction\models\ev_charging_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
