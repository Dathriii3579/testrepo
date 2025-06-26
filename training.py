import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

# Load CSV files
files = [
    '/Users/dathril/Documents/HAR/patient monitoring/dataset/har70plus/503.csv',
    '/Users/dathril/Documents/HAR/patient monitoring/dataset/har70plus/505.csv',
    '/Users/dathril/Documents/HAR/patient monitoring/dataset/har70plus/506.csv',
    '/Users/dathril/Documents/HAR/patient monitoring/dataset/har70plus/507.csv',
    '/Users/dathril/Documents/HAR/patient monitoring/dataset/har70plus/508.csv',
    '/Users/dathril/Documents/HAR/patient monitoring/dataset/har70plus/509.csv'
]

# Read and concatenate
dataframes = [pd.read_csv(file) for file in files]
data = pd.concat(dataframes, ignore_index=True)
print("✅ Data loaded successfully.")
print("Shape:", data.shape)
print("Columns:", data.columns)

# Drop missing rows
data = data.dropna()

# Drop non-numeric columns
non_numeric_cols = data.select_dtypes(include=['object', 'datetime']).columns
print("Dropping non-numeric columns:", non_numeric_cols)
data = data.drop(non_numeric_cols, axis=1)

# Define target
target_column = data.columns[-1]
X = data.drop(target_column, axis=1).values
y = data[target_column].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print("🌲 Random Forest RMSE:", rf_rmse)

# Try LSTM safely
try:
    X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    model = Sequential()
    model.add(LSTM(64, input_shape=(1, X_train.shape[1])))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)
    y_pred_lstm = model.predict(X_test_lstm)
    lstm_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lstm))
    print("🧠 LSTM RMSE:", lstm_rmse)

except Exception as e:
    print("⚠️ LSTM Training Failed:", e)
    y_pred_lstm = np.zeros_like(y_pred_rf)
    lstm_rmse = None

# Combine predictions
y_pred_combined = (y_pred_rf + y_pred_lstm.reshape(-1)) / 2

# Save predictions
output_df = pd.DataFrame({
    'Actual': y_test,
    'RF_Predicted': y_pred_rf,
    'LSTM_Predicted': y_pred_lstm.reshape(-1),
    'Combined_Predicted': y_pred_combined
})

# Make sure directory exists
output_path = "/Users/dathril/Documents/HAR/patient monitoring/dataset/har70plus/combined_predictions.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
output_df.to_csv(output_path, index=False)
print(f"✅ Combined predictions saved successfully to: {output_path}")

# Few sample predictions
print("\n🔍 Sample Predictions:")
for i in range(10):
    print(f"RF: {y_pred_rf[i]:.2f} | LSTM: {y_pred_lstm[i][0]:.2f} | Actual: {y_test[i]:.2f}")
