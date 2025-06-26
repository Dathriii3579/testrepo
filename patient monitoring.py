import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load CSV files (change path to actual file paths on your machine)
files = [
    '/Users/dathril/Documents/HAR/patient monitoring/dataset/har70plus/503.csv',
    '/Users/dathril/Documents/HAR/patient monitoring/dataset/har70plus/505.csv',
    '/Users/dathril/Documents/HAR/patient monitoring/dataset/har70plus/506.csv',
    '/Users/dathril/Documents/HAR/patient monitoring/dataset/har70plus/507.csv',
    '/Users/dathril/Documents/HAR/patient monitoring/dataset/har70plus/508.csv',
    '/Users/dathril/Documents/HAR/patient monitoring/dataset/har70plus/509.csv'
]

# Read and concatenate all files
dataframes = [pd.read_csv(file) for file in files]
data = pd.concat(dataframes, ignore_index=True)

print("Data loaded successfully. Shape:", data.shape)
print("Columns:", data.columns)

# Drop rows with missing values
data = data.dropna()

# Drop non-numeric columns (like timestamp)
non_numeric_cols = data.select_dtypes(include=['object', 'datetime']).columns
print("Dropping non-numeric columns:", non_numeric_cols)
data = data.drop(non_numeric_cols, axis=1)

# Define target column again after dropping
target_column = data.columns[-1]

# Split features and target
X = data.drop(target_column, axis=1).values
y = data[target_column].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Random Forest Model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))

# LSTM Model (reshaping for LSTM input)
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
print("LSTM RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lstm)))

# Sample Predictions
print("\nSample Predictions:")
for i in range(10):
    print(f"RF: {y_pred_rf[i]:.2f} | LSTM: {y_pred_lstm[i][0]:.2f} | Actual: {y_test[i]:.2f}")
