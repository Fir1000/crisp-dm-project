import yfinance as yf
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt

# Fetch CPALL stock data
stock_symbol = "CPALL.BK"  # For the Thai Stock Exchange (SET)
df = yf.download(stock_symbol, start="2023-01-01", end="2023-12-31")
df.head()
<img width="1440" alt="Screenshot 2568-03-15 at 15 06 55" src="https://github.com/user-attachments/assets/c30357e2-7864-446b-bce5-a4e28b05e7fe" />

# Check for missing data
print("Missing Values:\n", df.isnull().sum())

# Plot the closing price of CPALL stock
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["Close"], label="Closing Price", linewidth=2, color='blue')

plt.title("CPALL Stock Closing Price (2023)")
plt.xlabel("Date")
plt.ylabel("Price (THB)")
plt.legend()
plt.grid()

# Show the plot
plt.show()

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

data = df[["Close"]].values

# Scale the data to the range 0-1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Function to create time series dataset
def create_dataset(dataset, time_step=60):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

# Set a time step of 60 days for predicting the next day
time_step = 60
X, Y = create_dataset(scaled_data, time_step)

# Split into Train-Test (80-20)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# Reshape for LSTM model
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

# Compile the model
model.compile(optimizer="adam", loss="mean_squared_error")

# Train the model
model.fit(X_train, Y_train, batch_size=16, epochs=50, verbose=1)

# Predict stock prices for 2024
future_days = 365  # Number of days to predict
future_predictions = []

# Use the last 60 days as the starting point
last_60_days = scaled_data[-time_step:].reshape(1, time_step, 1)

for _ in range(future_days):
    pred = model.predict(last_60_days, verbose=0)
    pred = pred.reshape(1, 1, 1)  # Reshape to 3D
    future_predictions.append(pred[0, 0])
    last_60_days = np.append(last_60_days[:, 1:, :], pred, axis=1)

# Convert predictions back to real stock prices
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Convert last date to datetime
last_date = pd.to_datetime(df.index[-1])

# Create future dates for 2024
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq='D')

# Ensure df.index and future_dates are datetime objects
df.index = pd.to_datetime(df.index, errors="coerce")
future_dates = pd.to_datetime(future_dates, errors="coerce")

# Plot the predicted stock prices for 2024
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["Close"], label="Actual Price", color="blue")
plt.plot(future_dates, future_predictions, label="Predicted Price (2024)", color="red", linestyle="dashed")
plt.title("CPALL Stock Price Prediction for 2024")
plt.xlabel("Date")
plt.ylabel("Price (THB)")
plt.legend()
plt.grid()
plt.show()

# Fetch stock data for 2024
stock_symbol = "CPALL.BK"  # For the Thai Stock Exchange (SET)
df2 = yf.download(stock_symbol, start="2024-01-01", end="2024-12-31")
df2.head()

# Create a DataFrame for the predicted results
predicted_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Price": future_predictions.flatten()
})
predicted_df.head()

from sklearn.metrics import mean_squared_error, mean_absolute_error

# Predict test data
Y_pred = model.predict(X_test)

# Convert predictions back to actual stock prices
Y_test_inv = scaler.inverse_transform(Y_test.reshape(-1, 1))
Y_pred_inv = scaler.inverse_transform(Y_pred)

# Calculate MSE, RMSE, MAE
mse = mean_squared_error(Y_test_inv, Y_pred_inv)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y_test_inv, Y_pred_inv)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
