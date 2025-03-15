# CPALL Stock Price Prediction (2024)

This project uses machine learning to predict CPALL stock prices (in the Thai Stock Exchange) for 2024 using an LSTM (Long Short-Term Memory) model.

## Prerequisites
- Python 3.x
- Required libraries:
  - `yfinance`
  - `pandas`
  - `matplotlib`
  - `tensorflow`
  - `mplfinance`
  - `scikit-learn`

## Steps

### 1. Fetch CPALL Stock Data
We fetch CPALL stock data from Yahoo Finance for the year 2023:
```python
import yfinance as yf
import pandas as pd

stock_symbol = "CPALL.BK"  # For the Thai Stock Exchange (SET)
df = yf.download(stock_symbol, start="2023-01-01", end="2023-12-31")
df.head()

# Check Missing Data
print("Missing Values:\n", df.isnull().sum())
