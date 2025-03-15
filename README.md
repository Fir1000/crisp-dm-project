# CPALL Stock Price Prediction (2024)

โปรเจกต์นี้ใช้การเรียนรู้ของเครื่อง (Machine Learning) เพื่อทำนายราคาหุ้น CPALL (ในตลาดหลักทรัพย์ไทย) ในปี 2024 โดยใช้โมเดล LSTM (Long Short-Term Memory)

## รายการสิ่งที่ต้องเตรียม
- Python 3.x
- ไลบรารีที่จำเป็น:
  - `yfinance`
  - `pandas`
  - `matplotlib`
  - `tensorflow`
  - `mplfinance`
  - `scikit-learn`

## ขั้นตอนการดำเนินการ

### 1. ดึงข้อมูลหุ้น CPALL
เราดึงข้อมูลราคาหุ้น CPALL จาก Yahoo Finance สำหรับปี 2023:
```python
import yfinance as yf
import pandas as pd

stock_symbol = "CPALL.BK"  # สำหรับตลาดหลักทรัพย์ไทย (SET)
df = yf.download(stock_symbol, start="2023-01-01", end="2023-12-31")
df.head()
