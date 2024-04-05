import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime

ticker_symbol = "AAPL"
stock = yf.Ticker(ticker_symbol)
start_date = '2018-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')
data = stock.history(start=start_date, end=end_date)

n = 60  # Number of days to consider for prediction
for i in range(1, n+1):
    data[f'lag_{i}'] = data['Close'].shift(i)

# Drops any NaN values
data = data.dropna()

X = data[[f'lag_{i}' for i in range(1, n+1)]]
y = data['Close']

# Splits data into training and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

plt.figure(figsize=(10, 5))
plt.plot(data.index[train_size:], y_test, label='Actual Price', color='blue')
plt.plot(data.index[train_size:], y_pred, label='Predicted Price', color='orange')
plt.title('Stock Price Prediction using Random Forest')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

last_n_days = data[['Close']].tail(n).T
future_prediction = model.predict(last_n_days)
print(f"Future Prediction: {future_prediction[0]}")
