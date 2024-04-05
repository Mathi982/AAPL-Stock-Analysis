import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from datetime import datetime

ticker_symbol = "AAPL"
stock = yf.Ticker(ticker_symbol)
start_date = '2018-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')
data = stock.history(start=start_date, end=end_date)

n = 60
for i in range(1, n + 1):
    data[f'lag_{i}'] = data['Close'].shift(i)

# Drops any NaN values
data = data.dropna()

X = data[[f'lag_{i}' for i in range(1, n + 1)]]
y = data['Close']

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]



model = GradientBoostingRegressor(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

plt.figure(figsize=(10, 5))
plt.plot(data.index[train_size:], y_test, label='Actual Price', color='blue')
plt.plot(data.index[train_size:], model.predict(X_test), label='Predicted Price', color='orange')
plt.title('Stock Price Prediction using Gradient Boosting')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

num_days_to_predict = 14
future_predictions = []

last_n_days = X.tail(1)

for _ in range(num_days_to_predict):
    next_day_prediction = model.predict(last_n_days)[0]

    new_row = np.append(last_n_days.iloc[0, 1:], next_day_prediction).reshape(1, -1)
    last_n_days = pd.DataFrame(new_row, columns=last_n_days.columns)

    future_predictions.append(next_day_prediction)

future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=num_days_to_predict)

future_predictions = []

last_n_days = X.tail(1)

for _ in range(num_days_to_predict):
    next_day_prediction = model.predict(last_n_days)[0]

    new_row = np.append(last_n_days.iloc[0, 1:], next_day_prediction).reshape(1, -1)
    last_n_days = pd.DataFrame(new_row, columns=last_n_days.columns)

    future_predictions.append(next_day_prediction)

plt.figure(figsize=(10, 5))
plt.plot(data.index, data['Close'], label='Historical Price', color='blue')
plt.plot(future_dates, future_predictions, label='Future Predictions', color='red', linestyle='--')
plt.title('Future Stock Price Prediction using Gradient Boosting')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

print("Future Predictions:")
for date, prediction in zip(future_dates, future_predictions):
    print(f"{date.date()}: ${prediction:.2f}")

