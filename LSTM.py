import yfinance as yf
import math
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime



# Stock Data uses Yahoo Finance
def fetch_stock_data(ticker, start_date, end_date):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        return stock_data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# Get Apple Inc. stock data
ticker = 'AAPL'
start_date = '2018-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')
apple_data = fetch_stock_data(ticker, start_date, end_date)


# LSTM
data = apple_data.filter(['Close'])

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

train_data_len = math.ceil(len(data) * .8)
train_data = scaled_data[0:train_data_len, :]

x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Prepare the test dataset
test_data = scaled_data[train_data_len - 60:, :]
x_test = []
y_test = data.iloc[train_data_len:, :].values

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

valid = apple_data[train_data_len:][['Close']].copy()
valid['Predictions'] = predictions

plt.figure(figsize=(10, 5))
plt.title('Apple Stock Price Prediction using LSTM')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.plot(apple_data['Close'][:train_data_len], label='Training Closing Price', color='blue')
plt.plot(apple_data['Close'][train_data_len:], label='Actual Closing Price', color='red')
plt.plot(valid['Predictions'], label='Predicted Closing Price', color='orange')
plt.legend(loc='lower right')
plt.show()

