import os
import websockets
import json
import asyncio
import pandas as pd
import yfinance as yf
import ta
import logging
from tenacity import retry, stop_after_attempt, wait_fixed
from datetime import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Thiết lập logging
logging.basicConfig(level=logging.INFO)

def add_features(df, features):
    if 'ROC' in features:
        df['ROC'] = df['close'].pct_change(periods=14) * 100
    if 'RSI' in features:
        df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    if 'Bollinger Bands' in features:
        bb_indicator = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['BB_high'] = bb_indicator.bollinger_hband()
        df['BB_low'] = bb_indicator.bollinger_lband()
    if 'Moving Average' in features:
        df['MA'] = df['close'].rolling(window=20).mean()
    if 'Support/Resistance' in features:
        df['Support'] = df['close'].rolling(window=20).min()
        df['Resistance'] = df['close'].rolling(window=20).max()
    
    df.dropna(inplace=True)
    return df

@retry(stop=stop_after_attempt(5), wait=wait_fixed(60))
def get_yf_ticker(coin_id, vs_currency):
    coin_map = {
        'bitcoin': 'BTC',
        'ethereum': 'ETH',
        'cardano': 'ADA'
    }
    currency_map = {
        'usd': 'USD'
    }
    return f"{coin_map[coin_id.lower()]}-{currency_map[vs_currency.lower()]}"

def get_historical_data(coin_id, vs_currency):
    ticker = get_yf_ticker(coin_id, vs_currency)
    data = yf.download(ticker, period='3mo', interval='1h')
    data.rename(columns={"Adj Close": "close"}, inplace=True)
    data = data[['close']]
    data.reset_index(inplace=True)
    return data

if not os.path.exists('real_time_data.csv'):
    historical_data = get_historical_data('bitcoin', 'usd')
    historical_data.to_csv('historical_data.csv', index=False)
    historical_data.to_csv('real_time_data.csv', index=False)
else:
    logging.info("File real_time_data.csv đã tồn tại. Chỉ thêm dữ liệu thời gian thực mới.")

def train_model(coin_id, vs_currency, features):
    df = get_historical_data(coin_id, vs_currency)
    df = add_features(df, features)
    df.dropna(inplace=True)

    data = df.filter(['close'] + features)
    dataset = data.values
    training_data_len = int(np.ceil(len(dataset) * .8))

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0:training_data_len, :]
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=32, epochs=10)

    return model, scaler, training_data_len, dataset

# Huấn luyện mô hình
model, scaler, training_data_len, dataset = train_model('bitcoin', 'usd', ['ROC'])

async def fetch_real_time_data(symbol, interval='1m'):
    url = f'wss://stream.binance.com:9443/ws/{symbol}@kline_{interval}'
    async with websockets.connect(url) as websocket:
        logging.info(f"Connected to {url}")
        while True:
            data = await websocket.recv()
            data_json = json.loads(data)
            kline = data_json['k']
            if kline['x']:
                new_data = {
                    'timestamp': pd.to_datetime(kline['t'], unit='ms'),
                    'close': float(kline['c'])
                }
                yield new_data

async def append_real_time_data_and_predict(symbol):
    data = pd.read_csv('real_time_data.csv', parse_dates=['timestamp'], index_col='timestamp')

    async for new_data in fetch_real_time_data(symbol):
        new_row = pd.DataFrame([new_data])
        new_row.set_index('timestamp', inplace=True)

        data = pd.concat([data, new_row])
        data.to_csv('real_time_data.csv', mode='a', header=False, index=True)

        # Chuẩn bị dữ liệu để dự đoán giá nến kế tiếp
        scaled_data = scaler.transform(data)
        look_back = 60
        last_60_candles = scaled_data[-look_back:]
        X_test = np.reshape(last_60_candles, (1, look_back, scaled_data.shape[1]))

        # Dự đoán giá nến kế tiếp
        predicted_price_scaled = model.predict(X_test)
        predicted_price = scaler.inverse_transform(
            np.concatenate((predicted_price_scaled, np.zeros((predicted_price_scaled.shape[0], scaled_data.shape[1] - 1))), axis=1)
        )[:, 0]

        logging.info(f"Appended new data: {new_data}")
        logging.info(f"Giá dự đoán nến kế tiếp: {predicted_price[0]}")

def start_event_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    logging.info("Starting event loop")
    loop.run_until_complete(append_real_time_data_and_predict('btcusdt'))

import concurrent.futures
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
executor.submit(start_event_loop)

try:
    while True:
        pass
except KeyboardInterrupt:
    logging.info("Program interrupted, exiting...")
    executor.shutdown(wait=False)
