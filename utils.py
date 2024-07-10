import os
import websockets
import json
from datetime import date
import pandas as pd
import yfinance as yf
import ta 
from tenacity import retry, stop_after_attempt, wait_fixed

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
    # Ánh xạ coin_id và vs_currency sang ticker của yfinance
    ticker = get_yf_ticker(coin_id, vs_currency)

    # Lấy dữ liệu từ yfinance
    # start_date = '2018-01-01'
    # end_date = date.today()
    # data = yf.download(ticker, start=start_date, end=end_date)
    data = yf.download(ticker, period='3mo', interval='1h')

    # Đổi tên các cột
    data.rename(columns={"Adj Close": "price"}, inplace=True)
    
    # Chỉ giữ lại cột giá và thời gian
    data = data[["price"]]
    
    return data

def add_features(df, features):
    if 'ROC' in features:
        df['ROC'] = df['price'].pct_change(periods=14) * 100
    if 'RSI' in features:
        df['RSI'] = ta.momentum.RSIIndicator(df['price'], window=14).rsi()
    if 'Bollinger Bands' in features:
        bb_indicator = ta.volatility.BollingerBands(df['price'], window=20, window_dev=2)
        df['BB_high'] = bb_indicator.bollinger_hband()
        df['BB_low'] = bb_indicator.bollinger_lband()
    if 'Moving Average' in features:
        df['MA'] = df['price'].rolling(window=20).mean()
    if 'Support/Resistance' in features:
        df['Support'] = df['price'].rolling(window=20).min()
        df['Resistance'] = df['price'].rolling(window=20).max()
    
    df.dropna(inplace=True)
    return df

async def fetch_real_time_data(symbol, interval='1m'):
    url = f'wss://stream.binance.com:9443/ws/{symbol}@kline_{interval}'
    async with websockets.connect(url) as websocket:
        while True:
            data = await websocket.recv()
            data_json = json.loads(data)
            kline = data_json['k']
            if kline['x']:  # Check if the kline is closed
                yield {
                    'timestamp': pd.to_datetime(kline['t'], unit='ms'),
                    'open': float(kline['o']),
                    'high': float(kline['h']),
                    'low': float(kline['l']),
                    'close': float(kline['c']),
                    'volume': float(kline['v'])
                }

async def update_data(symbol, interval='1m', path='real_time_data.csv'):
    data_stream = fetch_real_time_data(symbol, interval)
    async for new_data in data_stream:
        df = pd.DataFrame([new_data])
        df.to_csv(path, mode='a', header=not os.path.exists(path), index=False)
        print(f"Appended new data: {new_data}")


get_historical_data('bitcoin','usd')