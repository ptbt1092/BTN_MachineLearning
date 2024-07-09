import pandas as pd
import requests
import ta 
from tenacity import retry, stop_after_attempt, wait_fixed

@retry(stop=stop_after_attempt(5), wait=wait_fixed(60)) 
def get_historical_data(coin_id, vs_currency, days, n_roc):
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart'
    params = {
        'vs_currency': vs_currency,
        'days': days
    }
    response = requests.get(url, params=params)
    response.raise_for_status() 
    data = response.json()

    prices = data['prices']
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Tính toán ROC
    df['ROC'] = df['price'].pct_change(periods=n_roc) * 100
    
    return df

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
