import sys
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb

# Thêm thư mục mẹ vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import get_historical_data, add_features

def predict_and_save_with_features(coin_id, vs_currency, features, file_path):
    # Kiểm tra và tạo thư mục nếu cần
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Lấy dữ liệu lịch sử
    df = get_historical_data(coin_id, vs_currency)
    
    # Thêm các đặc trưng
    df = add_features(df, features)
    df.dropna(inplace=True)  # Bỏ các giá trị NaN
    
    # Sử dụng các cột được chọn làm đặc trưng
    data = df.filter(['price'] + features)
    dataset = data.values
    training_data_len = math.ceil(len(dataset) * .8)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[:training_data_len, :]
    test_data = scaled_data[training_data_len:, :]
    
    x_train, y_train = train_data[:, 1:], train_data[:, 0]
    x_test, y_test = test_data[:, 1:], test_data[:, 0]

    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(x_train, y_train)
    
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(np.concatenate((predictions.reshape(-1, 1), np.zeros((predictions.shape[0], x_train.shape[1]))), axis=1))[:, 0]

    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions

    valid['Date'] = valid.index
    results = valid[['Date', 'price', 'Predictions']]
    results.to_csv(file_path, index=False)

    rmse = np.sqrt(mean_squared_error(valid['price'], valid['Predictions']))
    mae = mean_absolute_error(valid['price'], valid['Predictions'])
    mape = np.mean(np.abs((valid['price'] - valid['Predictions']) / valid['price'])) * 100

    print(f'{coin_id.upper()}-{vs_currency.upper()} RMSE: {rmse}')
    print(f'{coin_id.upper()}-{vs_currency.upper()} MAE: {mae}')
    print(f'{coin_id.upper()}-{vs_currency.upper()} MAPE: {mape}%')
