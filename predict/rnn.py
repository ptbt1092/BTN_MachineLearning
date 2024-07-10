import sys
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Dropout

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

    train_data = scaled_data[0:training_data_len, :]
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))

    model = Sequential()
    model.add(SimpleRNN(50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(SimpleRNN(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=32, epochs=10)

    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []
    y_test = dataset[training_data_len:, 0]

    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], x_train.shape[2]-1))), axis=1))[:, 0]

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