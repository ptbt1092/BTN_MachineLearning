import sys
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Thêm thư mục mẹ vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import get_historical_data, add_features

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerModel(nn.Module):
    def __init__(self, input_size, embed_dim, num_heads, num_layers, dropout=0.2):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim)
        self.transformer = nn.Transformer(
            d_model=embed_dim, 
            nhead=num_heads, 
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout
        )
        self.fc = nn.Linear(embed_dim, 1)
    
    def forward(self, src):
        src = self.embedding(src)
        src = self.positional_encoding(src)
        src = src.permute(1, 0, 2)  # Transformer expects input as (seq_len, batch_size, embed_dim)
        out = self.transformer(src, src)
        out = out.permute(1, 0, 2)  # Convert back to (batch_size, seq_len, embed_dim)
        return self.fc(out[:, -1, :])  # Use the output of the last time step

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

    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:i+seq_length]
            y = data[i+seq_length, 0]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    seq_length = 60
    x_train, y_train = create_sequences(train_data, seq_length)
    x_test, y_test = create_sequences(test_data, seq_length)

    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    embed_dim = 64
    num_heads = 8
    if embed_dim % num_heads != 0:
        raise ValueError("embed_dim must be divisible by num_heads")

    model = TransformerModel(input_size=x_train.shape[2], embed_dim=embed_dim, num_heads=num_heads, num_layers=2, dropout=0.2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(10):
        for batch in train_loader:
            x_batch, y_batch = batch
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred.squeeze(), y_batch)
            loss.backward()
            optimizer.step()

    model_name = f'{coin_id}_{"_".join(features)}.pth'
    model_dir = os.path.join('model', 'Transformer')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_name)
    torch.save(model.state_dict(), model_path)

    model.eval()
    with torch.no_grad():
        predictions = model(x_test).squeeze().numpy()

    predictions = scaler.inverse_transform(np.concatenate((predictions.reshape(-1, 1), np.zeros((predictions.shape[0], x_test.shape[2]-1))), axis=1))[:, 0]

    train = data[:training_data_len]
    valid = data[training_data_len+seq_length:]  # Adjust the start to match the length of predictions
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

