import os
from itertools import combinations
from predict.lstm import predict_and_save_with_features
from predict.rnn import predict_and_save_with_features as rnn_predict
from predict.xgboost import predict_and_save_with_features as xgboost_predict
from predict.transformer import predict_and_save_with_features as transformer_predict

# Tạo các thư mục lưu kết quả dự đoán
os.makedirs('./output/LSTM', exist_ok=True)
os.makedirs('./output/RNN', exist_ok=True)
os.makedirs('./output/XGBoost', exist_ok=True)
os.makedirs('./output/Transformer', exist_ok=True)

# Các đặc trưng nâng cao có thể chọn
advanced_features = ['RSI', 'Bollinger Bands', 'Moving Average', 'Support/Resistance']

# Tạo danh sách các tổ hợp của advanced_features từ 0 đến 4 đặc trưng
features_list = []
for i in range(len(advanced_features) + 1):
    for combo in combinations(advanced_features, i):
        # Luôn thêm "Close" và "ROC" và sắp xếp các đặc trưng theo thứ tự bảng chữ cái
        features = ['Close', 'ROC'] + list(combo)
        features_list.append(sorted(features))

# Danh sách các đồng tiền
coins = ['bitcoin', 'ethereum', 'cardano']

# Lưu kết quả dự đoán cho từng đồng tiền và từng bộ đặc trưng
for coin in coins:
    for features in features_list:
        file_suffix = '_'.join([f.lower().replace('/', '') for f in features])
        # LSTM
        file_path = f'./output/LSTM/{coin}_{file_suffix}.csv'
        predict_and_save_with_features(coin, 'usd', features, file_path)
        
        # RNN
        file_path = f'./output/RNN/{coin}_{file_suffix}.csv'
        rnn_predict(coin, 'usd', features, file_path)
        
        # XGBoost
        file_path = f'./output/XGBoost/{coin}_{file_suffix}.csv'
        xgboost_predict(coin, 'usd', features, file_path)
        
        # Transformer and Time Embeddings
        file_path = f'./output/Transformer/{coin}_{file_suffix}.csv'
        transformer_predict(coin, 'usd', features, file_path)
