import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from tensorflow.keras.models import load_model
import numpy as np
import plotly.graph_objs as go
import pandas as pd
from dash.dependencies import Input, Output, State
from utils import get_historical_data
from utils import add_features
import torch

# Tạo ra một app (web server)
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP, "https://use.fontawesome.com/releases/v5.10.0/css/all.css"])
app.title = "Crypto Prediction"
server = app.server

# Định dạng CSS 
CSS1 = {
    "margin-right": "18rem"
}
CSS2 = {
    "padding": "2rem 1rem",
    "margin-left": "18rem",  
    "margin-right": "2rem",
}
CSS3 = {
    "background-color": "#7fa934",
    "width": "16rem",
    "height": "50%",    
    "position": "fixed",
    "top": "0",
    "right": 0,
    "padding": "2rem 1rem",
}

def load_model_and_predict(coin_id, features, model_path, scaler, n_clicks):
    model = load_model(model_path)
    # Tạo dữ liệu mẫu để dự đoán
    df = get_historical_data(coin_id, 'usd')
    df = add_features(df, features)
    df.dropna(inplace=True)

    data = df.filter(['price'] + features)
    dataset = data.values

    scaled_data = scaler.transform(dataset)
    seq_length = 60
    start_idx = n_clicks * 7
    end_idx = start_idx + 7
    sample_data = scaled_data[start_idx:end_idx, :]

    def create_sequences(data, seq_length):
        xs = []
        for i in range(len(data) - seq_length):
            x = data[i:i+seq_length]
            xs.append(x)
        return np.array(xs)

    x_sample = create_sequences(sample_data, seq_length)
    x_sample = torch.tensor(x_sample, dtype=torch.float32)

    predictions = model.predict(x_sample)
    predictions = scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], x_sample.shape[2]-1))), axis=1))[:, 0]

    return predictions


method = [
    html.Li(
        dbc.Row(
            [
                dbc.Col("Method", style={
                        "fontWeight": "bold", "fontSize": "1.25rem"}),
                dbc.Col(
                    html.I(className="fas fa-chevron-right mr-3"), width="auto"
                ),
            ],
            className="my-1",
        ),
        style={"cursor": "pointer"},
        id="method",
    ),
    dbc.Collapse(
        [
            dbc.Form([
                dbc.RadioItems(
                    options=[
                        {"label": "LSTM", "value": "LSTM"},
                        {"label": "XGBoost", "value": "XGBoost"},
                        {"label": "RNN", "value": "RNN"},
                        {"label": "Transformer", "value": "Transformer"},
                    ],
                    value="LSTM",
                    style={"color": "white"},
                    id="radio-items",
                ),
            ]),
        ],
        id="collapse_method",
    ),
]

feature = [
    html.Li(
        dbc.Row(
            [
                dbc.Col("Feature", style={
                        "fontWeight": "bold", "fontSize": "1.25rem"}),
                dbc.Col(
                    html.I(className="fas fa-chevron-right mr-3"), width="auto"
                ),
            ],
            className="my-1",
        ),
        style={"cursor": "pointer"},
        id="feature",
    ),
    dbc.Collapse(
        [
            dbc.Form([
                dbc.Checklist(
                    options=[
                        {"label": "Close", "value": "Close"},
                        {"label": "ROC", "value": "ROC"},
                        {"label": "RSI", "value": "RSI"},
                        {"label": "Bollinger Bands", "value": "Bollinger Bands"},
                        {"label": "Moving Average", "value": "Moving Average"},
                        {"label": "Support/Resistance", "value": "Support/Resistance"},
                    ],
                    value=["Close", "ROC"],
                    style={"color": "white"},
                    id="checklist-items",
                ),
            ]),
        ],
        id="collapse_feature",
    ),
]

app.layout = html.Div([
    # side bar
    html.Div([
        html.H2("MENU",  style={"textAlign": "center"}),
        html.Hr(),
        dbc.Nav(method + feature, vertical=True),
    ],
        style=CSS3,
        id="sidebar",
    ),

    html.Div(id="page-content", style=CSS2),

    html.Div([
        html.H1("Crypto Price Analysis Dashboard",
                style={"textAlign": "center", "color": "#7fa934"}),
        html.Br(),
        html.Div([
            html.H3(id="dash-title",
                    style={"textAlign": "center"}),

            dcc.Dropdown(id='my-dropdown',
                         options=[{'label': 'BTC-USD', 'value': 'BTC-USD'},
                                  {'label': 'ETH-USD', 'value': 'ETH-USD'},
                                  {'label': 'ADA-USD', 'value': 'ADA-USD'},
                                  ],
                         multi=True, value=['BTC-USD'],
                         style={"display": "block", "margin-left": "auto",
                                "margin-right": "0", "width": "60%"}),

            dcc.Graph(id='compare'),
            html.Div([
                dcc.Graph(id='next-prediction', style={"position": "relative"}),
                html.Button('Predict Next Period', id='next-timeframe-button', n_clicks=0, style={
                    "position": "absolute",
                    "top": "820px",
                    "left": "10px",
                    "background-color": "#7fa934",
                    "color": "white",
                    "border": "none",
                    "padding": "10px 20px",
                    "cursor": "pointer"
                })
            ], className="dash-graph"),

        ], className="container"),
    ],
        style=CSS1,
    ),
])

# this function is used to toggle the is_open property of each Collapse
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

# this function applies the "open" class to rotate the chevron
def set_navitem_class(is_open):
    if is_open:
        return "open"
    return ""

# callback method
app.callback(
    Output(f"collapse_method", "is_open"),
    [Input(f"method", "n_clicks")],
    [State(f"collapse_method", "is_open")],
)(toggle_collapse)

app.callback(
    Output(f"method", "className"),
    [Input(f"collapse_method", "is_open")],
)(set_navitem_class)

# callback feature
app.callback(
    Output(f"collapse_feature", "is_open"),
    [Input(f"feature", "n_clicks")],
    [State(f"collapse_feature", "is_open")],
)(toggle_collapse)

app.callback(
    Output(f"feature", "className"),
    [Input(f"collapse_feature", "is_open")],
)(set_navitem_class)

# Ensure "Close" and "ROC" are always selected
@app.callback(
    Output("checklist-items", "value"),
    [Input("checklist-items", "value")],
)
def ensure_mandatory_features(selected_features):
    mandatory_features = {"Close", "ROC"}
    return list(mandatory_features.union(set(selected_features)))

@app.callback(Output('compare', 'figure'), [
    Input('my-dropdown', 'value'),
    Input("radio-items", "value"),
    Input("checklist-items", "value"),
])
def update_graph(selected_dropdown, radio_items_value, checklist_value):
    dropdown = {"BTC-USD": "BTC-USD", "ETH-USD": "ETH-USD",
                 "ADA-USD": "ADA-USD"}
    coin_name_map = {"BTC": "bitcoin", "ETH": "ethereum", "ADA": "cardano"}
    trace_predict = []
    trace_original = []
    for pair in selected_dropdown:
        coin_id = pair.split('-')[0].upper()
        coin_name = coin_name_map.get(coin_id, coin_id.lower())

        features = sorted(set(checklist_value) | {"Close", "ROC"})

        # Tạo file path từ các đặc trưng được chọn
        file_suffix = '_'.join([f.lower().replace('/', '') for f in features])
        file_path = f'./output/{radio_items_value}/{coin_name}_{file_suffix}.csv'
        
        # Đọc dữ liệu từ file
        df = pd.read_csv(file_path)
        df.head()
        df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format=True, errors='coerce')
        df.set_index('Date', inplace=True)

        # Load figure
        trace_predict.append(
            go.Scatter(x=df.index,
                       y=df["Predictions"],
                       mode='lines',
                       name=f'Predicted price of {dropdown[pair]}', textposition='bottom center'))
        trace_original.append(
            go.Scatter(x=df.index,
                       y=df["price"],
                       mode='lines',
                       name=f'Original price of {dropdown[pair]}', textposition='bottom center'))
    traces = [trace_original, trace_predict]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#f61111", '#00ff51', '#f0ff00',
                                            '#8900ff', '#00d2ff', '#ff7400'],
                                  height=600,
                                  xaxis={"title": "Date",
                                         'rangeselector': {'buttons': list([
                                            {'count': 1, 'label': '1D', 'step': 'day', 'stepmode': 'backward'},
                                            {'count': 7, 'label': '1W', 'step': 'day', 'stepmode': 'backward'},
                                            {'count': 1, 'label': '1M', 'step': 'month', 'stepmode': 'backward'},
                                            {'step': 'all'}
                                         ])},
                                         'rangeslider': {'visible': True}, 'type': 'date'},
                                  yaxis={"title": "Price (USD)"})}
    return figure

@app.callback(Output('next-prediction', 'figure'), [
    Input('my-dropdown', 'value'),
    Input("radio-items", "value"),
    Input("checklist-items", "value"),
    Input('next-timeframe-button', 'n_clicks')
])
def update_next_prediction(selected_dropdown, radio_items_value, checklist_value, n_clicks):
    dropdown = {"BTC-USD": "BTC-USD", "ETH-USD": "ETH-USD", "ADA-USD": "ADA-USD"}
    coin_name_map = {"BTC": "bitcoin", "ETH": "ethereum", "ADA": "cardano"}
    traces = []
    
    for pair in selected_dropdown:
        coin_id = pair.split('-')[0].upper()
        coin_name = coin_name_map.get(coin_id, coin_id.lower())

        features = sorted(set(checklist_value) | {"Close", "ROC"})

        # Tạo file path từ các đặc trưng được chọn
        file_suffix = '_'.join([f.lower().replace('/', '') for f in features])
        file_path = f'./output/{radio_items_value}/{coin_name}_{file_suffix}.csv'
        
        # Đọc dữ liệu từ file
        df = pd.read_csv(file_path)
        df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format=True, errors='coerce')
        df.set_index('Date', inplace=True)

        # Chia dataframe thành các phần 7 ngày
        start_idx = n_clicks * 30
        end_idx = start_idx + 30
        df_segment = df.iloc[start_idx:end_idx]

        # Tạo timestamp tiếp theo
        if end_idx < len(df):
            next_timestamp = df.index[end_idx]
            next_candle_prediction = df["Predictions"].iloc[end_idx]
        else:
            next_timestamp = df.index[-1] + pd.Timedelta(days=1)
            next_candle_prediction = df["Predictions"].iloc[-1]

        # traces.append(
        #     go.Scatter(x=df_segment.index, y=df_segment["price"], mode='lines', name=f'Actual price of {dropdown[pair]}'))
        traces.append(
            go.Scatter(x=df_segment.index, y=df_segment["Predictions"], mode='markers+lines', name=f'Predicted price of {dropdown[pair]}'))
        # traces.append(
        #     go.Scatter(x=[next_timestamp], y=[next_candle_prediction], mode='markers+lines',
        #                name=f'Next predicted price of {dropdown[pair]}'))

    figure = {'data': traces,
              'layout': go.Layout(colorway=["#f61111", '#00ff51', '#f0ff00',
                                            '#8900ff', '#00d2ff', '#ff7400'],
                                  height=600,
                                  xaxis={"title": "Date"},
                                  yaxis={"title": "Price (USD)"})}
    return figure

if __name__ == '__main__':
    app.run_server(debug=True, port=3000)
