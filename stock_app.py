import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
from dash.dependencies import Input, Output, State
from utils import append_real_time_data_and_predict, train_model_once, get_historical_data, fetch_real_time_data
import asyncio
import logging
import concurrent.futures
import os
import pytz
from datetime import datetime
from threading import Lock

# Khởi tạo đối tượng Lock
csv_lock = Lock()

# Tạo ra một app (web server)
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP, "https://use.fontawesome.com/releases/v5.10.0/css/all.css"])
app.title = "Crypto Prediction"
server = app.server

# Định dạng CSS 
CSS1 = {
    "margin-right": "18rem"
}
CSS2 = {
    "padding": "1.5rem 1rem",
    "margin-left": "18rem",  
    "margin-right": "2rem",
}
CSS3 = {
    "background-color": "#7fa934",
    "width": "16rem",
    "height": "50%",    
    "position": "fixed",
    "top": "65px",
    "right": 0,
    "padding": "2rem 1rem",
}

# CSS cho các tab
tabs_styles = {
    'height': '65px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold',
    'backgroundColor': '#f9f9f9',
    'color': 'black',
    'display': 'flex',
    'justify-content': 'center',
    'align-items': 'center',
    'font-size': '20px'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '2px solid #d6d6d6',
    'backgroundColor': '#7fa934',
    'color': 'white',
    'padding': '6px',
    'display': 'flex',
    'justify-content': 'center',
    'align-items': 'center',
    'font-size': '20px'
}

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
    dcc.Tabs(id="tabs-example", value='tab-1', children=[
        dcc.Tab(label='Stock Prediction', value='tab-1', style=tab_style, selected_style=tab_selected_style, children=[
            html.Div([
                html.Div([
                    html.H2("MENU",  style={"textAlign": "center"}),
                    html.Hr(),
                    dbc.Nav(method + feature, vertical=True),
                ],
                    style=CSS3,
                    id="sidebar",
                ),
                html.Div(id="page-content", style=CSS2),
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

                    dcc.Loading(
                        id="loading-compare",
                        type="default",
                        children=[dcc.Graph(id='compare')]
                    ),
                    html.Div([
                        dcc.Loading(
                            id="loading-next-prediction",
                            type="default",
                            children=[dcc.Graph(id='next-prediction', style={"position": "relative"})]
                        ),
                        html.Button('Next Timeframe', id='next-timeframe-button', n_clicks=0, style={
                            "position": "absolute",
                            "top": "870px",
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
            style=CSS1)
        ]),
        dcc.Tab(label='Real-time Data', value='tab-2', style=tab_style, selected_style=tab_selected_style, children=[
            html.Div([
                html.H1("Real-time Prediction",
                        style={"textAlign": "center", "color": "#7fa934", "margin-top": "30px"}),
                html.Br(),
                html.Div([
                    dcc.Loading(
                        id="loading-real-time-graph",
                        type="default",
                        children=[dcc.Graph(id='real-time-graph')]
                    ),
                    # dcc.Interval(
                    #     id='interval-component',
                    #     interval=1*60*1000,  # Cập nhật mỗi phút
                    #     n_intervals=0
                    # )
                ], className="container"),
            ],
            style=CSS1)
        ])
    ], style=tabs_styles)
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

        # Nhóm dữ liệu theo ngày và lọc những nhóm có đủ 24 dòng
        grouped = df.groupby(df.index.date)
        filtered_groups = [group for name, group in grouped if len(group) == 24]

        # Kiểm tra nếu có nhóm đủ 24 dòng
        if len(filtered_groups) > n_clicks:
            df_segment = filtered_groups[n_clicks]

            # Tạo tickvals và ticktext để định dạng trục x
            tickvals = df_segment.index[::4] 
            ticktext = [tickvals[0].strftime('%b %d, %Y, %H:%M')]
            ticktext.extend([t.strftime('%H:%M') for t in tickvals[1:]])

            traces.append(
                go.Scatter(x=df_segment.index, y=df_segment["Predictions"], mode='markers+lines', name=f'Predicted price of {dropdown[pair]}'))

    figure = {'data': traces,
              'layout': go.Layout(colorway=["#f61111", '#00ff51', '#f0ff00',
                                            '#8900ff', '#00d2ff', '#ff7400'],
                                  height=600,
                                  xaxis={
                                      "title": "Date",
                                      'tickvals': tickvals,
                                      'ticktext': ticktext,
                                  },
                                  yaxis={"title": "Price (USD)"})}
    return figure

def clean_csv_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

@app.callback(
    Output('real-time-graph', 'figure'),
    [Input('tabs-example', 'value'), 
    # Input('interval-component', 'n_intervals')
    ]
)
def update_real_time_graph(selected_tab):
    if selected_tab == 'tab-2':
        # Start fetching real-time data in a separate thread
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        executor.submit(start_event_loop)

        # Kiểm tra và khởi tạo tệp nếu không tồn tại
        if not os.path.exists('real_time_data.csv'):
            open('real_time_data.csv', 'w').close()

        # Đảm bảo đồng bộ khi đọc file CSV
        with csv_lock:
            try:
                data = pd.read_csv('real_time_data.csv', parse_dates=['timestamp'], index_col='timestamp')

                # Chuyển đổi index thành DatetimeIndex nếu chưa phải
                data.index = pd.to_datetime(data.index, errors='coerce', format='%Y-%m-%d %H:%M:%S')
                data = data.dropna()  # Bỏ các hàng có giá trị thời gian không hợp lệ

                # Chuyển đổi múi giờ sang GMT+7
                if data.index.tz is None:
                    data.index = data.index.tz_localize('UTC').tz_convert('Asia/Bangkok')
                else:
                    data.index = data.index.tz_convert('Asia/Bangkok')

                # Append real-time data and make predictions
                num_predictions = 10
                predictions = asyncio.run(append_real_time_data_and_predict('btcusdt', num_predictions))
                
                # Lọc dữ liệu để chỉ hiển thị giá của ngày hiện tại
                now = datetime.now(pytz.timezone('Asia/Bangkok'))
                today = now.date()
                data = data[data.index.date == today]

                trace = go.Scatter(x=data.index, y=data['close'], mode='markers+lines', name='Real-time BTC-USD')

                start_time = data.index[-1] + pd.Timedelta(minutes=1)
                prediction_times = pd.date_range(start=start_time, periods=len(predictions), freq='1min')
                trace_next_predictions = go.Scatter(x=prediction_times,
                                                   y=predictions,
                                                   mode='markers+lines',
                                                   marker=dict(color='red', size=8),
                                                   name='Next Predicted Prices')

                figure = {'data': [trace, trace_next_predictions],
                          'layout': go.Layout(colorway=["#00d2ff"],
                                              height=600,
                                              xaxis={"title": "Date", "tickformat": "%H:%M"},
                                              yaxis={"title": "Price (USD)"})}
                return figure
            except pd.errors.EmptyDataError:
                # Nếu tệp trống, không cập nhật đồ thị
                return go.Figure()

    return go.Figure()

def start_event_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    logging.info("Starting event loop")
    loop.run_until_complete(fetch_real_time_data('btcusdt'))

if __name__ == '__main__':
    # Ensure the model is trained once at the start
    train_model_once('bitcoin', 'usd', ['ROC'])
    
    # Check and ensure historical data is added only once
    if not os.path.exists('real_time_data.csv') or os.stat('real_time_data.csv').st_size == 0:
        historical_data = get_historical_data('bitcoin', 'usd')
        historical_data.rename(columns={"Datetime": "timestamp"}, inplace=True)
        historical_data.to_csv('real_time_data.csv', index=False)

    app.run_server(debug=True, port=3000)
