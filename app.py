import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
import pandas as pd
from dash.dependencies import Input, Output, State

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

        # Tạo file path từ các đặc trưng được chọn
        file_suffix = '_'.join([f.lower().replace('/', '') for f in checklist_value])
        file_path = f'./output/{radio_items_value}/{coin_name}_{file_suffix}.csv'
        
        # Đọc dữ liệu từ file
        df = pd.read_csv(file_path)
        df.head()
        df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d %H:%M:%S")
        df.index = df['Date']

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
                                         'rangeselector': {'buttons': list([{'count': 1, 'label': '1M',
                                                                             'step': 'month',
                                                                             'stepmode': 'backward'},
                                                                            {'count': 6, 'label': '6M',
                                                                             'step': 'month',
                                                                             'stepmode': 'backward'},
                                                                            {'step': 'all'}])},
                                         'rangeslider': {'visible': True}, 'type': 'date'},
                                  yaxis={"title": "Price (USD)"})}
    return figure


if __name__ == '__main__':
    app.run_server(debug=True, port=3000)
