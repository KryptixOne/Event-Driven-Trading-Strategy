import time
import datetime
import yfinance as yf
import pandas as pd
from optimization.strategy_inference import StrategyInference
from classical_trading.custom_studies.rsi_macd_cvd_chai_custom_study import buy_sell_signals
from optimization.strat_optimizer import StrategyOptimizer
from backtesting.plotting.plot_fcn import plot_combined_dashboard
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output

import io
import contextlib


pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
# Example best_params:
best_params = {
    'sma_short_len': 62,
    'sma_long_len': 164,
    'rsi_len': 62,
    'macd_fast': 8,
    'macd_slow': 59,
    'macd_sig': 22,
    'chaikin_fast': 20,
    'chaikin_slow': 9,
    'trailing_stop_pct': 0.041965480796475316
}

global_df = None
global_trades = None
global_equity_curve = None
global_live_signals = None

def build_df_live(symbol="TSLA", interval="60m", lookback_days="169d"):
    print(f"Fetching {symbol} at {interval} for {lookback_days} ...")
    df = yf.download(
        tickers=symbol,
        period=lookback_days,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    df.columns = df.columns.get_level_values(0)
    df['Buy'], df['Sell'] = buy_sell_signals(df)
    return df

def run_hourly(symbol="TSLA"):
    """
    Called by the Dash callback to refresh data & strategy outputs.
    Updates global variables so the Dash chart can use them.
    """
    global global_df, global_trades, global_equity_curve, global_live_signals

    df = build_df_live(symbol, interval="60m", lookback_days="730d")
    dummy = StrategyOptimizer(df)

    inference = StrategyInference(
        df=df,
        best_params=best_params,
        signal_func=buy_sell_signals,
        strategy_func=dummy.run_strategy_with_live_signal_output,
        unit_size=100,
        initial_cash=100000,
        position_mode='both'
    )

    equity_curve, trades, live_signals = inference.run()

    global_df = df
    global_trades = trades
    global_equity_curve = equity_curve
    global_live_signals = live_signals

    print(f"Strategy updated at {datetime.datetime.now()} - final equity: {equity_curve.iloc[-1]:.2f}")
    if not trades.empty:
        print("Recent trade:", trades.iloc[-1])
    if not live_signals.empty:
        print("Live signals:", live_signals)


# ============ 3) Create a Dash app for a single-tab browser ============

app = dash.Dash(__name__)

# 1) Add a <pre> to your layout for logs:
app.layout = html.Div([
    html.Button("Refresh Now", id="manual-refresh-btn", n_clicks=0),
    dcc.Graph(id="dashboard-graph"),
    html.Div(id="log-output"),
    dcc.Interval(id="update-interval", interval=5*60*1000, n_intervals=0)
])

# 2) Helper to capture prints:
@contextlib.contextmanager
def capture_stdout():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf
@app.callback(
    [Output("dashboard-graph", "figure"),
     Output("log-output", "children")],
    [Input("update-interval", "n_intervals"),
     Input("manual-refresh-btn", "n_clicks")]
)
def update_figure(n_intervals, n_clicks):
    # refresh data & strategy
    run_hourly("TSLA")

    # 1) Summary line
    summary = html.Div(
        f"Strategy updated at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} – "
        f"final equity: {global_equity_curve.iloc[-1]:.2f}"
    )

    # 2) Recent trade
    if not global_trades.empty:
        recent = global_trades.iloc[[-1]].reset_index(drop=True)
        recent_div = dash_table.DataTable(
            columns=[{"name": c, "id": c} for c in recent.columns],
            data=recent.to_dict("records"),
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left"}
        )
    else:
        recent_div = html.Div("No trades yet.")

    # 3) Live signals
    if not global_live_signals.empty:
        recent_signals = global_live_signals.sort_values(by="timestamp", ascending=False)

        signals_div = dash_table.DataTable(
            columns=[{"name": c, "id": c} for c in recent_signals.columns],
            data=recent_signals.to_dict("records"),
            page_size=10,
            style_table={
                "overflowX": "auto",
                "maxHeight": "300px",
                "overflowY": "auto"
            },
            style_cell={"textAlign": "left", "whiteSpace": "nowrap"}
        )
    else:
        signals_div = html.Div("No live signals.")

    # 4) Plot
    fig = plot_combined_dashboard(
        df=global_df,
        trades_df=global_trades,
        equity_curve=global_equity_curve,
        live_signals_df=global_live_signals
    )

    return fig, [
        summary,
        html.H4("Recent Trade"),
        recent_div,
        html.H4("Live Signals"),
        signals_div
    ]
if __name__ == "__main__":
    # Run Dash server in a single tab/window
    app.run(debug=True)