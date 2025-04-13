import time
import datetime
import yfinance as yf
import pandas as pd
from optimization.strategy_inference import StrategyInference
from classical_trading.custom_studies.rsi_macd_cvd_chai_custom_study import buy_sell_signals
from optimization.strat_optimizer import StrategyOptimizer
from backtesting.plotting.plot_fcn import plot_combined_dashboard


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


def build_df_live(symbol="TSLA", interval="60m", lookback_days="169d"):
    """
    Download the last 'lookback_days' of data at 'interval' from Yahoo Finance.
    """
    print(f"Fetching {symbol} at {interval} for {lookback_days} ...")
    df = yf.download(
        tickers=symbol,
        period=lookback_days,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    # Ensure the DataFrame has the needed columns in correct shape
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    df.columns = df.columns.get_level_values(0)
    df['Buy'], df['Sell'] = buy_sell_signals(df)
    return df


def run_hourly(symbol="TSLA"):
    """
    This function:
      1) Fetches data from Yahoo
      2) Runs the strategy inference
      3) Logs or alerts if there's a new Buy or Sell signal
    """
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

    # Check the last row of trades to see if a new trade triggered
    if not trades.empty:
        last_trade = trades.iloc[-1]
        print("Most recent trade:")
        print(last_trade)
        # You could add logic: if last_trade['exit_time'] == last bar => alert
    else:
        print("No trades so far.")

    if not live_signals.empty:
        print(f"Most recent signal was {live_signals.iloc[-1]}")

    # Optionally, detect if a new buy or sell happened on the final bar
    # (this is simpler if we record the final posState too, but for now we rely on trades)

    # Basic logging:
    print(f"Equity final: {equity_curve.iloc[-1]:.2f}")
    # Could do more advanced logic to e.g. email or Slack alert
    print(trades)
    plot_combined_dashboard(
        df=df, trades_df=trades, equity_curve=equity_curve, live_signals_df=live_signals)


def main_live_loop():
    """
    Main loop that runs once every hour (e.g. at minute 30).
    In real usage, you'd run this script continuously.
    """
    while True:
        now = datetime.datetime.now()
        # e.g. run at HH:35 (to ensure the 1-hour bar is closed)
        # or run at the top of hour if your data is posted quickly
        if now.minute == 35:
            print(f"Running strategy at {now}")
            run_hourly(symbol="TSLA")
            print("Sleeping until next hour...\n")
            time.sleep(60)  # to avoid repeated run in the same minute
        else:
            # Sleep for e.g. 30 seconds
            time.sleep(30)


if __name__ == "__main__":
    # Option 1: Just run once
    run_hourly("TSLA")

    # Option 2: Loop forever
    # main_live_loop()
