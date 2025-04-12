from strategy_inference import StrategyInference
from classical_trading.custom_studies.rsi_macd_cvd_chai_custom_study import buy_sell_signals
from run_optimization import build_df
from strat_optimizer import StrategyOptimizer

import matplotlib.pyplot as plt

def plot_pnl_with_markers(equity_curve, trades_df):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Left axis: Equity Curve
    ax1.plot(equity_curve.index, equity_curve, label="Equity Curve", color='blue')
    ax1.set_ylabel("Equity ($)")
    ax1.set_xlabel("Time")
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Right axis: PnL
    ax2 = ax1.twinx()
    ax2.step(trades_df['exit_time'], trades_df['cumulative_pnl'], where='post',
             label="Cumulative PnL", color='orange')
    ax2.scatter(trades_df['exit_time'], trades_df['cumulative_pnl'], color='red', zorder=5, label='PnL Point')
    ax2.set_ylabel("Cumulative PnL ($)")
    ax2.legend(loc='upper right')

    plt.title("Equity Curve vs. Cumulative PnL")
    plt.tight_layout()
    plt.show()

from matplotlib import pyplot as plt
if __name__ == "__main__":
    df = build_df(symbol="TSLA", time_interval="60m", time_period="730d")
    """Interval	Max Lookback Range
        "1m"	~7 days
        "2m"	~60 days
        "5m"	~60 days
        "15m"	~60 days
        "30m"	~60 days
        "60m"	~730 days (2y)
        "90m"	~60 days
        "1d"	Full history (years)
        "1wk"	Full history
        "1mo"	Full history

    """

    # Create a dummy instance just to grab the method
    dummy = StrategyOptimizer(df)

    best_params = {'sma_short_len': 80, 'sma_long_len': 120, 'rsi_len': 56, 'macd_fast': 8, 'macd_slow': 34, 'macd_sig': 26, 'chaikin_fast': 25, 'chaikin_slow': 46, 'trailing_stop_pct': 0.08326718442424871}
    #score was: 5.383531563496543
    strat = 'run_strategy_trailing_stop'
    sig_fcn = buy_sell_signals
    inference = StrategyInference(
        df=df,
        best_params=best_params,
        signal_func=sig_fcn,
        strategy_func=dummy.run_strategy_trailing_stop,
        unit_size=100,
        initial_cash=100000,
        position_mode='both'  # or 'long_only', 'short_only'
    )

    equity_curve = inference.run()
    print(equity_curve)
    # trades_df has columns ['exit_time','pnl'] with each row a closed trade
    trades_df = trades.sort_values(by='exit_time').reset_index(drop=True)

    # cumulative PnL
    trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()

    # convert to a time-indexed Series
    pnl_series = (
        trades_df.set_index('exit_time')['cumulative_pnl']
        .asfreq('S', method='pad')  # or reindex to match your equity times, or use a step plot directly
    )

    plot_pnl_with_markers(equity_curve, trades_df)



    print()

