from strat_optimizer import StrategyOptimizer
from backtesting.symbol_data import get_symbol_data
from classical_trading.custom_studies.rsi_macd_cvd_chai_custom_study import exit_signals, buy_sell_signals
import matplotlib.pyplot as plt
import numpy as np
def build_df(symbol: str = "TSLA", time_interval: str = "5m", time_period: str = "30d"):
    print(f'Building dataframe for {symbol} at time interval {time_interval} over the period of {time_period}')
    df = get_symbol_data(symbol, interval=time_interval, period=time_period)
    df.columns = df.columns.get_level_values(0)  # ['Open', 'High', ...]
    df['Buy'], df['Sell'] = buy_sell_signals(df)
    return df

def sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=1640):
    excess = returns - risk_free_rate
    std = excess.std()
    if std == 0 or np.isnan(std):
        return -np.inf  # or 0.0 if you prefer neutral instead of penalizing
    return (excess.mean() / std) * np.sqrt(periods_per_year)



def infer_frequency_and_sharpe(equity_curve, risk_free_rate=0.0):
    index = equity_curve.index
    if len(index) < 2:
        return np.nan

    # Infer average interval in seconds
    delta_seconds = np.median(np.diff(equity_curve.index).astype("timedelta64[s]").astype(float))
    periods_per_year = int((365.25 * 24 * 60 * 60) / delta_seconds)

    returns = equity_curve.pct_change().dropna()
    print("Start date:", equity_curve.index[0])
    print("End date:", equity_curve.index[-1])

    return sharpe_ratio(returns, risk_free_rate, periods_per_year)



def combined_cost_function(equity_curve, weight_sharpe=0.7, weight_return=0.3):
    """
    Cost = weighted combination of Sharpe Ratio and Total Return (Profit)
    """
    print("Interval (secs):", (equity_curve.index[1] - equity_curve.index[0]).total_seconds())
    print("Start:", equity_curve.index[0])
    print("Next:", equity_curve.index[1])

    sharpe = infer_frequency_and_sharpe(equity_curve, risk_free_rate=0.0)

    total_return = equity_curve.iloc[-1] - equity_curve.iloc[0]
    #print(total_return)
    #print(sharpe)
    # Normalize scales if needed
    score = (weight_sharpe * sharpe) +(weight_return * (total_return / equity_curve.iloc[0]))
    print(score)
    return score



def plot_strategy_vs_hold(df, equity_curve, initial_cash=100_000):
    """
    Compare optimized strategy performance vs. buy-and-hold.

    Parameters:
        df (pd.DataFrame): Must include a 'Close' column.
        equity_curve (pd.Series): From run_strategy().
        initial_cash (float): Starting capital.
    """
    # Buy and hold benchmark
    start_price = df['Close'].iloc[0]
    shares_held = initial_cash / start_price
    buy_hold_curve = shares_held * df['Close']

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve.index, equity_curve, label='Strategy Equity Curve')
    plt.plot(equity_curve.index, buy_hold_curve.loc[equity_curve.index], label='Buy & Hold', linestyle='--')

    plt.title("Strategy vs Buy & Hold")
    plt.ylabel("Portfolio Value ($)")
    plt.xlabel("Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


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

    optimizer = StrategyOptimizer(df, n_trials=100)

    # 2. Set your signal and cost functions
    optimizer.set_signal_function(buy_sell_signals)


    optimizer.set_cost_function(lambda curve: combined_cost_function(curve,1, 0))


    # 3. Run optimization
    param_bounds = {
        'sma_short_len': ("int", 10, 90, 10),
        'sma_long_len': ("int", 20, 200, 20),
        'rsi_len': ("int", 10, 90, 2),
        'macd_fast': ("int", 8, 32, 2),
        'macd_slow': ("int", 20, 64, 2),
        'macd_sig': ("int", 6, 32, 2),
        'chaikin_fast': ("int", 2,64, 1),
        'chaikin_slow': ("int", 8, 64, 2),
        #'stop_loss_pct':   ("float", 0.01, 0.05),  # continuous range from 1% to 3%
        #'take_profit_pct': ("float", 0.04, 0.10),
        'trailing_stop_pct': ("float", 0.01, 0.15)
    }
    #optimizer.set_strategy_function(optimizer.run_strategy_hard_stop_profit_taking)

    optimizer.set_strategy_function(optimizer.run_strategy_trailing_stop)

    best_params, best_score, best_equity_curve,study = optimizer.optimize_with_optuna(param_bounds)

    print("Best Params:", best_params)
    print("Best Score:", best_score)
    print(f"best_equity_curve {best_equity_curve}")

    plot_strategy_vs_hold(df, best_equity_curve, initial_cash=100000)
    print()
