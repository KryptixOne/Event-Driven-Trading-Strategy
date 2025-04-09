from symbol_data import get_symbol_data
from classical_trading.indicators.base_indicators_tos import multi_day_floor_ceiling, rsi_indicator, macd_indicator, \
    chaikin_oscillator, cumulative_delta_volume, daily_vwap_with_bands
from classical_trading.custom_studies.rsi_macd_cvd_chai_custom_study import exit_signals, buy_sell_signals
from classical_trading.indicators.custom_indicators import early_reversal_signals
from plotting.plot_fcn import plot_symbol_and_signals, plot_interactive, plot_interactive_with_indicators


def main():
    # 1) Download data (15-minute bars, last 30 days) for a symbol
    print('Getting Data for TSLA')
    df = get_symbol_data("TSLA", interval="15m", period="30d")
    df.columns = df.columns.get_level_values(0)  # ['Open', 'High', ...]


    df['Buy'], df['Sell'] = buy_sell_signals(df)
    df['EarlyBullFlip'], df['EarlyBearFlip'] = early_reversal_signals(df)
    floorLine, ceilingLine = multi_day_floor_ceiling(df, lookbackDays=5)
    df['Floor'] = floorLine
    df['Ceiling'] = ceilingLine
    df['ExitLong'], df['ExitShort'] = exit_signals(df)
    df['RSI_14'] = rsi_indicator(df)  # or whichever you've named
    df['MACD'], df['MACD_Signal'] = macd_indicator(df)
    df['Chaikin'] = chaikin_oscillator(df)
    df['CumDelta'] = cumulative_delta_volume(df)


    # 3) Compute daily-reset VWAP & optional bands
    df = daily_vwap_with_bands(df, dev_factor=2.0)

    # 4) Plot price with signals, plus optional indicator charts
    #plot_symbol_and_signals(df, show_indicators=False, show_vwap_bands=False, market_hours_only=True)
    plot_interactive_with_indicators(df)

if __name__ == "__main__":


    main()
