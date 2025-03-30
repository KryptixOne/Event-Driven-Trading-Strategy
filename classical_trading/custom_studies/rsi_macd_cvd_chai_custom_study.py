from ..indicators.custom_indicators import  early_reversal_signals
from ..indicators.base_indicators_tos import macd_indicator, sma_indicator, cumulative_delta_volume,rsi_indicator,chaikin_oscillator, multi_day_floor_ceiling,vwap_indicator
import pandas as pd
def exit_signals(df):
    """
    Determine exit signals (ExitLong/ExitShort) based on prior Buy/Sell signals and early reversals.

    Example:
        Input:
            df -> pandas DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
        Output:
            (exitLong, exitShort)
            Each is a boolean pandas Series of length == len(df).

    """
    buySignal, sellSignal = buy_sell_signals(df)
    earlyBullishFlip, earlyBearishFlip = early_reversal_signals(df)

    # Track the last signal state (1=buy, -1=sell)
    lastSignal = [0]
    for i in range(1, len(df)):
        if buySignal[i]:
            lastSignal.append(1)
        elif sellSignal[i]:
            lastSignal.append(-1)
        else:
            lastSignal.append(lastSignal[-1])
    lastSignal = pd.Series(lastSignal, index=df.index)

    exitLong = (lastSignal == 1) & earlyBearishFlip
    exitShort = (lastSignal == -1) & earlyBullishFlip
    return exitLong, exitShort


def buy_sell_signals(df, sma_len=20, rsi_len=14, macd_fast=12, macd_slow=26, macd_sig=9, chaikin_fast=3,
                     chaikin_slow=10):
    """
    Generate composite Buy/Sell signals based on multiple indicators:
    SMA, RSI, MACD, Chaikin Oscillator, and Cumulative Delta Volume.

    Example:
        Input:
            df -> pandas DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
        Output:
            (buySignal, sellSignal)
            Each is a boolean pandas Series of length == len(df).

    """
    sma = sma_indicator(df, sma_len)
    print(f'sma is type{type(sma)}')
    rsi = rsi_indicator(df, rsi_len)
    print(f'rsi is type{type(rsi)}')
    macd_val, macd_avg = macd_indicator(df, macd_fast, macd_slow, macd_sig)
    print(f'macd_val and macd_avg is type{type(macd_val)} type{type(macd_avg)}')
    chaikin_val = chaikin_oscillator(df, chaikin_fast, chaikin_slow)
    print(f'chaikin_val is type{type(chaikin_val)}')
    cumdelta = cumulative_delta_volume(df)
    print(f'cumdelta is type{type(cumdelta)}')
    priceAboveSMA = df['Close'] > sma
    priceBelowSMA = df['Close'] < sma
    rsiBullish = rsi > 50
    rsiBearish = rsi < 50
    macdBullish = macd_val > macd_avg
    macdBearish = macd_val < macd_avg
    chaikinBullish = chaikin_val > 0
    chaikinBearish = chaikin_val < 0
    cdvBullish = cumdelta > cumdelta.shift(1)
    cdvBearish = cumdelta < cumdelta.shift(1)

    buySignal = priceAboveSMA & macdBullish & rsiBullish & chaikinBullish & cdvBullish
    sellSignal = priceBelowSMA & macdBearish & rsiBearish & chaikinBearish & cdvBearish
    return buySignal, sellSignal



def combined_study_signals(df):
    """
    This function demonstrates how to combine all indicators and produce
    the final signals and lines similar to the Thinkorswim study. It:
      1) Computes each indicator (SMA, RSI, MACD, Chaikin, CDV, VWAP)
      2) Creates Buy/Sell signals
      3) Creates Early Reversal signals
      4) Creates Multi-day floor/ceiling lines
      5) Creates Exit signals

    Returns:
        The original DataFrame with extra columns for signals and indicators.
    """

    df['SMA_20'] = sma_indicator(df, length=20)
    df['RSI_14'] = rsi_indicator(df, length=14)
    df['MACD'], df['MACD_Signal'] = macd_indicator(df, fast=12, slow=26, signal=9)
    df['Chaikin'] = chaikin_oscillator(df, fast=3, slow=10)
    df['CumDelta'] = cumulative_delta_volume(df)
    df['VWAP'] = vwap_indicator(df)  # new VWAP column

    # --- Buy/Sell signals
    buySignal, sellSignal = buy_sell_signals(
        df,
        sma_len=20, rsi_len=14,
        macd_fast=12, macd_slow=26, macd_sig=9,
        chaikin_fast=3, chaikin_slow=10
    )
    df['Buy'] = buySignal
    df['Sell'] = sellSignal

    # --- Early Reversal signals
    earlyBullishFlip, earlyBearishFlip = early_reversal_signals(
        df, sma_len=20, macd_fast=12, macd_slow=26, macd_sig=9
    )
    df['EarlyBullFlip'] = earlyBullishFlip
    df['EarlyBearFlip'] = earlyBearishFlip

    # --- Multi-day Floor/Ceiling
    floorLine, ceilLine = multi_day_floor_ceiling(df, lookbackDays=5)
    df['Floor'] = floorLine
    df['Ceiling'] = ceilLine

    # --- Exit signals
    exitLong, exitShort = exit_signals(df)
    df['ExitLong'] = exitLong
    df['ExitShort'] = exitShort

    return df


