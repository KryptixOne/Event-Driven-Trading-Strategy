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


def buy_sell_signals(df, sma_short_len=20, sma_long_len =50, rsi_len=14, macd_fast=12, macd_slow=26, macd_sig=9, chaikin_fast=3,
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
    sma_short = sma_indicator(df, sma_short_len)
    sma_long = sma_indicator(df, sma_long_len)
    rsi = rsi_indicator(df, rsi_len)
    macd_val, macd_avg = macd_indicator(df, macd_fast, macd_slow, macd_sig)
    chaikin_val = chaikin_oscillator(df, chaikin_fast, chaikin_slow)
    cumdelta = cumulative_delta_volume(df)

    # Define the warm-up period based on the largest lookback
    warmup = max(sma_short_len, sma_long_len, rsi_len, macd_slow, macd_sig, chaikin_slow)

    # Trim the dataframe to only include rows after warm-up
    df = df.iloc[warmup:].copy()
    sma_short = sma_short.iloc[warmup:]
    sma_long = sma_long.iloc[warmup:]
    rsi = rsi.iloc[warmup:]
    macd_val = macd_val.iloc[warmup:]
    macd_avg = macd_avg.iloc[warmup:]
    chaikin_val = chaikin_val.iloc[warmup:]
    cumdelta = cumdelta.iloc[warmup:]

    priceAboveSMAshort = df['Close'] > sma_short
    priceBelowSMAshort = df['Close'] < sma_short
    priceAboveSMAlong = df['Close'] > sma_long
    priceBelowSMAlong = df['Close'] < sma_long

    rsiBullish = rsi > 50
    rsiBearish = rsi < 50
    macdBullish = macd_val > macd_avg
    macdBearish = macd_val < macd_avg
    chaikinBullish = chaikin_val > 0
    chaikinBearish = chaikin_val < 0
    cdvBullish = cumdelta > cumdelta.shift(1)
    cdvBearish = cumdelta < cumdelta.shift(1)

    buySignal = priceAboveSMAshort & priceAboveSMAlong & macdBullish & rsiBullish & chaikinBullish & cdvBullish
    sellSignal = priceBelowSMAshort & priceBelowSMAlong & macdBearish & rsiBearish & chaikinBearish & cdvBearish

    buySignal = buySignal.astype(bool)
    sellSignal = sellSignal.astype(bool)

    # 1) detect new signals
    buy_entry_raw = buySignal & ~buySignal.shift(1).fillna(False)
    sell_entry_raw = sellSignal & ~sellSignal.shift(1).fillna(False)

    # 2) shift for next-bar
    buy_entry = buy_entry_raw.shift(1).fillna(False).astype(bool)
    sell_entry = sell_entry_raw.shift(1).fillna(False).astype(bool)

    return buy_entry, sell_entry

