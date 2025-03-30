from .base_indicators_tos import macd_indicator, sma_indicator

import pandas as pd


def early_reversal_signals(df, sma_len=20, macd_fast=12, macd_slow=26, macd_sig=9):
    """
    Identify early bullish/bearish reversals based on SMA and MACD crossover logic.

    Example:
        Input:
            df -> pandas DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
        Output:
            (earlyBullishFlip, earlyBearishFlip)
            Each is a boolean pandas Series indicating an early reversal flip.

    """
    sma = sma_indicator(df, sma_len)
    macd_val, macd_avg = macd_indicator(df, macd_fast, macd_slow, macd_sig)

    # Potential reversals
    potBearish = (df['Close'] < sma) | (macd_val < macd_avg)
    potBullish = (df['Close'] > sma) | (macd_val > macd_avg)

    # Reversal state replication
    reversalState = [0]
    for i in range(1, len(df)):
        if potBullish[i] and not potBearish[i]:
            reversalState.append(1)
        elif potBearish[i] and not potBullish[i]:
            reversalState.append(-1)
        else:
            reversalState.append(reversalState[-1])
    reversalState = pd.Series(reversalState, index=df.index)

    prevState = reversalState.shift(1)
    earlyBullishFlip = (reversalState == 1) & (prevState == -1)
    earlyBearishFlip = (reversalState == -1) & (prevState == 1)
    return earlyBullishFlip, earlyBearishFlip
