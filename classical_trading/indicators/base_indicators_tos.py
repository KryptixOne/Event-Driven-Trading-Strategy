import pandas as pd
import numpy as np


def sma_indicator(df, length=20):
    """
    Calculate the Simple Moving Average (SMA) of the 'Close' column.

    Example:
        Input:
            df -> pandas DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
            length -> 20
        Output:
            A pandas Series of SMA values of length == len(df).

    """
    return df['Close'].rolling(window=length).mean()


def rsi_indicator(df, length=14):
    """
    Calculate the Relative Strength Index (RSI).

    Example:
        Input:
            df -> pandas DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
            length -> 14
        Output:
            A pandas Series of RSI values, range typically [0..100].

    """
    diff = df['Close'].diff()
    gain = diff.clip(lower=0).abs()
    loss = (-diff).clip(lower=0).abs()
    avg_gain = gain.rolling(window=length).mean()
    avg_loss = loss.rolling(window=length).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def macd_indicator(df, fast=12, slow=26, signal=9):
    """
    Calculate MACD (Moving Average Convergence Divergence) and its signal line.

    Example:
        Input:
            df -> pandas DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
            fast  -> 12
            slow  -> 26
            signal -> 9
        Output:
            macd_value (pd.Series), macd_signal (pd.Series)
            Both are length == len(df).

    """
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    macd_value = ema_fast - ema_slow
    macd_signal = macd_value.ewm(span=signal, adjust=False).mean()
    return macd_value, macd_signal


def chaikin_oscillator(df, fast=3, slow=10):
    """
    Calculate the Chaikin Oscillator by manually computing the A/D line, then subtracting EMAs.

    Example:
        Input:
            df -> pandas DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
            fast -> 3
            slow -> 10
        Output:
            A pandas Series of Chaikin Oscillator values.

    """
    high_low = df['High'] - df['Low']
    mf_multiplier = np.where(
        high_low != 0,
        ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / high_low,
        0
    )
    # Money Flow Volume
    mf_volume = mf_multiplier * df['Volume']

    # A/D line (cumulative)
    ad_line = mf_volume.cumsum()

    # Chaikin Oscillator
    short_ema = ad_line.ewm(span=fast, adjust=False).mean()
    long_ema = ad_line.ewm(span=slow, adjust=False).mean()
    return short_ema - long_ema


def cumulative_delta_volume(df):
    """
    Calculate a custom Cumulative Delta Volume (CDV).

    Example:
        Input:
            df -> pandas DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
        Output:
            A pandas Series tracking the cumulative delta volume.

    Returns:
        Series of cumulative delta volume.
    """
    tw = df['High'] - np.maximum(df['Open'], df['Close'])
    bw = np.minimum(df['Open'], df['Close']) - df['Low']
    body = (df['Close'] - df['Open']).abs()

    denom = tw + bw + body
    bullish = df['Open'] <= df['Close']
    numerator = 0.5 * (tw + bw + 2 * body * bullish.astype(float))
    rate = numerator / denom.replace(0, np.nan)
    rate = rate.fillna(0.5)

    deltaup = df['Volume'] * rate
    deltadown = df['Volume'] * (1 - rate)
    delta = np.where(df['Close'] >= df['Open'], deltaup, -deltadown)

    return pd.Series(delta.ravel(), index=df.index).cumsum()


def vwap_indicator(df):
    """
    Volume-Weighted Average Price (VWAP).

    Example:
        Input:
            df -> pandas DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
        Output:
            A pandas Series of VWAP values for each row of df.

    Formula:
        Typical Price = (High + Low + Close) / 3
        VWAP = cumulative( TypicalPrice * Volume ) / cumulative( Volume )
    """
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    return vwap


def daily_vwap_with_bands(df, dev_factor=2.0):
    """
    Computes a daily-resetting VWAP plus optional upper/lower bands using standard deviation.

    Parameters:
        df (DataFrame): A DataFrame with columns ['Open','High','Low','Close','Volume']
                        and a DateTimeIndex.
        dev_factor (float): Standard deviation multiplier for upper/lower bands (default=2.0).

    Returns:
        DataFrame: A DataFrame containing columns:
            'VWAP' - daily-reset VWAP values,
            'VWAP_Upper' - VWAP + dev_factor * daily std,
            'VWAP_Lower' - VWAP - dev_factor * daily std.
    """
    # We'll need a separate grouping by each day's date:
    # Typical price for each bar
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3

    # For daily grouping, create a date column (without time)
    df['DateOnly'] = df.index.date

    # We'll group by DateOnly so each day gets its own cumulative sums
    # then reassign the per-day computations back to the main DataFrame
    vwap_vals = []
    vwap_std_vals = []

    for date_val, group in df.groupby('DateOnly'):
        # cumulative sums for that day
        tpv_cum = (group['Volume'] * ((group['High'] + group['Low'] + group['Close']) / 3)).cumsum()
        vol_cum = group['Volume'].cumsum()

        # daily vwap
        vwap_series = tpv_cum / vol_cum

        # standard deviation of typical price * volume weighting
        # we approximate by using the rolling std of typical_price
        # weighted by volume. A simpler approach is unweighted std.
        # For simplicity, we’ll do an unweighted rolling std of typical_price so far in the day:
        day_std = (group['High'] + group['Low'] + group['Close']) / 3
        day_std = day_std.expanding().std()  # expanding std from the start of the day

        vwap_vals.append(vwap_series)
        vwap_std_vals.append(day_std)

    # Combine the daily VWAP pieces
    vwap_vals = pd.concat(vwap_vals)
    vwap_std_vals = pd.concat(vwap_std_vals)

    # Insert final columns
    df['VWAP'] = vwap_vals
    # Upper and Lower bands
    df['VWAP_Upper'] = df['VWAP'] + dev_factor * vwap_std_vals
    df['VWAP_Lower'] = df['VWAP'] - dev_factor * vwap_std_vals

    # Remove helper column
    df.drop(columns=['DateOnly'], inplace=True)
    return df


def multi_day_floor_ceiling(df, lookbackDays=5):
    """
    Determine floor and ceiling lines over N-day highs and lows.

    Example:
        Input:
            df -> pandas DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
            lookbackDays -> 5
        Output:
            (floorLine, ceilLine)
            Each is a pandas Series representing the rolling min and max over 'lookbackDays'.

    """
    floorLine = df['Low'].rolling(window=lookbackDays).min()
    ceilLine = df['High'].rolling(window=lookbackDays).max()
    return floorLine, ceilLine
