import yfinance as yf


def get_symbol_data(symbol, interval='15m', start=None, end=None, period='60d'):
    """
    Retrieves historical data for a given symbol using yfinance.

    Parameters:
        symbol (str): The stock symbol to download (e.g., "AAPL").
        interval (str): The time interval between data points (default "15m").
        start (str): Start date string (e.g., "2022-01-01"). Optional if 'period' is given.
        end (str): End date string (e.g., "2022-06-01"). Optional if 'period' is given.
        period (str): yfinance period (e.g. '60d') if 'start'/'end' are not provided.

    Returns:
        DataFrame: A pandas DataFrame with columns ['Open','High','Low','Close','Volume'].
    """
    df = yf.download(symbol, interval=interval, start=start, end=end, period=period, progress=False)
    # Keep only essential columns and drop any rows with missing values
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    return df
