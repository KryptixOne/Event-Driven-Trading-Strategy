import multiprocessing
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

def add_weekly_return_label(df, days_ahead=5):
    """
    Adds a new column 'NextWeekReturn' to the DataFrame, which calculates
    the % price change from day t to day t+days_ahead.

    Then creates a binary label 'Label' = 1 if NextWeekReturn > 0, else 0.
    """
    df = df.copy().sort_values('Date')
    df['NextWeekReturn'] = (
            df['Close Price'].shift(-days_ahead) / df['Close Price'] - 1
    )
    df['Label'] = (df['NextWeekReturn'] > 0).astype(int)

    return df


def create_weekly_labels_and_split(df):
    # 1) Add 5-day forward return label
    df = add_weekly_return_label(df, days_ahead=5)

    # 2) Drop rows at the end that can't have NextWeekReturn
    df.dropna(subset=['NextWeekReturn', 'Label'], inplace=True)

    # 3) Split by date (example boundaries)
    df_train, df_val, df_test = split_data_by_date(df,
                                                   train_end='2021-12-31',
                                                   val_end='2022-12-31')
    return df_train, df_val, df_test


def split_data_by_explicit_dates(df):
    """
    Splits the DataFrame into train/val/test sets by explicit date cutoffs.

    Train: 2021-12-01 to 2023-12-31
    Validation: 2024-01-01 to 2024-06-30
    Test: 2024-07-01 to 2025-02-19

    Returns: df_train, df_val, df_test
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # Define explicit date splits
    train_end = "2023-12-31"
    val_start = "2024-01-01"
    val_end = "2024-06-30"
    test_start = "2024-07-01"

    # Apply splits
    df_train = df[df['Date'] <= train_end]
    df_val = df[(df['Date'] >= val_start) & (df['Date'] <= val_end)]
    df_test = df[df['Date'] >= test_start]

    return df_train, df_val, df_test

def split_data_by_date(df,
                       train_end='2021-12-31',
                       val_end='2022-12-31'):
    """
    Splits the DataFrame into train/val/test sets by Date.
    Returns: df_train, df_val, df_test
    """
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    df_train = df[df['Date'] <= train_end]
    df_val = df[(df['Date'] > train_end) & (df['Date'] <= val_end)]
    df_test = df[df['Date'] > val_end]

    return df_train, df_val, df_test





def scale_ticker_data(ticker_data, feature_cols, lookback_days):
    """
    Scales data for a single ticker using a rolling window.

    Parameters:
    - ticker_data: DataFrame containing a single ticker's data.
    - feature_cols: List of feature columns to scale.
    - lookback_days: Number of past days to use for scaling.

    Returns:
    - Scaled ticker DataFrame
    """
    ticker_data = ticker_data.sort_values("Date")  # Ensure time ordering

    # Store results
    scaled_values = np.zeros_like(ticker_data[feature_cols].values)

    for i in range(len(ticker_data)):
        # Get past lookback window
        start_idx = max(0, i - lookback_days)
        past_data = ticker_data.iloc[start_idx:i]

        if len(past_data) > 10:  # Ensure enough data
            scaler = StandardScaler()
            scaler.fit(past_data[feature_cols])  # Fit on past `lookback_days`
            scaled_values[i] = scaler.transform(ticker_data.iloc[[i]][feature_cols])
        else:
            scaled_values[i] = np.nan  # If not enough data, mark as NaN

    ticker_data.loc[:, feature_cols] = scaled_values  # Update scaled values
    return ticker_data.dropna()  # Remove NaN rows

def scale_features_parallel(df_train, df_val, df_test, feature_cols, lookback_days=252, num_workers=4):
    """
    Scales features for each ticker separately using a rolling lookback window with parallel processing.

    - Training, validation, and test sets are all scaled using **only past data**.
    - Prevents data leakage while keeping scaling **realistic for trading**.
    - Uses multiprocessing for faster execution.

    Returns: Scaled df_train, df_val, df_test
    """
    df_train_scaled = df_train.copy()
    df_val_scaled = df_val.copy()
    df_test_scaled = df_test.copy()

    tickers = df_train['Ticker'].unique()

    # Use multiprocessing to scale multiple tickers in parallel
    with multiprocessing.Pool(num_workers) as pool:
        train_results = pool.starmap(scale_ticker_data, [(df_train[df_train['Ticker'] == t], feature_cols, lookback_days) for t in tickers])
        val_results = pool.starmap(scale_ticker_data, [(df_val[df_val['Ticker'] == t], feature_cols, lookback_days) for t in tickers])
        test_results = pool.starmap(scale_ticker_data, [(df_test[df_test['Ticker'] == t], feature_cols, lookback_days) for t in tickers])

    # Combine results
    df_train_scaled = pd.concat(train_results)
    df_val_scaled = pd.concat(val_results)
    df_test_scaled = pd.concat(test_results)

    return df_train_scaled, df_val_scaled, df_test_scaled



def scale_features_per_ticker_with_lookback_complete(df_train, df_val, df_test, feature_cols, lookback_days=252):
    """
    Scales features for each ticker separately using a rolling lookback window.

    - Training, validation, and test sets are all scaled using **only past data**.
    - Prevents data leakage while keeping scaling **realistic for trading**.

    Returns: Scaled df_train, df_val, df_test
    """
    df_train_scaled = df_train.copy()
    df_val_scaled = df_val.copy()
    df_test_scaled = df_test.copy()

    scalers = {}  # Store scalers per ticker

    for ticker in df_train['Ticker'].unique():
        # Select only this ticker's data
        ticker_train = df_train[df_train['Ticker'] == ticker]
        ticker_val = df_val[df_val['Ticker'] == ticker]
        ticker_test = df_test[df_test['Ticker'] == ticker]

        train_indices = ticker_train.index.tolist()
        val_indices = ticker_val.index.tolist()
        test_indices = ticker_test.index.tolist()

        train_scaled_values = np.zeros_like(ticker_train[feature_cols].values)
        val_scaled_values = np.zeros_like(ticker_val[feature_cols].values)
        test_scaled_values = np.zeros_like(ticker_test[feature_cols].values)

        # ✅ Apply Rolling Lookback for Training Data
        for i, train_idx in enumerate(train_indices):
            past_data = ticker_train[ticker_train.index <= train_idx]  # Get all past data up to this point

            if len(past_data) > lookback_days:
                past_data = past_data.iloc[-lookback_days:]  # Keep only last `lookback_days`

            if len(past_data) > 10:  # Ensure enough data
                temp_scaler = StandardScaler()
                temp_scaler.fit(past_data[feature_cols])  # Fit on rolling past data
                train_scaled_values[i] = temp_scaler.transform(ticker_train.loc[[train_idx], feature_cols])
            else:
                train_scaled_values[i] = np.nan  # If not enough data, assign NaN

        df_train_scaled.loc[df_train['Ticker'] == ticker, feature_cols] = train_scaled_values

        # ✅ Apply Rolling Lookback for Validation Data
        for i, val_idx in enumerate(val_indices):
            past_data = df_train[df_train['Ticker'] == ticker].copy()
            past_data = past_data[past_data.index <= val_idx]  # Keep only past training data

            if len(past_data) > lookback_days:
                past_data = past_data.iloc[-lookback_days:]  # Use only last `lookback_days`

            if len(past_data) > 10:
                temp_scaler = StandardScaler()
                temp_scaler.fit(past_data[feature_cols])  # Fit on rolling past data
                val_scaled_values[i] = temp_scaler.transform(ticker_val.loc[[val_idx], feature_cols])
            else:
                val_scaled_values[i] = np.nan

        df_val_scaled.loc[df_val['Ticker'] == ticker, feature_cols] = val_scaled_values

        # ✅ Apply Rolling Lookback for Test Data
        for i, test_idx in enumerate(test_indices):
            past_data = df_train[df_train['Ticker'] == ticker].copy()
            past_data = pd.concat([past_data, df_val[df_val['Ticker'] == ticker]])  # ✅ Correct
            # Include validation as history
            past_data = past_data[past_data.index <= test_idx]  # Keep only past data

            if len(past_data) > lookback_days:
                past_data = past_data.iloc[-lookback_days:]  # Use only last `lookback_days` days

            if len(past_data) > 10:
                temp_scaler = StandardScaler()
                temp_scaler.fit(past_data[feature_cols])  # Fit on rolling past data
                test_scaled_values[i] = temp_scaler.transform(ticker_test.loc[[test_idx], feature_cols])
            else:
                test_scaled_values[i] = np.nan

        df_test_scaled.loc[df_test['Ticker'] == ticker, feature_cols] = test_scaled_values

        # Drop NaN values after scaling (to ensure valid inputs for training)
        df_train_scaled = df_train_scaled.dropna(subset=feature_cols)
        df_val_scaled = df_val_scaled.dropna(subset=feature_cols)
        df_test_scaled = df_test_scaled.dropna(subset=feature_cols)

    return df_train_scaled, df_val_scaled, df_test_scaled


def scale_features_per_ticker_with_lookback(df_train, df_val, df_test, feature_cols, lookback_days=252):
    """
    Scales features for each ticker separately using only past data up to each test point.

    - Trains one StandardScaler per ticker based on the training set.
    - Applies a rolling window for test data, ensuring no future data is used for scaling.

    Returns: Scaled df_train, df_val, df_test
    """
    df_train_scaled = df_train.copy()
    df_val_scaled = df_val.copy()
    df_test_scaled = df_test.copy()

    scalers = {}  # Store scalers per ticker

    for ticker in df_train['Ticker'].unique():
        # Select only this ticker's data
        ticker_train = df_train[df_train['Ticker'] == ticker]
        ticker_val = df_val[df_val['Ticker'] == ticker]
        ticker_test = df_test[df_test['Ticker'] == ticker]

        # Fit the scaler ONLY on the training set
        scaler = StandardScaler()
        scaler.fit(ticker_train[feature_cols])  # Fit on past data only
        scalers[ticker] = scaler

        # Apply to training & validation sets normally
        df_train_scaled.loc[df_train['Ticker'] == ticker, feature_cols] = scaler.transform(ticker_train[feature_cols])
        df_val_scaled.loc[df_val['Ticker'] == ticker, feature_cols] = scaler.transform(ticker_val[feature_cols])

        # Rolling Scaling for the Test Set to Avoid Leakage
        test_indices = ticker_test.index.tolist()  # Get row indices
        test_scaled_values = np.zeros_like(ticker_test[feature_cols].values)  # Placeholder for scaled test data

        for i, test_idx in enumerate(test_indices):
            # Get historical window (at most `lookback_days` past days)
            past_data = df_train[df_train['Ticker'] == ticker].copy()
            past_data = past_data.append(df_val[df_val['Ticker'] == ticker])  # Add validation data
            past_data = past_data[past_data.index <= test_idx]  # Keep only past data

            if len(past_data) > lookback_days:
                past_data = past_data.iloc[-lookback_days:]  # Use only recent `lookback_days` days

            # If we have enough past data, fit a new scaler
            if len(past_data) > 10:  # Ensure enough samples
                temp_scaler = StandardScaler()
                temp_scaler.fit(past_data[feature_cols])  # Fit scaler on rolling past data
                test_scaled_values[i] = temp_scaler.transform(ticker_test.loc[[test_idx], feature_cols])
            else:
                test_scaled_values[i] = np.nan  # Handle cases with insufficient data

        # Apply the scaled values to the test set
        df_test_scaled.loc[df_test['Ticker'] == ticker, feature_cols] = test_scaled_values

    return df_train_scaled, df_val_scaled, df_test_scaled


def scale_features_per_ticker(df_train, df_val, df_test, feature_cols):
    """
    Applies StandardScaler to each ticker separately in train, val, and test sets.

    Parameters:
    - df_train, df_val, df_test: DataFrames for train, validation, test
    - feature_cols: List of feature columns to scale

    Returns:
    - Scaled train, val, test DataFrames
    """
    df_train_scaled = df_train.copy()
    df_val_scaled = df_val.copy()
    df_test_scaled = df_test.copy()

    # Scale per Ticker
    scalers = {}  # Store scalers for each ticker

    for ticker in df_train['Ticker'].unique():
        # Select only the ticker's data
        ticker_train = df_train[df_train['Ticker'] == ticker]
        ticker_val = df_val[df_val['Ticker'] == ticker]
        ticker_test = df_test[df_test['Ticker'] == ticker]

        # Fit only on training data for this ticker
        scaler = StandardScaler()
        scaler.fit(ticker_train[feature_cols])

        # Save the scaler (optional, if needed later)
        scalers[ticker] = scaler

        # Transform each dataset separately
        df_train_scaled.loc[df_train['Ticker'] == ticker, feature_cols] = scaler.transform(ticker_train[feature_cols])
        df_val_scaled.loc[df_val['Ticker'] == ticker, feature_cols] = scaler.transform(ticker_val[feature_cols])
        df_test_scaled.loc[df_test['Ticker'] == ticker, feature_cols] = scaler.transform(ticker_test[feature_cols])

    return df_train_scaled, df_val_scaled, df_test_scaled, scalers  # Return scalers if needed


def scale_features(df_train, df_val, df_test, feature_cols):
    """
    Scales the feature_cols using StandardScaler from sklearn.
    Applies the scaler (fit on train) to val & test.
    Returns: df_train, df_val, df_test with scaled features.
    """
    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols])

    df_train[feature_cols] = scaler.transform(df_train[feature_cols])
    df_val[feature_cols] = scaler.transform(df_val[feature_cols])
    df_test[feature_cols] = scaler.transform(df_test[feature_cols])

    return df_train, df_val, df_test, scaler

