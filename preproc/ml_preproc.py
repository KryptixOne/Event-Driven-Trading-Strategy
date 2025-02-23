
import pandas as pd
from sklearn.preprocessing import StandardScaler


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

