import pandas as pd

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_colwidth', 20)


def clean_data(df):
    # Suppose df is your large dataset with multiple tickers
    # Ensure Date is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort by Ticker and Date
    df = df.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)

    # Example of forward filling missing data for Market Cap or volume
    df['Volume'] = df.groupby('Ticker')['Volume'].ffill()

    # Some missing mention counts might actually be 0 (if there is no mention)
    df['Mentions'] = df['Mentions'].fillna(0)
    df['Percent Mention Change'] = df['Percent Mention Change'].fillna(0)
    # Handling any missing price by forward/backward fill
    df['Open Price'] = df.groupby('Ticker')['Open Price'].ffill().bfill()
    df['Close Price'] = df.groupby('Ticker')['Close Price'].ffill().bfill()
    df = df[df['Ticker'].notna()]
    df = df[df['Close Price'].notna()]
    return df


def eda(df):
    # Create daily return for each ticker
    df['Return'] = df.groupby('Ticker')['Close Price'].pct_change()

    # Also compute daily mention change
    df['Mentions_Change'] = df.groupby('Ticker')['Mentions'].pct_change()

    # Drop rows that became NaN after these computations
    df = df.dropna(subset=['Return', 'Mentions_Change'])

    # Quick correlation check (overall, across all tickers)
    correlation = df[['Return', 'Mentions_Change']].corr().iloc[0, 1]
    print(f"Overall correlation between daily return and mention change: {correlation:.4f}")

    print()


def feat_eng(df):
    df['Mentions_Lag1'] = df.groupby('Ticker')['Mentions'].shift(1)
    df['Mentions_Lag2'] = df.groupby('Ticker')['Mentions'].shift(2)

    # Rolling average of Mentions over 7 days
    df['Mentions_Rolling7'] = df.groupby('Ticker')['Mentions'].transform(
        lambda x: x.rolling(window=7).mean()
    )

    # Volume change
    df['Volume_Change'] = df.groupby('Ticker')['Volume'].pct_change()

    # Rolling average of Volume over 7 days
    df['Volume_Rolling7'] = df.groupby('Ticker')['Volume'].transform(
        lambda x: x.rolling(window=7).mean()
    )

    # Drop any new NaNs created by shifting
    df = df.dropna(subset=['Mentions_Lag1', 'Return', 'Volume_Change'])
    return df


if __name__ == '__main__':
    path_to_data = "../data/final_csv.csv"
    data = pd.read_csv(path_to_data)
    df = pd.DataFrame(data)
    df = clean_data(df)
    eda(df)
    df = feat_eng(df)

