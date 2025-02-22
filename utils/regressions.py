import statsmodels.api as sm


def fit_ols_model(df):
    # Example: pick a ticker
    sample_ticker = "AAPL"
    df_sample = df[df['Ticker'] == sample_ticker].dropna(subset=['Return', 'Mentions_Lag1', 'Volume_Change'])

    X = df_sample[['Mentions_Lag1', 'Volume_Change']]
    y = df_sample['Return']

    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print(model.summary())