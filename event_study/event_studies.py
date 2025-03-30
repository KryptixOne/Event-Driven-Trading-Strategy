from preproc.preproc import eda, clean_data, feat_eng

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from matplotlib import pyplot as plt


def mention_spike_event_study(df,
                              ticker="AAPL",
                              mention_window=30,
                              mention_z=4,
                              horizon_list=[1, 5, 10, 20],
                              plot=True):
    """
    Identifies days where Mentions are abnormally high AND the daily return is positive.
    Then, computes future returns over multiple horizons to see if price tends to hold up or revert.

    Parameters
    ----------
    df : pd.DataFrame
        Your master DataFrame with columns at least:
         ['Date', 'Ticker', 'Close Price', 'Mentions'].
    ticker : str
        The ticker symbol to analyze (e.g., "AAPL").
    mention_window : int, optional
        Rolling window size for mean/std of Mentions.
    mention_z : float, optional
        Number of standard deviations above the mean to define a spike.
    horizon_list : list of int, optional
        List of day-horizons to measure forward returns (default [1, 5, 10]).
    plot : bool, optional
        If True, produces an event-study style plot of average post-event returns.

    Returns
    -------
    event_df : pd.DataFrame
        A filtered DataFrame for the chosen ticker with columns:
        ['Date', 'Close Price', 'Return', 'Mention_Spike', 'Price_Up',
         'Event_Day', 'Forward_Xd_Return'...]
        plus any rolling stats used.

    Prints summary stats and (optionally) shows a plot of average returns after event days.
    """

    # 1) Filter for the chosen ticker
    df_ticker = df[df['Ticker'] == ticker].copy()
    df_ticker = df_ticker.sort_values(by='Date')

    # 2) Calculate daily return
    df_ticker['Return'] = df_ticker['Close Price'].pct_change()

    # 3) Compute rolling mean and std of Mentions
    df_ticker['Mentions_RollMean'] = (
        df_ticker['Mentions']
        .rolling(window=mention_window, min_periods=mention_window)
        .mean()
    )
    df_ticker['Mentions_RollStd'] = (
        df_ticker['Mentions']
        .rolling(window=mention_window, min_periods=mention_window)
        .std()
    )

    # 4) Define mention spike day
    df_ticker['Mention_Spike'] = (
            df_ticker['Mentions'] >
            df_ticker['Mentions_RollMean'] + mention_z * df_ticker['Mentions_RollStd']
    )

    # 5) Price up day (Return > 0)
    df_ticker['Price_Up'] = df_ticker['Return'] > 0

    # 6) Event day: mention spike & price up
    df_ticker['Event_Day'] = df_ticker['Mention_Spike'] & df_ticker['Price_Up']

    # 7) Compute forward returns for each horizon
    #    We'll use the (1 + Return).cumprod() approach and shift
    df_ticker['CumProd'] = (1 + df_ticker['Return']).cumprod()

    max_horizon = max(horizon_list)
    # For each horizon h, define Forward_hd_Return = (CumProd.shift(-h) / CumProd) - 1
    for h in horizon_list:
        col_name = f'Forward_{h}d_Return'
        df_ticker[col_name] = (
                df_ticker['CumProd'].shift(-h) / df_ticker['CumProd'] - 1
        )

    # Drop rows where forward returns can't be computed (the last max_horizon days)
    df_ticker.dropna(subset=[f'Forward_{h}d_Return' for h in horizon_list], inplace=True)

    # 8) Compare average forward returns on event vs. non-event days
    event_mask = df_ticker['Event_Day']
    print(f"\nEvent Study for {ticker} (Mentions spike & price up)\n{'-' * 50}")

    for h in horizon_list:
        col_name = f'Forward_{h}d_Return'
        event_returns = df_ticker.loc[event_mask, col_name]
        nonevent_returns = df_ticker.loc[~event_mask, col_name]

        mean_event = event_returns.mean()
        mean_nonevent = nonevent_returns.mean()

        print(f"\nForward {h}-day Return:")
        print(f"  Event days mean: {mean_event:.4%}  (n={len(event_returns)})")
        print(f"  Non-event days:  {mean_nonevent:.4%}  (n={len(nonevent_returns)})")

        # T-test
        if len(event_returns) > 1 and len(nonevent_returns) > 1:
            t_stat, p_val = ttest_ind(event_returns, nonevent_returns, equal_var=False)
            print(f"  T-test: t-stat={t_stat:.3f}, p-value={p_val:.5f}")

    # 9) (Optional) Plot event-study style average return path up to max horizon
    if plot:
        # We'll align each event day at day 0 and track the returns for next X days
        # including day 0 as baseline (0%).
        # Create another column for daily cumulative product for the entire series
        df_ticker['CumProd'] = (1 + df_ticker['Return']).cumprod()

        # Recompute event_days after dropping for forward data
        event_days = df_ticker.index[df_ticker['Event_Day'] == True].tolist()

        # We'll plot out to the max horizon
        paths = []
        for idx in event_days:
            # If going out of range, skip
            if idx + max_horizon < df_ticker.index[-1]:
                start_val = df_ticker.loc[idx, 'CumProd']
                # Grab the range idx ... idx+max_horizon (by index position)
                # We'll find integer positions from the DataFrame
                # to do that safely, let's do get_loc
                pos_start = df_ticker.index.get_loc(idx)
                pos_end = pos_start + max_horizon
                if pos_end < len(df_ticker):
                    window_slice = df_ticker.iloc[pos_start: pos_end + 1]
                    window_vals = window_slice['CumProd'] / start_val - 1
                    paths.append(window_vals.values)

        if len(paths) > 0:
            aligned = np.array([p for p in paths if len(p) == max_horizon + 1])
            avg_path = aligned.mean(axis=0)

            plt.figure(figsize=(8, 5))
            plt.plot(range(max_horizon + 1), avg_path, label='Avg Post-Event Return')
            plt.axhline(0, color='red', linestyle='--', alpha=0.7, label='Zero')
            plt.title(f'Event Study: {ticker}\n(Mentions Spike & Price Up)')
            plt.xlabel('Days After Event')
            plt.ylabel('Cumulative Return (relative to Event Day)')
            plt.legend()
            plt.show()
        else:
            print("\nNo valid events to plot an average return path (check if you have enough event days).")

    return df_ticker


def event_study_ttest(df,
                      ticker="AAPL",
                      mention_window=30,
                      mention_z=4,
                      horizon_list=[1, 5, 10, 20],
                      price_up_min=0):
    """
    Runs an event study for the given ticker and parameters.
    Returns a dictionary of p-values, mean event returns, and mean non-event returns
    for each horizon in horizon_list.

    - mention_window: rolling window size for Mentions (mean + z*std).
    - mention_z: z-score threshold above rolling mean to define mention spike.
    - price_up_min: minimum daily return threshold (default=0 means any positive return).
    """

    df_ticker = df[df['Ticker'] == ticker].copy()
    df_ticker = df_ticker.sort_values('Date')

    # Daily return
    df_ticker['Return'] = df_ticker['Close Price'].pct_change()

    # Rolling mean/std for Mentions
    df_ticker['Mentions_RollMean'] = (
        df_ticker['Mentions']
        .rolling(window=mention_window, min_periods=mention_window)
        .mean()
    )
    df_ticker['Mentions_RollStd'] = (
        df_ticker['Mentions']
        .rolling(window=mention_window, min_periods=mention_window)
        .std()
    )

    # Define mention spike
    df_ticker['Mention_Spike'] = (
            df_ticker['Mentions'] >
            df_ticker['Mentions_RollMean'] + mention_z * df_ticker['Mentions_RollStd']
    )

    # Price up condition: Return > price_up_min
    df_ticker['Price_Up'] = df_ticker['Return'] > price_up_min

    # Event day = Mention_Spike & Price_Up
    df_ticker['Event_Day'] = df_ticker['Mention_Spike'] & df_ticker['Price_Up']

    # Forward returns
    df_ticker['CumProd'] = (1 + df_ticker['Return']).cumprod()

    # Compute forward returns for each horizon
    for h in horizon_list:
        col_name = f'Forward_{h}d_Return'
        df_ticker[col_name] = (
                df_ticker['CumProd'].shift(-h) / df_ticker['CumProd'] - 1
        )

    # Drop rows at end that can't have forward returns
    df_ticker.dropna(subset=[f'Forward_{h}d_Return' for h in horizon_list], inplace=True)

    # Prepare results
    results = {
        'ticker': ticker,
        'mention_window': mention_window,
        'mention_z': mention_z,
        'price_up_min': price_up_min,
        'num_events': int(df_ticker['Event_Day'].sum()),
        'num_total': len(df_ticker),
        'horizons': {}
    }

    event_mask = df_ticker['Event_Day'] == True

    # For each horizon, do a t-test
    for h in horizon_list:
        col_name = f'Forward_{h}d_Return'
        event_returns = df_ticker.loc[event_mask, col_name].dropna()
        nonevent_returns = df_ticker.loc[~event_mask, col_name].dropna()

        if len(event_returns) < 2 or len(nonevent_returns) < 2:
            # Not enough data for a valid test
            t_stat, p_val = (np.nan, np.nan)
            mean_event, mean_nonevent = (np.nan, np.nan)
        else:
            mean_event = event_returns.mean()
            mean_nonevent = nonevent_returns.mean()
            t_stat, p_val = ttest_ind(event_returns, nonevent_returns, equal_var=False)

        results['horizons'][h] = {
            'mean_event': mean_event,
            'mean_nonevent': mean_nonevent,
            't_stat': t_stat,
            'p_val': p_val,
            'count_event': len(event_returns),
            'count_nonevent': len(nonevent_returns)
        }

    return results


def hyperparam_search(df,
                      ticker,
                      mention_window_list=[10, 20, 30],
                      mention_z_list=[2, 3, 4],
                      horizon_list=[1, 5, 10, 20],
                      price_up_min_list=[0],
                      significance_metric='avg_pval'):
    """
    Loops over multiple mention_window, mention_z, price_up_min, etc.
    Calls event_study_ttest() each time.
    Stores results in a list (or DataFrame) and picks the best param set
    according to a significance metric.

    significance_metric can be:
    - 'avg_pval': average p-value across horizons (lower is better)
    - 'count_sig': count of horizons where p < 0.05 (higher is better)
    - 'avg_diff': average difference in forward returns (event - non-event) across horizons (higher is better)
    """

    results_list = []

    for mw in mention_window_list:
        for mz in mention_z_list:
            for pu in price_up_min_list:
                res = event_study_ttest(
                    df=df,
                    ticker=ticker,
                    mention_window=mw,
                    mention_z=mz,
                    horizon_list=horizon_list,
                    price_up_min=pu
                )
                results_list.append(res)

    # Convert results_list to a "flat" DataFrame
    flat_rows = []
    for r in results_list:
        base_cols = {
            'ticker': r['ticker'],
            'mention_window': r['mention_window'],
            'mention_z': r['mention_z'],
            'price_up_min': r['price_up_min'],
            'num_events': r['num_events'],
            'num_total': r['num_total']
        }
        # For each horizon, store p_val, mean_event, mean_nonevent
        # We'll also compute difference in means: mean_event - mean_nonevent
        row_horizons = {}
        for h, d in r['horizons'].items():
            row_horizons[f'h{h}_p_val'] = d['p_val']
            row_horizons[f'h{h}_mean_event'] = d['mean_event']
            row_horizons[f'h{h}_mean_nonevent'] = d['mean_nonevent']
            row_horizons[f'h{h}_mean_diff'] = d['mean_event'] - d['mean_nonevent'] if pd.notnull(
                d['mean_event']) and pd.notnull(d['mean_nonevent']) else np.nan

        flat_rows.append({**base_cols, **row_horizons})

    df_res = pd.DataFrame(flat_rows)

    # Define a scoring metric
    if significance_metric == 'avg_pval':
        # average p-value across the horizons (only the hX_p_val columns)
        horizon_cols = [c for c in df_res.columns if c.endswith('_p_val')]
        df_res['score'] = df_res[horizon_cols].mean(axis=1)  # lower is better
        df_res.sort_values('score', inplace=True)
    elif significance_metric == 'count_sig':
        # number of horizons with p < 0.05
        horizon_cols = [c for c in df_res.columns if c.endswith('_p_val')]
        df_res['score'] = df_res[horizon_cols].apply(lambda x: sum(x < 0.05), axis=1)  # higher is better
        df_res.sort_values('score', ascending=False, inplace=True)
    elif significance_metric == 'avg_diff':
        # average difference in forward returns across horizons
        horizon_cols = [c for c in df_res.columns if c.endswith('_mean_diff')]
        df_res['score'] = df_res[horizon_cols].mean(axis=1)  # higher is better
        df_res.sort_values('score', ascending=False, inplace=True)
    elif significance_metric == 'profit_significance':
        # 1) Compute average difference across horizons
        diff_cols = [c for c in df_res.columns if c.endswith('_mean_diff')]
        df_res['avg_diff'] = df_res[diff_cols].mean(axis=1)

        # 2) Compute average p-value across horizons
        pval_cols = [c for c in df_res.columns if c.endswith('_p_val')]
        df_res['avg_pval'] = df_res[pval_cols].mean(axis=1)

        # 3) Combine into a ratio (higher is better)
        eps = 1e-6
        # If avg_pval is extremely small, the ratio becomes huge => indicates strong significance + big effect
        df_res['score'] = df_res['avg_diff'] / (df_res['avg_pval'] + eps)

        # Sort descending: bigger ratio is better
        df_res.sort_values('score', ascending=False, inplace=True)

    # Return the entire DataFrame, best at top
    return df_res



if __name__ == '__main__':
    path_to_data = "../data/final_csv.csv"
    data = pd.read_csv(path_to_data)
    df = pd.DataFrame(data)
    df = clean_data(df)
    df = eda(df)
    df = feat_eng(df)

    mention_window_list = [5, 10, 20, 30, 60, 180]  # you could add 40, 50, etc.
    mention_z_list = [2, 3, 4]
    price_up_min_list = [0, 0.005, 0.01, 0.015, 0.02]  # e.g., 0 means >0, 0.01 means >1% daily up

    df_search_results = hyperparam_search(
        df=df,
        ticker='RGTI',
        mention_window_list=mention_window_list,
        mention_z_list=mention_z_list,
        price_up_min_list=price_up_min_list,
        horizon_list=[1, 5, 10, 20, 30, 40],  # e.g., we can try these horizons
        significance_metric='profit_significance'  # or 'count_sig' or 'avg_diff'
    )

    print(df_search_results.head(10))
