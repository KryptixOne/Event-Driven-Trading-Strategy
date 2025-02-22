from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


def simple_plot(df, sample_ticker="AAPL"):
    # Let's assume 'df' is your full DataFrame
    # and you want to visualize data for a given ticker
    sample_ticker = sample_ticker

    # Filter for just that ticker
    df_filtered = df[df['Ticker'] == sample_ticker].copy()
    df_filtered = df_filtered.sort_values(by='Date')

    # Compute mean of mentions for that ticker
    mentions_mean = df_filtered['Mentions'].mean()

    # Subtract mean and apply floor at 0
    df_filtered['Mentions_Adjusted'] = np.maximum(df_filtered['Mentions'] - mentions_mean, 0)

    # Plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Close Price on left Y-axis
    color_price = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Close Price', color=color_price)
    ax1.plot(df_filtered['Date'], df_filtered['Close Price'], color=color_price, label='Close Price')
    ax1.tick_params(axis='y', labelcolor=color_price)

    # Adjusted Mentions on right Y-axis
    ax2 = ax1.twinx()
    color_mentions = 'tab:orange'
    ax2.set_ylabel('Mentions (Mean-Adjusted, Floor=0)', color=color_mentions)
    ax2.plot(df_filtered['Date'], df_filtered['Mentions_Adjusted'], color=color_mentions,
             label='Mentions - Mean (Clipped at 0)')
    ax2.tick_params(axis='y', labelcolor=color_mentions)

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    plt.title(f'Close Price vs. Adjusted Mentions for {sample_ticker}')
    plt.show()
