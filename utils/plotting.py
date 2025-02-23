from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt



def plot_training_curves(train_losses, val_losses, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss", marker='o')
    plt.plot(epochs, val_losses, label="Validation Loss", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label="Validation Accuracy", marker='o', color='green')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()

    plt.show()

def plot_signals_on_market_data(df_test_signals):
    """
    Plots the test set's market data (Close Price) and overlays buy/sell signals.

    - Green triangles indicate buy signals.
    - Red triangles indicate sell signals.

    Parameters:
    - df_test_signals: DataFrame containing test period with 'Date', 'Close Price', and 'Signal'
    """
    df_test_signals = df_test_signals.copy()
    df_test_signals['Date'] = pd.to_datetime(df_test_signals['Date'])

    plt.figure(figsize=(12, 6))

    # Plot the closing price
    plt.plot(df_test_signals['Date'], df_test_signals['Close Price'], label='Close Price', color='blue', linewidth=1)

    # Find indices of Buy & Sell signals
    buy_signals = df_test_signals[df_test_signals['Signal'] == 'BUY']
    sell_signals = df_test_signals[df_test_signals['Signal'] == 'SELL']

    # Plot Buy signals
    plt.scatter(buy_signals['Date'], buy_signals['Close Price'],
                color='green', marker='^', s=100, label='Buy Signal')

    # Plot Sell signals
    plt.scatter(sell_signals['Date'], sell_signals['Close Price'],
                color='red', marker='v', s=100, label='Sell Signal')

    # Labels & legend
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Buy/Sell Signals on Market Data')
    plt.legend()
    plt.grid(True)
    plt.show()


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
