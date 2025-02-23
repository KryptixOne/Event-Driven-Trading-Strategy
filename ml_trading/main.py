import torch
import pandas as pd
from ml_trading.dataset import FinancialDataset, FinancialWeeklyDataset
from preproc.ml_preproc import split_data_by_date, scale_features, create_weekly_labels_and_split, \
    split_data_by_explicit_dates, scale_features_parallel
from ml_trading.train import train_model
from ml_trading.model import LSTMClassifier
from torch.utils.data import DataLoader
from preproc.preproc import clean_data, eda
from utils.plotting import plot_signals_on_market_data, plot_training_curves
from utils.utils import save_scaled_data
import numpy as np

def run_weekly_model(df):
    # Step A: Generate weekly labels
    df_train, df_val, df_test = create_weekly_labels_and_split(df)

    # Step B: Scale chosen features
    feature_cols = ['Mentions', 'Volume', 'Open Price', 'Close Price',
                    'Return', 'Mentions_Change', 'Volume_Change']

    df_train, df_val, df_test, scaler = scale_features(df_train, df_val, df_test, feature_cols)

    # Step C: Create Datasets
    seq_length = 20
    train_dataset = FinancialWeeklyDataset(df_train, feature_cols=feature_cols, seq_length=seq_length)
    val_dataset = FinancialWeeklyDataset(df_val, feature_cols=feature_cols, seq_length=seq_length)
    test_dataset = FinancialWeeklyDataset(df_test, feature_cols=feature_cols, seq_length=seq_length)

    # Step D: Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Step E: Model & Training
    input_dim = len(feature_cols)
    model = LSTMClassifier(input_dim=input_dim, hidden_dim=64, num_layers=1, dropout=0.2)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trained_model = train_model(model, train_loader, val_loader, epochs=10, lr=1e-3, device=device)

    # Step F: Inference on Test
    df_test_signals = generate_weekly_signals(trained_model, df_test, test_dataset, feature_cols, seq_length, device)

    return df_test_signals


def generate_weekly_signals(model, df_test, test_dataset, feature_cols, seq_length, device='cpu'):
    model.eval()
    model.to(device)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    preds_list = []
    actual_list = []

    with torch.no_grad():
        idx_counter = 0
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            logits = model(X_batch)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            preds_list.extend(preds)
            actual_list.extend(y_batch.cpu().numpy())

            idx_counter += 1

    # Align predictions with the test data
    df_test_shifted = df_test.iloc[seq_length:].copy()
    df_test_shifted['Predicted'] = preds_list
    df_test_shifted['ActualLabel'] = actual_list

    df_test_shifted['Signal'] = df_test_shifted['Predicted'].apply(lambda x: 'BUY' if x == 1 else 'SELL')

    return df_test_shifted


def create_signals(df,
                   hidden_dim=128,
                   num_layers=2,
                   dropout=0.3,
                   lr=1e-4,
                   lookback_days=252,
                   save_data=False
                   ):
    """
    Main function that:
    1) Splits data
    2) Scales features
    3) Builds datasets & dataloaders
    4) Trains LSTM
    5) Generates signals on test set
    Returns: df_test with predicted signals
    """
    # -----------------------------------------
    # 1. Split data by date
    df_train, df_val, df_test = split_data_by_explicit_dates(df)

    # -----------------------------------------
    # 2. Scale features
    feature_cols = ['Mentions', 'Volume', 'Open Price', 'Close Price', 'Mentions_Change']

    #df_train, df_val, df_test = scale_features_parallel(df_train, df_val, df_test,
    #                                                            feature_cols,
    #                                                             lookback_days=lookback_days,
    #                                                             num_workers=8)
    df_train, df_val, df_test, _ = scale_features(df_train, df_val, df_test, feature_cols)
    if save_data:
        save_scaled_data(df_train, df_val, df_test, save_dir="./data/scaled_data")
    # -----------------------------------------
    # 3. Create PyTorch Datasets & Loaders
    seq_length = 20

    train_dataset = FinancialDataset(df_train, feature_cols=feature_cols,
                                     label_col='Return', seq_length=seq_length,
                                     return_indices=False)

    val_dataset = FinancialDataset(df_val, feature_cols=feature_cols,
                                   label_col='Return', seq_length=seq_length,
                                   return_indices=False)

    test_dataset = FinancialDataset(df_test, feature_cols=feature_cols,
                                    label_col='Return', seq_length=seq_length,
                                    return_indices=True)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # -----------------------------------------
    # 4. Instantiate Model & Train
    input_dim = len(feature_cols)
    hidden_dim = hidden_dim
    num_layers = num_layers
    dropout = dropout
    model = LSTMClassifier(input_dim=input_dim, hidden_dim=hidden_dim,
                           num_layers=num_layers,
                           dropout=dropout, )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device is {device}')

    trained_model, train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader,
        epochs=1, lr=lr, device=device, save_path="../checkpoints/best_model.pth"
    )
    # -----------------------------------------
    # 5. Generate Buy/Sell Signals on Test Set
    model.eval()
    model.to(device)

    # We need to reconstruct the test data indices so we can store predictions

    test_signals = []
    test_labels = []
    all_indices = []

    with torch.no_grad():
        for X_batch, y_batch, idx_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_true = y_batch.numpy()
            row_ids = idx_batch.numpy()  # original DataFrame indices

            test_signals.extend(preds)
            test_labels.extend(y_true)
            all_indices.extend(row_ids)

    df_test_signals = df_test.copy()

    df_test_signals['Predicted'] = np.nan
    df_test_signals['ActualLabel'] = np.nan

    # 2) Assign predictions by index
    df_test_signals.loc[all_indices, 'Predicted'] = test_signals
    df_test_signals.loc[all_indices, 'ActualLabel'] = test_labels

    # 3) Create a buy/sell column
    df_test_signals['Signal'] = df_test_signals['Predicted'].apply(
        lambda x: 'BUY' if x == 1 else ('SELL' if x == 0 else np.nan)
    )

    return df_test_signals,train_losses, val_losses, val_accuracies



if __name__ == "__main__":
    plot = True

    single_day_test = True
    weekly_test = False

    data_path = "../data/final_csv.csv"
    # Suppose your DataFrame is called df_all
    print('loading data...')
    df_all = pd.read_csv(data_path)

    df_all['Date'] = pd.to_datetime(df_all['Date'])  # Ensure Date is in datetime format
    # Get the overall start and end dates of the dataset
    start_date = df_all['Date'].min()
    end_date = df_all['Date'].max()
    # Calculate the number of entries per ticker
    ticker_counts = df_all.groupby('Ticker').size().reset_index(name='Total Entries')

    print(f'Cleaning..')
    df_all = clean_data(df_all)
    df_all = eda(df_all)
    df_all = df_all.dropna(subset=['Volume'])
    print(f'Data in form of {df_all.head(5)}')
    print('checking for nans')
    print(df_all.isna().sum())  # Check for NaNs in dataset

    if single_day_test:
        print('running single day')
        df_signals, train_losses, val_losses, val_accuracies = create_signals(df_all,
                                                                              hidden_dim=128,
                                                                              num_layers=2,
                                                                              dropout=0.3,
                                                                              lr=1e-4,
                                                                              lookback_days=180
                                                                              # 180 day window used for scaling.
                                                                              )
        print(df_signals[['Date', 'Close Price', 'Predicted', 'Signal']].head(30))

        # Now you can evaluate the strategy or check accuracy on the test set
        accuracy = (df_signals['Predicted'] == df_signals['ActualLabel']).mean()
        print(f"Test Accuracy: {accuracy:.3f}")
        if plot:
            plot_training_curves(train_losses, val_losses, val_accuracies)
            plot_signals_on_market_data(df_signals)

    if weekly_test:
        df_test_signals = run_weekly_model(df_all)
        print(df_test_signals[['Date', 'Close Price', 'Signal', 'Predicted', 'ActualLabel', 'NextWeekReturn']])

        # Evaluate performance
        accuracy = (df_test_signals['Predicted'] == df_test_signals['ActualLabel']).mean()
        print(f"Weekly Direction Accuracy: {accuracy:.3f}")

    # Potential backtest / PnL calc using next 5-day window
    # ...
