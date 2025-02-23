import torch
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data import Dataset
import numpy as np


class FinancialDataset(Dataset):
    def __init__(self, df, feature_cols, label_col='Return', seq_length=20):
        """
        Creates a dataset ensuring that each sequence only contains data from the same ticker.

        df: DataFrame with all data (assumed time-sorted by date).
        feature_cols: list of feature column names.
        label_col: name of the column with daily returns or next-day returns.
        seq_length: number of past days to include in each sample (window size).
        """
        self.seq_length = seq_length
        self.feature_cols = feature_cols
        self.sequences = []

        # Compute labels per ticker
        df['Label'] = df.groupby('Ticker')[label_col].shift(-1) > 0  # Binary label (1 for up, 0 for down)
        df['Label'] = df['Label'].astype(int)  # Convert True/False to 1/0

        # Drop rows where we cannot get a label
        df = df.dropna(subset=['Label'])

        # Store sequences separately for each stock
        for ticker, group in df.groupby('Ticker'):
            group = group.sort_values('Date')  # Ensure time order within each stock

            X = group[feature_cols].values
            y = group['Label'].values

            # Create rolling sequences per stock
            for i in range(len(group) - seq_length):
                self.sequences.append((X[i:i + seq_length], y[i + seq_length - 1]))

        print(f"Total Sequences Created: {len(self.sequences)}")  # Debugging output

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        X_seq, y_label = self.sequences[idx]
        return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y_label, dtype=torch.long)


class FinancialWeeklyDataset(Dataset):
    def __init__(self, df, feature_cols, seq_length=20):
        """
        df: DataFrame with columns including 'Label' (1 or 0),
            and feature_cols to be used as input features.
        seq_length: number of past days to include in each sample.
        """
        df = df.dropna(subset=['Label'])  # remove rows where we can't form a label
        self.seq_length = seq_length
        self.feature_cols = feature_cols

        # Convert DataFrame to numpy arrays
        self.X = df[feature_cols].values
        self.y = df['Label'].values
        self.n_samples = len(df)

    def __len__(self):
        return self.n_samples - self.seq_length

    def __getitem__(self, idx):
        # Window of features from idx to idx+seq_length
        X_seq = self.X[idx: idx + self.seq_length]
        # Label is the label of the last row in the sequence
        y_label = self.y[idx + self.seq_length - 1]

        X_seq = torch.tensor(X_seq, dtype=torch.float32)
        y_label = torch.tensor(y_label, dtype=torch.long)

        return X_seq, y_label

