import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class FinancialDataset(Dataset):
    def __init__(self, df, feature_cols, label_col='Return', seq_length=20, return_indices=False):
        self.seq_length = seq_length
        self.feature_cols = feature_cols
        self.return_indices = return_indices
        self.sequences = []

        df['Label'] = df.groupby('Ticker')[label_col].shift(-1) > 0
        df['Label'] = df['Label'].astype(int)
        df = df.dropna(subset=['Label'])

        for ticker, group in df.groupby('Ticker'):
            group = group.sort_values('Date')
            X = group[feature_cols].values
            y = group['Label'].values
            idx_arr = group.index.values  # store the original row indices

            for i in range(len(group) - seq_length):
                seq_X = X[i:i + seq_length]
                seq_y = y[i + seq_length - 1]
                # only save row index if return_indices=True
                row_idx = idx_arr[i + seq_length - 1] if self.return_indices else None
                self.sequences.append((seq_X, seq_y, row_idx))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        X_seq, y_label, row_idx = self.sequences[idx]
        if self.return_indices:
            return (
                torch.tensor(X_seq, dtype=torch.float32),
                torch.tensor(y_label, dtype=torch.long),
                row_idx
            )
        else:
            return (
                torch.tensor(X_seq, dtype=torch.float32),
                torch.tensor(y_label, dtype=torch.long)
            )

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

