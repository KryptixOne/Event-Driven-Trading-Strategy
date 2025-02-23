import torch
from torch.utils.data import Dataset, DataLoader
class FinancialDataset(Dataset):
    def __init__(self, df, feature_cols, label_col='Return', seq_length=20):
        """
        df: DataFrame with all data (assumed time-sorted).
        feature_cols: list of feature column names.
        label_col: name of the column with daily returns or next-day returns.
        seq_length: number of past days to include in each sample (window size).
        """
        self.seq_length = seq_length
        self.feature_cols = feature_cols

        # We define the label as next day's direction: 1 if next day return > 0 else 0
        # Shift the return column by -1 to label each day by next day's return
        df['Label'] = (df[label_col].shift(-1) > 0).astype(int)

        # Drop last row(s) that cannot have a label
        df = df.dropna(subset=['Label'])

        self.X = df[feature_cols].values
        self.y = df['Label'].values

        # For indexing sequences, we store the data as numpy arrays
        self.n_samples = len(df)

    def __len__(self):
        # The last (seq_length-1) rows can't form a full sequence
        return self.n_samples - self.seq_length

    def __getitem__(self, idx):
        # Window of features from idx to idx+seq_length
        X_seq = self.X[idx: idx + self.seq_length]
        y_label = self.y[idx + self.seq_length - 1]  # label is at the end of the window
        # Convert to PyTorch tensors
        X_seq = torch.tensor(X_seq, dtype=torch.float32)
        y_label = torch.tensor(y_label, dtype=torch.long)

        return X_seq, y_label


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

