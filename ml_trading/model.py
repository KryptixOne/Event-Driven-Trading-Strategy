import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Final classifier
        self.fc = nn.Linear(hidden_dim, 2)  # 2 classes: buy or sell

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        # LSTM output shape: (batch_size, seq_length, hidden_dim)
        # We only need the last hidden state for classification
        lstm_out, (h_n, c_n) = self.lstm(x)

        # h_n shape: (num_layers, batch_size, hidden_dim)
        # Take the last layer's hidden state
        last_hidden = h_n[-1]  # shape: (batch_size, hidden_dim)

        out = self.fc(last_hidden)  # shape: (batch_size, 2)
        return out
