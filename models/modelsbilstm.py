import torch.nn as nn

class BiLSTMModel(nn.Module):
    def __init__(self, input_size, lstm_hidden_size, lstm_layers, lstm_dropout, fc_dropout, output_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0,
            bidirectional=True          # Key: Bidirectional
        )
        self.fc_dropout = nn.Dropout(fc_dropout)
        # Double the output dimension of the bidirectional time series
        self.fc = nn.Linear(lstm_hidden_size * 2, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)               # (batch, seq_len, hidden*2)
        last_step = out[:, -1, :]           # (batch, hidden*2)
        out = self.fc_dropout(last_step)
        return self.fc(out)                 # (batch, 1)