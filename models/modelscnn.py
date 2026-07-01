import torch.nn as nn

class CNNOnlyModel(nn.Module):
    def __init__(self, input_channels, cnn_out_channels, fc_dropout, output_size):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=64, out_channels=cnn_out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(cnn_out_channels),
            nn.ReLU(inplace=True)
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)   # Global average pooling, eliminating time steps
        self.fc_dropout = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(cnn_out_channels, output_size)

    def forward(self, x):
        # x: (batch, seq_len, features) → (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)               # (batch, cnn_out_channels, seq_len)
        x = self.global_pool(x)       # (batch, cnn_out_channels, 1)
        x = x.squeeze(-1)             # (batch, cnn_out_channels)
        x = self.fc_dropout(x)
        return self.fc(x)             # (batch, 1)