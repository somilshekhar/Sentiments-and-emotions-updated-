import torch
import torch.nn as nn
import config

class CNNBiLSTM(nn.Module):
    def __init__(self, n_mels=config.N_MELS, n_classes=len(config.EMO_LABELS)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, 3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d((2, 2)),
        )
        self.lstm = nn.LSTM(
            input_size=(n_mels // 4) * 32,
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        conv_out = self.conv(x)
        b, c, h, w = conv_out.shape
        lstm_in = conv_out.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)
        lstm_out, _ = self.lstm(lstm_in)
        pooled = torch.mean(lstm_out, dim=1)
        logits = self.fc(pooled)
        return logits
