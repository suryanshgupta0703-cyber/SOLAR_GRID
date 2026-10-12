import torch
import torch.nn as nn

class SolarLSTM(nn.Module):
    """
    Stacked LSTM for temporal time-series forecasting of solar generation.
    """
    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _   = self.lstm(x)
        last_out = out[:, -1, :]
        dropped  = self.dropout(last_out)
        return self.fc(dropped)
