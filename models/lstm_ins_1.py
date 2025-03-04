import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

"""
This model is a simple-task LSTM model that takes in a sequence of input (21 features) and one output (installment_done) a sequence of output

Model Structure: 1 Layer LSTM + 1 FC + Sigmoid
"""


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            # nn.ReLU(),
            nn.Sigmoid()
        )

    def forward(self, x, lengths):  # Accepts raw input and lengths
        # Pack sequence
        packed_x = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed_x)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        # Pass through MLP
        out = self.mlp(lstm_out)
        return out
