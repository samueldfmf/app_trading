import torch.nn as nn
import torch

class LSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, dropout=0.2):
        super(LSTMWithAttention, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True, dropout=dropout)
        self.attention = nn.Linear(hidden_layer_size, 1)
        self.fc = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attn_output = torch.sum(lstm_out * attn_weights, dim=1)
        return self.fc(attn_output)
