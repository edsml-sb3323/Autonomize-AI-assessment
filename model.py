import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class CpGPredictor(nn.Module):
    def __init__(self, input_size=6, hidden_size=256, num_layers=2, dropout=0.3):
        super(CpGPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.layer_norm = nn.LayerNorm(2 * hidden_size)
        self.fc1 = nn.Linear(2 * hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        # One-hot encode the input
        x = torch.nn.functional.one_hot(x, num_classes=6).float()
        
        # Pack the padded sequence
        packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        # LSTM forward pass
        packed_lstm_out, _ = self.lstm(packed_x)
        
        # Unpack the output
        lstm_out, _ = pad_packed_sequence(packed_lstm_out, batch_first=True)
        
        # Get the last relevant output based on the sequence length
        out = torch.stack([lstm_out[i, lengths[i] - 1, :] for i in range(len(lengths))])

        # Apply layer normalization, dropout, and fully connected layers
        out = self.layer_norm(out)
        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        logits = self.fc2(out)

        return logits.squeeze()

# Weight initialization function
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                torch.nn.init.zeros_(param.data)
