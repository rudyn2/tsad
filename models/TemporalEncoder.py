from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn as nn
import torch


class RNNEncoder(nn.Module):
    def __init__(self, hidden_size: int = 2046):
        super(RNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=8192, hidden_size=hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size + 3, 8192)

    def forward(self, embedding, action, embedding_length):
        """
        Output dim: BxHiddenSize
        """
        x_pack = pack_padded_sequence(embedding, embedding_length, batch_first=True)
        _, hidden = self.lstm(x_pack)
        hidden = hidden[0].squeeze(0)
        hidden_cat_action = torch.cat([hidden, action], dim=1)
        output = self.output(hidden_cat_action)
        return output

