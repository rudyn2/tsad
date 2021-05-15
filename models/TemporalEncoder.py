from torch.nn.utils.rnn import pack_padded_sequence
from .convolutional_rnn import Conv2dLSTM
import torch.nn as nn
import torch


class RNNEncoder(nn.Module):
    def __init__(self, num_layers: int = 2, hidden_size: int = 2046):
        super(RNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = Conv2dLSTM(
            in_channels=512,  # Corresponds to input size
            out_channels=hidden_size,  # Corresponds to hidden size
            kernel_size=3,  # Int or List[int]
            num_layers=num_layers,
            bidirectional=True,
            stride=2, #dropout=0.5, dilation=2, 
            batch_first=True)
        #self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size+3, 512)

    def forward(self, embedding, action, embedding_length):
        """
        Output dim: BxHiddenSize
        """
        x_pack = pack_padded_sequence(embedding, embedding_length, batch_first=True)
        h = None
        _, hidden = self.lstm(x_pack, h)
        print(hidden.shape)
        print(action.shape)
        hidden = hidden[0].squeeze(0)
        hidden_cat_action = torch.cat([hidden, action], dim=1)
        output = self.output(hidden_cat_action)
        return output
