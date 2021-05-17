from torch.nn.utils.rnn import pack_padded_sequence
from .convolutional_rnn import Conv2dLSTM
import torch.nn as nn
import torch


class RNNEncoder(nn.Module):
    def __init__(self, num_layers: int = 2, hidden_size: int = 512):
        super(RNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = Conv2dLSTM(
            in_channels=513,  # Corresponds to input size
            out_channels=hidden_size,  # Corresponds to hidden size
            kernel_size=3,  # Int or List[int]
            num_layers=num_layers,
            bidirectional=True,
            stride=2, #dropout=0.5, dilation=2, 
            batch_first=True)

        #self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, batch_first=True)
        self.action_cod = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 16)
        )
        #self.output = nn.Linear(hidden_size+3, 512)

    def forward(self, embedding, action, embedding_length):
        """
        Output dim: BxHiddenSize
        """
        # Action (B, 3) => (B, 16) => (B, 1, 1, 4, 4)
        action_cod = self.action_cod(action).view(-1, 1, 1, 4, 4)
        # Action (B, 1, 1, 4, 4) => (B, 4, 1, 4, 4)
        action_cod = torch.cat((action_cod, action_cod, action_cod, action_cod), dim=1)
        # Cat embeddings and action (B, 4, 512, 4, 4) + (B, 4, 1, 4, 4) => (B, 4, 513, 4, 4)
        action_emb = torch.cat((embedding, action_cod), dim=2)
        
        x_pack = pack_padded_sequence(action_emb, embedding_length, batch_first=True)
        #x_pack = pack_padded_sequence(action_emb, torch.ones((32)), batch_first=True)
        h = None
        y, h = self.lstm(x_pack, h)
        # Output of lstm is stacked through all outputs (#outputs == #inputs), we get last output
        #y = y.data.view(embedding.shape)[:, -1, :, :, :].squeeze(dim=1)
        y = torch.mean(y.data.view(embedding.shape), dim=1)

        return y