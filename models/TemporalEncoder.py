import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch import Tensor

from models.convolutional_rnn import Conv2dLSTM


class RNNEncoder(nn.Module):
    def __init__(self, num_layers: int = 2, hidden_size: int = 512, action__chn: int = 64, speed_chn: int = 64):
        super(RNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = Conv2dLSTM(
            in_channels=512 + action__chn + speed_chn,  # Corresponds to input size
            out_channels=hidden_size,  # Corresponds to hidden size
            kernel_size=3,  # Int or List[int]
            num_layers=num_layers,
            bidirectional=True,
            stride=2,  # dropout=0.5, dilation=2,
            batch_first=True)

        self.action_cod = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 16)
        )
        self.action_conv = nn.Conv2d(
            1, action__chn, kernel_size=(1, 1)
        )
        self.speed_cod = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 16)
        )
        self.speed_conv = nn.Conv2d(
            1, speed_chn, kernel_size=(1, 1)
        )

        self.output_conv = nn.Conv2d(
            hidden_size * 2 * 4, 512, kernel_size=(1, 1)
        )

    def forward(self, embedding, action, speed, embedding_length):
        """
        Output dim: BxHiddenSize
        """
        # Action (B, 3) => (B, 16) => (B, 1, 4, 4)
        action_cod = self.action_cod(action).view(-1, 1, 4, 4)
        # Action (B, 1, 4, 4) => (B, 64, 4, 4) => (B, 1, 64, 4, 4)
        action_cod = self.action_conv(action_cod).unsqueeze(dim=1)
        # Action (B, 1, 64, 4, 4) => (B, 4, 64, 4, 4)
        action_cod = torch.cat((action_cod, action_cod, action_cod, action_cod), dim=1)

        # Speed (B, 1) => (B, 16) => (B, 1, 4, 4)
        speed_cod = self.speed_cod(speed).view(-1, 1, 4, 4)
        # Speed (B, 1, 4, 4) => (B, 64, 4, 4) => (B, 1, 64, 4, 4)
        speed_cod = self.speed_conv(speed_cod).unsqueeze(dim=1)
        # Speed (B, 1, 64, 4, 4) => (B, 4, 64, 4, 4)
        speed_cod = torch.cat((speed_cod, speed_cod, speed_cod, speed_cod), dim=1)

        # Cat embeddings and action (B, 4, 512, 4, 4) + (B, 4, 1, 4, 4) => (B, 4, 513, 4, 4)
        action_emb = torch.cat((embedding, action_cod, speed_cod), dim=2)

        x_pack = pack_padded_sequence(action_emb, embedding_length, batch_first=True)
        # x_pack = pack_padded_sequence(action_emb, torch.ones((32)), batch_first=True)
        h = None
        y, h = self.lstm(x_pack, h)
        # Output of lstm is stacked through all outputs (#outputs == #inputs), we get last output
        y = self.output_conv(y.data.view(embedding.shape[0], -1, embedding.shape[-2], embedding.shape[-1]))
        # y = y.data.view(embedding.shape)[:, -1, :, :, :].squeeze(dim=1)
        # y = torch.mean(y.data.view(embedding.shape), dim=1)

        return y


class VanillaRNNEncoder(nn.Module):
    def __init__(self, num_layers: int = 2, hidden_size: int = 512, action__chn: int = 64, speed_chn: int = 64,
                 dropout: float = 0, bidirectional: bool = False):
        super(VanillaRNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=8192 + action__chn + speed_chn,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True
        )

        self.action_cod = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, action__chn)
        )

        self.speed_cod = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, speed_chn)
        )

        input_fc_size = 4 * (2 if bidirectional else 1) * hidden_size   # times 4 because every sequence has length 4
        self.output_fc = nn.Linear(input_fc_size, 512*4*4)

    def forward(self, embedding, action, speed):
        """
        Output dim: BxHiddenSize
        """
        # Action (B, 3) => (B, action_chn) => (B, 1, action_chn)
        action_cod = self.action_cod(action).unsqueeze(dim=1)
        # Action (B, 1, action_chn) => (B, 4, action_chn)
        action_cod = torch.cat((action_cod, action_cod, action_cod, action_cod), dim=1)
        # Speed (B, 3) => (B, 1, speed_chn)
        speed_cod = self.speed_cod(speed).unsqueeze(dim=1)
        # Speed (B, 1, speed_chn) => (B, 4, speed_chn)
        speed_cod = torch.cat((speed_cod, speed_cod, speed_cod, speed_cod), dim=1)

        # (B, T, 512, 4, 4) => (B, T, 8192)
        vis_embedding = embedding.view(embedding.shape[0], embedding.shape[1], -1)

        # Cat embeddings and action (B, 4, 8192) + (B, 4, speed_chn) + (B, 4, action_chn) =>
        # (B, 4, 8192 + speed_chn + action_chn)
        action_emb = torch.cat((vis_embedding, action_cod, speed_cod), dim=2)

        y, _ = self.lstm(action_emb)
        # (B, 4, (2 if bidirectional else 1)*hidden_size)
        y = self.output_fc(y.reshape(embedding.shape[0], -1))
        # y (B, 512, 4, 4)
        return y.view(-1, embedding.shape[-3], embedding.shape[-2], embedding.shape[-1])

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
