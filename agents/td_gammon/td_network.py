import torch
from torch import nn


class TDNetwork(nn.Module):
    def __init__(self, input_size=14, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.net(x)
        return x.squeeze()
