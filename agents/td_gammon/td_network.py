import torch
from torch import nn


class TDNetwork(nn.Module):
    def __init__(self, input_size=14, hidden_size=256):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        x = x.unsqueeze(0)
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = self.fc3(x)
        return x.squeeze()
