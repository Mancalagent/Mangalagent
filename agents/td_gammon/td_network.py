import torch
from torch import nn


class TDNetwork(nn.Module):
    def __init__(self, input_size=14, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        x = x.unsqueeze(0)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x.squeeze()
