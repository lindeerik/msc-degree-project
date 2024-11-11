"""
Neural network arhcitecture for predicting throughput. Has two hidden layers of size 64, ReLu activation and dropout included. 
"""

import torch
from torch import nn


class ThroughputPredictor(nn.Module):
    def __init__(self, input_size, dropout_rate=0.3):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 64)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(64, 64)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x.squeeze(-1)
