"""
Neural network arhcitecture for predicting throughput
"""

import torch
from torch import nn


class ThroughputPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        # Define layers (no normalization layer)
        self.fc1 = nn.Linear(input_size, 64)  # First dense layer
        self.fc2 = nn.Linear(64, 64)  # Second dense layer
        self.fc3 = nn.Linear(64, 1)  # Final dense layer (single output)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # First dense layer + ReLU
        x = torch.relu(self.fc2(x))  # Second dense layer + ReLU
        x = self.fc3(x)  # Final dense layer (no activation)
        return x.squeeze(-1)
