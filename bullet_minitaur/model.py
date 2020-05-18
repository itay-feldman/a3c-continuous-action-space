import torch
import torch.nn as nn
import numpy as np


HID_SIZE = 128


class ModelA3C(nn.Module):
    """
        The model class
        The neural network architecture is defined here
    """
    def __init__(self, obs_size, act_size):
        super(ModelA3C, self).__init__()

        self.base = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.ReLU(),
        )
        self.mu = nn.Sequential(
            nn.Linear(HID_SIZE, act_size),
            nn.Tanh(),
        )
        self.var = nn.Sequential(
            nn.Linear(HID_SIZE, act_size),
            nn.Softplus(),
        )
        self.value = nn.Linear(HID_SIZE, 1)

    def forward(self, x):
        base_out = self.base(x)
        return self.mu(base_out), self.var(base_out), self.value(base_out)
