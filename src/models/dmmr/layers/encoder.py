import os, sys
import torch.nn as nn

from .lstm import LSTM

class Encoder(nn.Module):
    def __init__(self, input_dim=310, hid_dim=64, n_layers=2):
        super(Encoder, self).__init__()
        self.theta = LSTM(input_dim, hid_dim, n_layers)
    def forward(self, x):
        x_h = self.theta(x)
        return x_h