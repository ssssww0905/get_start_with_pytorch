import torch
from torch import nn
device = "cuda" if torch.cuda.is_available() else "cpu"


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

    def forward(self, x):
        return x
