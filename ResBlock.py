from torch import nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """
    Iniialize a residual block with two convolutions followed by batchnorm layers
    """
    def __init__(self, in_size: int, hidden_size: int, out_size: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_size, hidden_size, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_size, out_size, 3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(hidden_size)
        self.batchnorm2 = nn.BatchNorm2d(hidden_size)
        self.batchnorm3 = nn.BatchNorm2d(out_size)

    def convblock(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        x = F.relu(self.batchnorm3(self.conv3(x)))
        return x

    def forward(self, x):
        return x + self.convblock(x)