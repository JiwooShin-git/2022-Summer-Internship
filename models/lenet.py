import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import *


# TODO: to be implemented...
class LeNet(nn.Module):
    def __init__(self, data, num_classes):
        super(LeNet, self).__init__()

        if data == 'MNIST':
            self.conv1 = Block(1, 32)
            self.linear = nn.Linear(128 * 3 * 3, num_classes)
        else:
            self.conv1 = Block(3, 32)
            self.linear = nn.Linear(128 * 4 * 4, num_classes)

        self.conv2 = Block(32, 64)
        self.conv3 = Block(64, 128)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class Block(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=5, stride=1, padding=2)
        self.bn = nn.BatchNorm2d(outplanes)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.prelu(self.conv1(x))
        x = self.prelu(self.conv2(x))
        x = self.bn(x)
        x = self.pool(x)
        return x
