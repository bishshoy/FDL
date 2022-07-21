from __future__ import print_function
import argparse, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
import math


class EnsembleNet(nn.Module):
    def __init__(self, bignet):
        super().__init__()
        self.bignet = bignet
        self.block = nn.Linear(20, 10)

    def forward(self, x):
        with torch.no_grad():
            self.bignet.eval()
            (p1, x1_1, x2_1), (p2, x1_2, x2_2) = self.bignet(x.detach())
        p = torch.cat((p1, p2), dim=1)
        p = self.block(p)
        return p, (x1_1, x1_2), (x2_1, x2_2)


class BigNet(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.model1 = Net(n)
        self.model2 = Net(n)
        self.model1.init_conv_layers()
        self.model2.init_conv_layers()

    def forward(self, x):
        return self.model1(x), self.model2(x)


class Net(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, n, 3, 2, 1),
            nn.BatchNorm2d(n),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(nn.Conv1d(7 * 7 * n, 10, 1, 1, 0))

    def forward(self, x):
        x = self.block1(x)
        x1 = x
        x = torch.flatten(x, 1).unsqueeze(-1)
        x = self.block2(x)
        x2 = x
        x = torch.flatten(x, 1)
        return x, x1, x2

    def init_conv_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.zero_()
