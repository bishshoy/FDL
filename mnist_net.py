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
    def __init__(self):
        super().__init__()
        self.model1 = Net()
        self.model2 = Net()
        self.model1.load_state_dict(torch.load("./saved-models/mnist-net.ckpt"))
        self.model2.load_state_dict(torch.load("./saved-models/mnist-net.ckpt"))

    def forward(self, x):
        return self.model1(x), self.model2(x)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )
        self.block2 = nn.Sequential(
            nn.Linear(9216, 128),
            nn.Dropout2d(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.block1[:2](x)
        x1 = x
        x = self.block1[2:](x)
        x2 = x
        x = torch.flatten(x, 1)
        x = self.block2(x)
        return x, x1, x2
