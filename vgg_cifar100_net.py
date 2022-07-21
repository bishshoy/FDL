from __future__ import print_function
import argparse, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torchvision.models as models
import numpy as np
import math


class EnsembleNet(nn.Module):
    def __init__(self, bignet):
        super().__init__()
        self.bignet = [bignet]
        self.block = nn.Conv1d(200, 100, 1, 1, 0)
        self.cuda()

    def forward(self, x):
        self.bignet[0].eval()
        with torch.no_grad():
            (p1, _), (p2, _) = self.bignet[0](x.detach())
        p = torch.cat((p1, p2), dim=1)
        p = self.block(p.unsqueeze(-1))
        p = p[:, :, 0]
        return p, 0


class BigNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = Net()
        self.model2 = Net()
        self.cuda()

    def forward(self, x):
        return self.model1(x), self.model2(x)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg_net = [models.vgg16_bn(pretrained=False, progress=False)]
        self.fe = self.vgg_net[0].features
        self.conv_layers_idx = [0]
        for i, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                self.conv_layers_idx.append(i - 1)
        self.cl = nn.Sequential(nn.Conv1d(512, 100, 1, 1, 0))
        self.cuda()

    def forward(self, x):
        features = []
        for i, j in zip(self.conv_layers_idx, self.conv_layers_idx[1:]):
            x = self.fe[i:j](x)
            features.append(x)
        x = self.fe[j:](x)[:, :, 0]
        x = self.cl(x)[:, :, 0]
        return x, features
