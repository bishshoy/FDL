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
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from cifar_resnets import *


class EnsembleNet(nn.Module):
    def __init__(self, bignet, cifar):
        super().__init__()
        self.bignet = [bignet]
        self.block = nn.Conv1d(2 * cifar, cifar, 1, 1, 0)
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
    def __init__(self, cifar, resnet):
        super().__init__()
        self.model1 = Net(cifar, resnet)
        self.model2 = Net(cifar, resnet)
        self.cuda()

    def forward(self, x):
        return self.model1(x), self.model2(x)


class Net(nn.Module):
    def __init__(self, cifar, resnet):
        super().__init__()
        if resnet == 20:
            model = resnet20
        elif resnet == 32:
            model = resnet32
        elif resnet == 44:
            model = resnet44
        elif resnet == 56:
            model = resnet56
        elif resnet == 110:
            model = resnet110
        self.resnet = model(num_classes=cifar).cuda()
        
        graph_nodes = get_graph_node_names(self.resnet)
        self.conv_nodes = [x for x in graph_nodes[1] if 'conv' in x]
        self.model = create_feature_extractor(self.resnet, return_nodes=dict(zip(self.conv_nodes, self.conv_nodes))).cuda()

    def forward(self, x):
        logits = self.resnet(x)
        features = []
        features_dict = self.model(x)
        
        for c in self.conv_nodes:
            features.append(features_dict[c])
        
        return logits, features
