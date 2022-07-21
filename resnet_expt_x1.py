from __future__ import print_function
import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import math
from utils import load_lr, write_lr, load_momentum, write_momentum, read_params
import argparse
import multiprocessing
from cifar_resnets import *

parser = argparse.ArgumentParser()
parser.add_argument('--cifar', type=int)
parser.add_argument('--resnet', type=int)

args = parser.parse_args()


def lr_scheduler(optimizer, epoch):
    lr = 1e-5 + 0.1 * np.exp(-0.03 * epoch)
    if epoch > 220:
        lr = 2e-5
    optimizer.param_groups[0]["lr"] = lr
    return lr


def similarityLoss(x1, x2):
    loss = torch.mean((x2 - x1) ** 2.0)
    return loss


max_acc = 0


def train(model, train_loader, test_loader, num_epochs):
    global max_acc
    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4, nesterov=True)

    for epoch in range(num_epochs):
        collate_loss = []
        lr_scheduler(optimizer, epoch)
        print(
            "lr:",
            optimizer.param_groups[0]["lr"],
            "momentum:",
            optimizer.param_groups[0]["momentum"],
        )
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            model.train()
            optimizer.zero_grad()
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()
            collate_loss.append(loss.item())
            sys.stdout.write(
                "\rEpoch: {} ({:.0f}%) Loss: {:.6f}".format(
                    epoch, 100.0 * batch_idx / len(train_loader), np.mean(collate_loss)
                )
            )
            sys.stdout.flush()
        print("\n")
        accuracy, test_loss = test(model, test_loader)
        max_acc = max(accuracy, max_acc)
        print('Max accuracy:', max_acc)


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()  # sum up batch loss
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "Test set: Average loss: {:.8f}, Accuracy: {}/{} ({:.0f}%)".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return correct, test_loss


def main(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    root = "datasets"
    cifar = args.cifar
    resnet = args.resnet
    if cifar == 10:
        dataset = datasets.CIFAR10
    elif cifar == 100:
        dataset = datasets.CIFAR100
    train_loader = torch.utils.data.DataLoader(
        dataset(
            root=root,
            train=True,
            transform=transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
            download=False,
        ),
        batch_size=128,
        shuffle=True,
        num_workers=multiprocessing.cpu_count(),
        persistent_workers=True,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset(
            root=root,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        ),
        batch_size=128,
        shuffle=False,
        num_workers=multiprocessing.cpu_count(),
        persistent_workers=True,
    )

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
    resnet = model(num_classes=cifar).cuda()

    train(resnet, train_loader, test_loader, 400)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main(args)
