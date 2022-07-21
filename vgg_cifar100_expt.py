from __future__ import print_function
import argparse, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import math
from vgg_cifar100_net import Net, BigNet, EnsembleNet
from utils import load_lr, write_lr, load_momentum, write_momentum, read_params


def lr_scheduler(optimizer, epoch):
    lr = 1e-5 + 0.1 * np.exp(-0.03 * epoch)
    if epoch > 220:
        lr = 2e-5
    optimizer.param_groups[0]["lr"] = lr
    return lr


def similarityLoss(x1, x2):
    loss = torch.mean((x2 - x1) ** 2.0)
    return loss


def train_joint_model(bignet, ensemblenet, train_loader, test_loader, num_epochs):
    model1, model2 = bignet.model1, bignet.model2

    optimizer1 = optim.SGD(
        model1.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4, nesterov=True
    )
    optimizer2 = optim.SGD(
        model2.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4, nesterov=True
    )
    accuracy_diff_optimizer = optim.SGD(
        bignet.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4, nesterov=True
    )

    similarity_optimizer = optim.SGD(
        bignet.parameters(), lr=1e-5, momentum=0.85, weight_decay=5e-4, nesterov=True
    )
    ensemblenet_optimizer = optim.SGD(
        ensemblenet.parameters(),
        lr=1e-1,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True,
    )

    max_accuracy = 0

    def zero_grad_all_optimizers():
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        accuracy_diff_optimizer.zero_grad()
        similarity_optimizer.zero_grad()
        ensemblenet_optimizer.zero_grad()

    def schedule_all_optimizers(epoch):
        lr_scheduler(optimizer1, epoch)
        lr_scheduler(optimizer2, epoch)
        lr = lr_scheduler(accuracy_diff_optimizer, epoch)
        # lr_scheduler(similarity_optimizer, epoch)
        # lr = lr_scheduler(ensemblenet_optimizer, epoch)
        print("lr:", lr)

    for epoch in range(num_epochs):
        _loss1, _loss2, _loss_acc, _loss12, _loss_ens = [], [], [], [], []
        schedule_all_optimizers(epoch)
        for batch_idx, (data, target) in enumerate(train_loader):
            model1.train(), model2.train(), ensemblenet.train()
            data, target = data.cuda(), target.cuda()
            # All optimizations of BigNet in one loop
            # zero_grad_all_optimizers()
            # (output1, x1_1, x2_1), (output2, x1_2, x2_2) = bignet(data)
            # loss1 = nn.CrossEntropyLoss()(output1, target)
            # loss2 = nn.CrossEntropyLoss()(output2, target)
            # loss_acc = nn.MSELoss()(loss1, loss2)
            # loss = loss1 + loss2 + loss_acc
            # loss.backward()
            # accuracy_diff_optimizer.step()

            # Train Model 1
            zero_grad_all_optimizers()
            output1, features1 = model1(data)
            loss1 = nn.CrossEntropyLoss()(output1, target)
            loss1.backward()
            optimizer1.step()

            # Train Model 2
            zero_grad_all_optimizers()
            output2, features2 = model2(data)
            loss2 = nn.CrossEntropyLoss()(output2, target)
            loss2.backward()
            optimizer2.step()

            # Train Model 1-2 difference
            zero_grad_all_optimizers()
            (output1, _), (output2, _) = bignet(data)
            loss1 = nn.CrossEntropyLoss()(output1, target)
            loss2 = nn.CrossEntropyLoss()(output2, target)
            loss_acc = nn.MSELoss()(loss1, loss2)
            loss_acc.backward()
            accuracy_diff_optimizer.step()
            # loss_acc = torch.zeros(1)

            # Train Dissimilarity between features
            zero_grad_all_optimizers()
            (_, features1), (_, features2) = bignet(data)
            loss12 = torch.zeros(1).cuda()
            for a, b in zip(features1, features2):
                loss12 += similarityLoss(a, b)
            loss12 /= len(features1)
            loss12 *= -1
            loss12.backward()
            similarity_optimizer.step()

            # Train EnsembleNet
            zero_grad_all_optimizers()
            output, _ = ensemblenet(data)
            loss_ens = nn.CrossEntropyLoss()(output, target)
            loss_ens.backward()
            ensemblenet_optimizer.step()

            _loss1.append(loss1.item()), _loss2.append(loss2.item()), _loss_acc.append(
                loss_acc.item()
            ), _loss12.append(loss12.item()), _loss_ens.append(loss_ens.item())
            if batch_idx % 1 == 0:
                sys.stdout.write(
                    "\rBigNet Epoch: {} [{}/{}]\tLoss: {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        np.mean(_loss1),
                        np.mean(_loss2),
                        np.mean(_loss_acc),
                        np.mean(_loss12),
                        np.mean(_loss_ens),
                    )
                )
                sys.stdout.flush()
        print("\n")
        _, _ = test(model1, test_loader)
        _, _ = test(model2, test_loader)
        ens_acc, _ = test(ensemblenet, test_loader)
        if ens_acc > max_accuracy:
            max_accuracy = ens_acc
        print("Max accuracy:", max_accuracy)


def train(model, train_loader, test_loader, num_epochs):
    optimizer = optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True
    )
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
            output, _ = model(data)
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


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output, _ = model(data)
            test_loss += nn.CrossEntropyLoss()(
                output, target
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
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


def main():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    root = "/.ssd/Datasets/"
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(
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
        num_workers=12,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(
            root=root,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        ),
        batch_size=1024,
        shuffle=False,
        num_workers=12,
        pin_memory=True,
    )

    # print('Model 1::')
    # model1 = Net()
    # train(model1, train_loader, test_loader, num_epochs=250)

    bignet = BigNet()
    ensemblenet = EnsembleNet(bignet)

    print("Pretrain Model1::")
    train(bignet.model1, train_loader, test_loader, 1)
    print("Pretrain Model2::")
    train(bignet.model2, train_loader, test_loader, 1)
    print("BigNet::")
    train_joint_model(bignet, ensemblenet, train_loader, test_loader, 10000)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
