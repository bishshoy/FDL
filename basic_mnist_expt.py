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
from basic_mnist import BigNet, EnsembleNet, Net


def train_ensemble_net(args, ensemblenet, device, train_loader, epoch):
    ensemblenet.train(), ensemblenet.bignet.eval()
    optimizer = optim.SGD(
        ensemblenet.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4
    )
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, _, _ = ensemblenet(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            sys.stdout.write(
                "\rEnsemble Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            sys.stdout.flush()
    print("\n")


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, _, _ = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            sys.stdout.write(
                "\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            sys.stdout.flush()
    print("\n")


def similarityLoss(x1, x2):
    loss = ((x2 - x1) ** 2).mean()
    return loss


def train_joint_model(args, bignet, device, train_loader, epoch):
    bignet.train()
    model1, model2 = bignet.model1, bignet.model2
    optimizer1 = optim.SGD(
        bignet.model1.parameters(), lr=1e-1, momentum=0.95, weight_decay=1e-4
    )
    optimizer2 = optim.SGD(
        bignet.model2.parameters(), lr=1e-1, momentum=0.95, weight_decay=1e-4
    )
    accuracy_diff_optimizer = optim.SGD(
        bignet.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-4
    )
    all_optimizer = optim.SGD(
        bignet.parameters(), lr=1e-2, momentum=0.85, weight_decay=1e-4
    )
    similarity_optimizer = optim.SGD(
        bignet.parameters(), lr=2e-4, momentum=0.85, weight_decay=1e-4
    )
    _loss, _loss1, _loss2, _loss12, _loss_acc = [], [], [], [], []

    def zero_grad_all_optimizers():
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        accuracy_diff_optimizer.zero_grad()
        all_optimizer.zero_grad()
        similarity_optimizer.zero_grad()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # All optimizations of BigNet in one loop
        # zero_grad_all_optimizers()
        # (output1, x1_1, x2_1), (output2, x1_2, x2_2) = bignet(data)
        # loss1 = nn.CrossEntropyLoss()(output1, target)
        # loss2 = nn.CrossEntropyLoss()(output2, target)
        # loss_acc = nn.MSELoss()(loss1, loss2)
        # loss = loss1 + loss2 + loss_acc
        # loss.backward()
        # accuracy_diff_optimizer.step()

        zero_grad_all_optimizers()
        output1, x1_1, x2_1 = bignet.model1(data)
        loss1 = nn.CrossEntropyLoss()(output1, target)
        loss1.backward()
        optimizer1.step()

        zero_grad_all_optimizers()
        output2, x1_2, x2_2 = bignet.model2(data)
        loss2 = nn.CrossEntropyLoss()(output2, target)
        loss2.backward()
        optimizer2.step()

        if epoch != 1:
            zero_grad_all_optimizers()
            (output1, x1_1, x2_1), (output2, x1_2, x2_2) = bignet(data)
            loss1 = nn.CrossEntropyLoss()(output1, target)
            loss2 = nn.CrossEntropyLoss()(output2, target)
            loss_acc = nn.MSELoss()(loss1, loss2)
            loss_acc.backward()
            accuracy_diff_optimizer.step()
        else:
            loss_acc = torch.zeros(1)

        zero_grad_all_optimizers()
        (output1, x1_1, x2_1), (output2, x1_2, x2_2) = bignet(data)
        loss12 = -1 * similarityLoss(x1_1, x1_2)
        loss12.backward()
        similarity_optimizer.step()

        loss = loss1 + loss2
        _loss.append(loss.item()), _loss1.append(loss1.item()), _loss2.append(
            loss2.item()
        ), _loss_acc.append(loss_acc.item()), _loss12.append(loss12.item())
        if batch_idx % 1 == 0:
            sys.stdout.write(
                "\rBigNet Epoch: {} [{}/{}]\tLoss: {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    np.mean(_loss),
                    np.mean(_loss1),
                    np.mean(_loss2),
                    np.mean(_loss_acc),
                    np.mean(_loss12),
                )
            )
            sys.stdout.flush()
    print("\n")


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    _loss12 = [0]
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, x1, x2 = model(data)
            # loss12 = similarityLoss(x1[:,0,:,:], x1[:,1,:,:])
            try:
                x1_1, x1_2 = x1
                x2_1, x2_2 = x2
                loss12 = similarityLoss(x1_1, x1_2)
                _loss12.append(loss12.item())
            except:
                pass
            test_loss += nn.CrossEntropyLoss()(
                output, target
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "Test set: Average loss: {:.4f}, Similarity loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            test_loss,
            np.mean(_loss12),
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def main():
    args = argparse.ArgumentParser().parse_args()
    args.gamma = 0.7
    args.log_interval = 10
    device = torch.device("cuda")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("../data", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        dataset1, batch_size=256, num_workers=20, shuffle=True, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset2, batch_size=10000, num_workers=20, shuffle=False, pin_memory=True
    )

    # print('Model 1::')
    # model1 = Net(2).to(device)
    # optimizer1 = optim.Adadelta(model1.parameters(), lr=args.lr)
    #
    # scheduler1 = StepLR(optimizer1, step_size=1, gamma=args.gamma)
    # for epoch in range(1, args.epochs + 1):
    #     train(args, model1, device, train_loader, optimizer1, epoch)
    #     test(model1, device, test_loader)
    #     scheduler1.step()

    # print('Model 2::')
    # model2 = Net(4).to(device)
    # optimizer2 = optim.Adadelta(model2.parameters(), lr=args.lr)
    #
    # scheduler2 = StepLR(optimizer2, step_size=1, gamma=args.gamma)
    # for epoch in range(1, args.epochs + 1):
    #     train(args, model2, device, train_loader, optimizer2, epoch)
    #     test(model2, device, test_loader)
    #     scheduler2.step()

    # for i in range(9):
    #     n = int(np.power(2, i))
    #     print('n ==', n)
    #     model = Net(n).to(device)
    #     optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    #
    #     scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    #     for epoch in range(10):
    #         train(args, model, device, train_loader, optimizer, epoch)
    #         test(model, device, test_loader)
    #         scheduler.step()

    print("BigNet::")
    bignet = BigNet(256).to(device)
    ensemblenet = EnsembleNet(bignet).to(device)

    # Pretrain
    optimizer1 = optim.Adadelta(bignet.model1.parameters(), lr=1.0)
    optimizer2 = optim.Adadelta(bignet.model2.parameters(), lr=1.0)
    scheduler1 = StepLR(optimizer1, step_size=1, gamma=args.gamma)
    scheduler2 = StepLR(optimizer2, step_size=1, gamma=args.gamma)
    for epoch in range(10):
        train(args, bignet.model1, device, train_loader, optimizer1, epoch)
        test(bignet.model1, device, test_loader)
        train(args, bignet.model2, device, train_loader, optimizer2, epoch)
        test(bignet.model2, device, test_loader)
        scheduler1.step()
        scheduler2.step()

    for epoch in range(1, 1000):
        print("\n\n\n")
        train_joint_model(args, bignet, device, train_loader, epoch)
        test(bignet.model1, device, test_loader)
        test(bignet.model2, device, test_loader)
        train_ensemble_net(args, ensemblenet, device, train_loader, epoch)
        test(ensemblenet, device, test_loader)


if __name__ == "__main__":
    main()
