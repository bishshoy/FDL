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
from vgg_net_x3 import Net, BigNet, EnsembleNet
from utils import load_lr, write_lr, load_momentum, write_momentum, read_params
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--cifar', type=int)

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


def train_joint_model(bignet, ensemblenet, train_loader, test_loader, num_epochs):
    model1, model2, model3 = bignet.model1, bignet.model2, bignet.model3

    optimizer1 = optim.SGD(model1.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4, nesterov=True)
    optimizer2 = optim.SGD(model2.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4, nesterov=True)
    optimizer3 = optim.SGD(model3.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4, nesterov=True)

    accuracy_diff_optimizer = optim.SGD(bignet.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4, nesterov=True)

    if args.cifar == 100:
        similarity_optimizer = optim.SGD(bignet.parameters(), lr=1e-5, momentum=0.85, weight_decay=5e-4, nesterov=True)
    elif args.cifar == 10:
        similarity_optimizer = optim.SGD(bignet.parameters(), lr=1e-4, momentum=0.85, weight_decay=5e-4, nesterov=True)

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
        optimizer3.zero_grad()
        accuracy_diff_optimizer.zero_grad()
        similarity_optimizer.zero_grad()
        ensemblenet_optimizer.zero_grad()

    def schedule_all_optimizers(epoch):
        lr_scheduler(optimizer1, epoch)
        lr_scheduler(optimizer2, epoch)
        lr_scheduler(optimizer3, epoch)
        lr = lr_scheduler(accuracy_diff_optimizer, epoch)
        # lr_scheduler(similarity_optimizer, epoch)
        # lr = lr_scheduler(ensemblenet_optimizer, epoch)
        print("lr:", lr)

    for epoch in range(num_epochs):
        _loss1, _loss2, _loss3, _loss_acc1, _loss_acc2, _loss_acc3, _loss_diff, _loss_ens = (
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
        )

        schedule_all_optimizers(epoch)

        for batch_idx, (data, target) in enumerate(train_loader):
            model1.train(), model2.train(), model3.train(), ensemblenet.train()
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

            # Train Model 3
            zero_grad_all_optimizers()
            output3, features3 = model3(data)
            loss3 = nn.CrossEntropyLoss()(output3, target)
            loss3.backward()
            optimizer3.step()

            rand = np.random.randint(1, 4, 1)[0]

            if rand == 1:
                # Train Model 1-2 difference
                zero_grad_all_optimizers()
                (output1, _), (output2, _), (output3, _) = bignet(data)
                loss1 = nn.CrossEntropyLoss()(output1, target)
                loss2 = nn.CrossEntropyLoss()(output2, target)
                loss_acc1 = nn.MSELoss()(loss1, loss2)
                loss_acc1.backward()
                accuracy_diff_optimizer.step()
                _loss_acc1.append(loss_acc1.item())

            elif rand == 2:
                # Train Model 2-3 difference
                zero_grad_all_optimizers()
                (output1, _), (output2, _), (output3, _) = bignet(data)
                loss2 = nn.CrossEntropyLoss()(output2, target)
                loss3 = nn.CrossEntropyLoss()(output3, target)
                loss_acc2 = nn.MSELoss()(loss2, loss3)
                loss_acc2.backward()
                accuracy_diff_optimizer.step()
                _loss_acc2.append(loss_acc2.item())

            elif rand == 3:
                # Train Model 3-1 difference
                zero_grad_all_optimizers()
                (output1, _), (output2, _), (output3, _) = bignet(data)
                loss3 = nn.CrossEntropyLoss()(output3, target)
                loss1 = nn.CrossEntropyLoss()(output1, target)
                loss_acc3 = nn.MSELoss()(loss3, loss1)
                loss_acc3.backward()
                accuracy_diff_optimizer.step()
                _loss_acc3.append(loss_acc3.item())

            # Train Dissimilarity between features
            zero_grad_all_optimizers()
            (_, features1), (_, features2), (_, features3) = bignet(data)
            loss_diff = torch.zeros(1).cuda()
            for a, b, c in zip(features1, features2, features3):
                loss_diff += (similarityLoss(a, b) + similarityLoss(b, c) + similarityLoss(c, a)) / 3.0
            loss_diff /= len(features1)
            loss_diff *= -1
            loss_diff.backward()
            similarity_optimizer.step()

            # Train EnsembleNet
            zero_grad_all_optimizers()
            output, _ = ensemblenet(data)
            loss_ens = nn.CrossEntropyLoss()(output, target)
            loss_ens.backward()
            ensemblenet_optimizer.step()

            _loss1.append(loss1.item()), _loss2.append(loss2.item()), _loss3.append(loss3.item())
            _loss_diff.append(loss_diff.item()), _loss_ens.append(loss_ens.item())

            if batch_idx % 1 == 0:
                sys.stdout.write(
                    "\rBigNet Epoch: {} [{}/{}]\tLosses: {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        np.mean(_loss1),
                        np.mean(_loss2),
                        np.mean(_loss3),
                        np.mean(_loss_acc1),
                        np.mean(_loss_acc2),
                        np.mean(_loss_acc3),
                        np.mean(_loss_diff),
                        np.mean(_loss_ens),
                    )
                )
                sys.stdout.flush()
        print("\n")
        _, _ = test(model1, test_loader)
        _, _ = test(model2, test_loader)
        _, _ = test(model3, test_loader)
        ens_acc, _ = test(ensemblenet, test_loader)
        if ens_acc > max_accuracy:
            max_accuracy = ens_acc
        print("Max accuracy:", max_accuracy)


def train(model, train_loader, test_loader, num_epochs):
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
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


def main(cifar):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    root = "datasets"
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
            download=True,
        ),
        batch_size=128,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
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
        batch_size=1024,
        shuffle=False,
        num_workers=12,
        pin_memory=True,
    )

    # print('Model 1::')
    # model1 = Net()
    # train(model1, train_loader, test_loader, num_epochs=250)

    bignet = BigNet(cifar)
    ensemblenet = EnsembleNet(bignet, cifar)

    print("Pretrain Model1::")
    train(bignet.model1, train_loader, test_loader, 1)
    print("Pretrain Model2::")
    train(bignet.model2, train_loader, test_loader, 1)
    print("Pretrain Model3::")
    train(bignet.model3, train_loader, test_loader, 1)

    print("Train BigNet::")
    train_joint_model(bignet, ensemblenet, train_loader, test_loader, 10000)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main(cifar=args.cifar)
