#!/usr/bin/env python
# *-* encoding: utf-8 *-*
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, MNIST, CIFAR100, CelebA
import numpy as np
from random import sample
from typing import Dict, List, Tuple, Union, Any
import flwr as fl

# Workaround while Pull Request #1115 is not merged into flower:
# https://github.com/adap/flower/pull/1115
import workaround

# Disallow TF32 usage to avoid numerical errors
if torch.cuda.is_available():
    try:
        import torch.backends.cuda
        import torch.backends.cudnn
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except:
        pass

import logging
logger = logging.getLogger(__name__)

DEVICE = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else "cpu")

DATASETS = (MNIST, CIFAR10, CIFAR100, CelebA)


def load_data(trainset, batchsize, batches):
    indices = sample(range(len(trainset)), k=batches * batchsize)
    trainset = Subset(trainset, indices=indices)
    trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True)
    return trainloader


def train(net, trainloader, epochs, optimizer):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer_c = getattr(torch.optim, optimizer)
    if optimizer_c is torch.optim.SGD:
        kwargs = {"lr": 0.001, "momentum": 0.9}
    else:
        kwargs = {}
    optimizer = optimizer_c(net.parameters(), **kwargs)
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = net(images)
            #logger.debug(f'output: {out.detach().cpu().numpy()}')
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()


def test(net, testloader):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy


class Net(nn.Module):
    def __init__(self, dataset, hidden_layers=2) -> None:
        super(Net, self).__init__()
        if dataset is MNIST:
            in_channels = 1
            in_x = 28
            in_y = in_x
            n_out = 10
        elif dataset in (CIFAR10,):
            in_channels = 3
            in_x = 32
            in_y = in_x
            n_out = 10
        elif dataset in (CIFAR100,):
            in_channels = 3
            in_x = 32
            in_y = in_x
            n_out = 100
        elif dataset in (CelebA,):
            in_channels = 3
            in_x = 178
            in_y = 218
            n_out = 10178
        else:
            assert False, 'Invalid dataset'
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.x = in_x // 4 - 3
        self.y = in_y // 4 - 3
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(16 * self.x*self.y, 120))
        for _ in range(hidden_layers-2):
            self.fcs.append(nn.Linear(120, 120))
        self.fcs.append(nn.Linear(120, 84))
        self.fc_final = nn.Linear(84, n_out)

    def intermediate(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * self.x * self.y)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.intermediate(x)
        for layer in self.fcs:
            x = F.relu(layer(x))
        x = self.fc_final(x)
        return x

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.load_state_dict(state_dict, strict=True)


class MyClient(fl.client.NumPyClient):
    def __init__(self, trainloader, dataset, optimizer, epochs):
        super(MyClient, self).__init__()
        self.net = Net(dataset=dataset)
        self.net = self.net.to(device=DEVICE)
        self.trainloader = trainloader
        self.optimizer = optimizer
        self.epochs = epochs

    def get_parameters(self):
        return self.net.get_parameters()

    def set_parameters(self, parameters):
        self.net.set_parameters(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.net, self.trainloader, optimizer=self.optimizer, epochs=self.epochs)
        return self.get_parameters(), len(self.trainloader), {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, Any]
    ) -> Union[
        Tuple[float, int, Dict[str, Any]],
        Tuple[int, float, float],  # Deprecated
        Tuple[int, float, float, Dict[str, Any]],  # Deprecated
    ]:
        return np.nan, 0, {}

    # Workaround for Bug #1113 of flower:
    # https://github.com/adap/flower/issues/1113
    def get_properties(self, config):
        return {}


def main(client: MyClient, port=8080):
    logger.debug(f'Client connecting to port {port!s}')
    fl.client.start_numpy_client(f"[::]:{port!s}", client=client)
