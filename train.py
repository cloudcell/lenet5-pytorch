import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from config import cfg

from data.fashion_mnist import  load_mnist, FashionMNISTDataset
from lenet5 import LeNet5

from tensorboardX import SummaryWriter


def train(model, device, train_loader, optimizer, criterion, epoch, tb_writer):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).long()

        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % cfg.TRAIN.LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

        # TODO: how to set the idx ?
        tb_writer.add_scalar('loss', loss.item(), batch_idx)

        # TODO: Add precision function like in the udacity course.


def test(model, device, test_loader, tb_writer):
    model.eval()
    # TODO


if __name__ == '__main__':
    # TODO: Log the config form a file.
    print(cfg)

    use_cuda = not cfg.DEVICE and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('Device: ', device)

    train_images, train_labels = load_mnist(cfg.PATHS.DATA, kind='train')
    test_images, test_labels = load_mnist(cfg.PATHS.DATA, kind='t10k')

    # TODO: How to normalize ? Should I do it like MNIST ?
    train_dataset = FashionMNISTDataset(train_images, train_labels, torchvision.transforms.ToTensor())
    test_dateset = FashionMNISTDataset(test_images, test_labels, torchvision.transforms.ToTensor())

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        test_dateset,
        batch_size=cfg.TEST.BATCH_SIZE, shuffle=True, **kwargs
    )

    # TODO: Start with SGD and then try other methods.
    model = LeNet5().to(device)
    optimizer = optim.SGD(model.parameters(), lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM)
    # optimizer = optim.Adam(model.parameters(), lr=0.003)

    # TODO: How to add the lr schedule in the paper ?
    #criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()

    # TensorboardX writer.
    tb_writer = SummaryWriter(cfg.TRAIN.LOG_DIR)

    for epoch in range(cfg.TRAIN.EPOCHS):
        train(model, device, train_loader, optimizer, criterion, epoch, tb_writer)
        test(model, device, test_loader, tb_writer)
