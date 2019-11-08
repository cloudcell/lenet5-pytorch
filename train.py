import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F

from config import cfg, load_from_yaml

from data.fashion_mnist import  load_mnist, FashionMNISTDataset
from lenet5 import LeNet5, Net

from tensorboardX import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser(description='Train config.')
    parser.add_argument(
        '--cfg_path', type=str, required=True, help='Path to YAML config file.')
    args = parser.parse_args()
    return args


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
            tb_writer.add_scalar('train_loss', loss.item(), batch_idx + epoch * len(train_loader))

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, criterion, tb_writer, tb_idx):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).long()

            output = model(data)
            test_loss += criterion(output, target).sum().item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    tb_writer.add_scalar('val_loss', test_loss, tb_idx)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    args = parse_args()
    load_from_yaml(args.cfg_path)
    print(cfg)

    use_cuda = not cfg.DEVICE and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('Available device: ', device)

    train_images, train_labels = load_mnist(cfg.PATHS.DATASET, kind='train')
    test_images, test_labels = load_mnist(cfg.PATHS.DATASET, kind='t10k')

    # TODO: How to normalize ? Should I do it like MNIST ?
    train_dataset = FashionMNISTDataset(train_images[:32], train_labels[:32], torchvision.transforms.ToTensor())
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

    # TODO: How to add the lr schedule in the paper ?
    optimizer = optim.SGD(model.parameters(), lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM)
    #optimizer = optim.Adam(model.parameters(), lr=0.003)

    criterion = nn.NLLLoss()

    # TensorboardX writer.
    tb_writer = SummaryWriter(cfg.TRAIN.LOG_DIR, flush_secs=1)

    for epoch in range(cfg.TRAIN.EPOCHS):
        train(model, device, train_loader, optimizer, criterion, epoch, tb_writer)

        # tb_test_idx = epoch * (len(train_loader) + 1)
        # test(model, device, test_loader, criterion, tb_writer, tb_test_idx)
