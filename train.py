import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from config import cfg, load_from_yaml

from train_logger import TrainLogger, plot_history
from data.fashion_mnist import load_mnist, FashionMNISTDataset
from lenet5 import LeNet5

from utils import model_checker

from tensorboardX import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser(description='Train config.')
    parser.add_argument(
        '--cfg_path', type=str, required=True, help='Path to YAML config file.')
    args = parser.parse_args()
    return args


def train(model, device, train_loader, optimizer, criterion, epoch, tb_writer, scheduler, logger):
    model.train()

    train_steps = len(train_loader)
    if cfg.TRAIN.STEPS_PER_EPOCH != -1:
        train_steps = cfg.TRAIN.STEPS_PER_EPOCH

    for batch_idx in range(train_steps):
        data, target = next(iter(train_loader))
        data, target = data.to(device), target.to(device).long()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % cfg.TRAIN.LOG_INTERVAL == 0:
            tb_idx = batch_idx + epoch * train_steps
            tb_writer.add_scalar('train_loss', loss.item(), tb_idx)
            tb_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], tb_idx)
            logger.add_train(loss, tb_idx)

            num_samples = batch_idx * cfg.TRAIN.BATCH_SIZE
            tot_num_samples = train_steps * cfg.TRAIN.BATCH_SIZE
            completed = 100. * batch_idx / train_steps
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {:.2e}'.format(
                epoch, num_samples, tot_num_samples, completed, loss.item(),
                optimizer.param_groups[0]['lr']))

        scheduler.step()

    return train_steps


def test(model, device, test_loader, criterion, tb_writer, tb_idx, logger):
    model.eval()

    val_steps = len(test_loader)
    if cfg.TEST.STEPS != -1:
        val_steps = cfg.TEST.STEPS

    val_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx in range(val_steps):
            data, target = next(iter(test_loader))
            data, target = data.to(device), target.to(device).long()

            output = model(data)
            val_loss += criterion(output, target).mean().item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= val_steps
    accuracy = 100. * correct / (val_steps * cfg.TEST.BATCH_SIZE)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}%\n'.format(val_loss, accuracy))

    tb_writer.add_scalar('val_loss', val_loss, tb_idx)
    tb_writer.add_scalar('val_accuracy', accuracy, tb_idx)
    logger.add_val(val_loss, accuracy / 100., tb_idx)


def main():
    args = parse_args()
    load_from_yaml(args.cfg_path)
    print(cfg)

    torch.manual_seed(0)

    use_cuda = (not cfg.DEVICE == 'cpu') and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('Available device: ', device)

    train_images, train_labels = load_mnist(cfg.PATHS.DATASET, kind='train')
    test_images, test_labels = load_mnist(cfg.PATHS.DATASET, kind='t10k')

    transform = torchvision.transforms.ToTensor()
    train_dataset = FashionMNISTDataset(train_images, train_labels, transform)
    test_dateset = FashionMNISTDataset(test_images, test_labels, transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=1)

    test_loader = torch.utils.data.DataLoader(
        test_dateset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=True, num_workers=1)

    model = LeNet5(
        orig_c3=cfg.MODEL.ORIG_C3,
        orig_subsample=cfg.MODEL.ORIG_SUBSAMPLE,
        activation=cfg.MODEL.ACTIVATION,
        dropout=cfg.MODEL.DROPOUT,
        use_bn=cfg.MODEL.BATCHNORM
    )
    model.to(device)
    model_checker(model, train_dataset, device)

    optimizer = optim.SGD(model.parameters(),
                          lr=cfg.TRAIN.LR,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1.e-4, max_lr=1., step_size_up=300, gamma=0.5)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1.0e-4, last_epoch=-1)

    criterion = nn.NLLLoss()

    # TensorboardX writer.
    tb_writer = SummaryWriter(cfg.TRAIN.LOG_DIR, flush_secs=1)

    # A simple logger is used for the losses.
    logger = TrainLogger()

    for epoch in range(cfg.TRAIN.EPOCHS):
        train_steps = train(model, device, train_loader, optimizer, criterion, epoch, tb_writer,
                            scheduler, logger)
        tb_test_idx = (epoch + 1) * train_steps
        test(model, device, test_loader, criterion, tb_writer, tb_test_idx, logger)

        # Save checkpoint.
        if cfg.PATHS.CHECKPOINTS_PATH != '':
            checkpoint_path = os.path.join(
                cfg.PATHS.CHECKPOINTS_PATH, 'lenet5_epoch_' + str(epoch) + '.pt')
            torch.save(model.state_dict(), checkpoint_path)

    plot_history(logger, save_path=cfg.TRAIN.LOG_DIR + '/history.png')
    tb_writer.close()


if __name__ == '__main__':
    main()
