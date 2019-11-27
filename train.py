import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from config import cfg, load_from_dict, load_from_yaml

from train_logger import TrainLogger, plot_history
from data.fashion_mnist import load_mnist, FashionMNISTDataset
from lenet5 import LeNet5, Net

import utils
from utils import model_checker, save_checkpoint, load_checkpoint

from tensorboardX import SummaryWriter

import optuna


# --cfg_path=/Users/maorshutman/repos/lenet5-pytorch/cfg_files/cfg.yaml


def parse_args():
    parser = argparse.ArgumentParser(description='Train config.')
    parser.add_argument(
        '--cfg_path', type=str, required=True, help='Path to YAML config file.')
    return parser.parse_args()


def train(model, device, train_loader, optimizer, criterion, epoch, scheduler,
          visualizer=None, tb_writer=None, logger=None):
    model.train()

    train_steps = len(train_loader)
    if cfg.TRAIN.STEPS_PER_EPOCH != -1:
        train_steps = cfg.TRAIN.STEPS_PER_EPOCH

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).long()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % cfg.TRAIN.LOG_INTERVAL == 0:
            tb_idx = batch_idx + epoch * train_steps

            if tb_writer is not None:
                tb_writer.add_scalar('train_loss', loss.item(), tb_idx)
                tb_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], tb_idx)

            if logger is not None:
                logger.add_train(loss, tb_idx)

            num_samples = batch_idx * cfg.TRAIN.BATCH_SIZE
            tot_num_samples = train_steps * cfg.TRAIN.BATCH_SIZE
            completed = 100. * batch_idx / train_steps
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {:.2e}'.format(
                epoch, num_samples, tot_num_samples, completed, loss.item(),
                optimizer.param_groups[0]['lr']))

            # Evaluate on a fixed test batch for visualization.
            if visualizer is not None:
                preds = utils.eval(model, visualizer.batch, device)
                #visualizer.add_preds(torch.exp(preds).detach(), tb_idx)

    # Advance the scheduler at the end of an epoch.
    scheduler.step()

    return train_steps


def test(model, device, test_loader, criterion, batch_size=None, verbose=True):
    model.eval()

    steps = len(test_loader)
    loss = 0
    correct = 0
    targets = []
    preds = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device).long()

            output = model(data)
            loss += criterion(output, target).mean().item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()

            targets += list(target.cpu().numpy())
            preds += list(pred.cpu().numpy())

    loss /= steps
    accuracy = 100. * correct / (steps * batch_size)

    if verbose:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}%\n'.format(loss, accuracy))

    return loss, accuracy, targets, preds


def objective(trial):
    torch.manual_seed(0)
    use_cuda = (not cfg.DEVICE == 'cpu') and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('Available device: ', device)

    # Set up the data set.
    images, labels = load_mnist(cfg.PATHS.DATASET, kind='train')
    train_images, val_images, train_labels, val_labels = train_test_split(
        images, labels, test_size=cfg.TRAIN.VAL_SIZE, random_state=0)

    test_images, test_labels = load_mnist(cfg.PATHS.DATASET, kind='t10k')

    # As all the images are aligned, we use only horizontal flips and small rotations for
    # augmentation.
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor()
    ])

    test_transofrm = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = FashionMNISTDataset(train_images, train_labels, train_transform)
    val_dataset = FashionMNISTDataset(val_images, val_labels, test_transofrm)
    test_dateset = FashionMNISTDataset(test_images, test_labels, test_transofrm)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=1)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=True, num_workers=1)

    # test_loader = torch.utils.data.DataLoader(
    #     test_dateset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=True, num_workers=1)

    model = LeNet5(
        orig_c3=cfg.MODEL.ORIG_C3,
        orig_subsample=cfg.MODEL.ORIG_SUBSAMPLE,
        activation=cfg.MODEL.ACTIVATION,
        dropout=trial.suggest_uniform('dropout', 0.0, 0.2),
        use_bn=cfg.MODEL.BATCHNORM
    )

    model.to(device)
    # Check model dependencies using backprop.
    model_checker(model, train_dataset, device)

    # Load pretrained model if specified.
    if cfg.TRAIN.PRETRAINED_PATH != '':
        load_checkpoint(model, cfg.TRAIN.PRETRAINED_PATH)

    # TODO: Tune.
    lr = trial.suggest_loguniform('lr', 1.0e-4, 1.0)
    #weight_decay = trial.suggest_loguniform('weight decay', 1.0e-6, 0.1)
    #momentum = trial.suggest_uniform('momentum', 0.0, 0.9)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # TODO: Tune.
    gamma = trial.suggest_uniform('gamma', 0.3, 1.0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1.e-4, max_lr=1., step_size_up=300, gamma=1.0)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1.0e-4, last_epoch=-1)

    criterion = nn.NLLLoss()

    # TensorboardX writer.
    tb_writer = SummaryWriter(cfg.TRAIN.LOG_DIR, flush_secs=1)

    # A simple logger is used for the losses.
    logger = TrainLogger()

    # Init a visualizer on a fixed test batch.
    vis = utils.init_vis(test_dateset, cfg.TRAIN.LOG_DIR)

    # Train.
    val_accuracy = 0.0
    for epoch in range(cfg.TRAIN.EPOCHS):
        train_steps = train(model, device, train_loader, optimizer, criterion, epoch,
                            scheduler, visualizer=vis, tb_writer=tb_writer, logger=logger)

        tb_test_idx = (epoch + 1) * train_steps

        val_loss, val_accuracy, targets, preds = test(model, device, val_loader, criterion, cfg.TEST.BATCH_SIZE)

        cm = confusion_matrix(targets, preds)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        if vis is not None:
            vis.add_conf_mat(cm, tb_test_idx)

        # The training set loss will allow us to see the generalization error.
        train_loss, train_accuracy, _, _ = test(model, device, train_loader, criterion, cfg.TEST.BATCH_SIZE)

        tb_writer.add_scalar('val_loss', val_loss, tb_test_idx)
        tb_writer.add_scalar('val_accuracy', val_accuracy, tb_test_idx)
        tb_writer.add_scalar('train_accuracy', train_accuracy, tb_test_idx)

        tb_writer.add_scalar('GE', val_loss - train_loss, tb_test_idx)
        tb_writer.add_scalars('val_train', {'val': val_loss,
                                            'train': train_loss}, tb_test_idx)

        logger.add_val(val_loss, val_accuracy / 100., tb_test_idx)

        # Save checkpoint.
        if cfg.PATHS.CHECKPOINTS_PATH != '':
            save_checkpoint(model, optimizer, epoch, cfg.PATHS.CHECKPOINTS_PATH)

    plot_history(logger, save_path=cfg.TRAIN.LOG_DIR + '/history.png')
    tb_writer.close()

    return val_accuracy


if __name__ == '__main__':
    args = parse_args()

    load_from_yaml(args.cfg_path)
    print(cfg)

    # Hyper-parameter optimization.
    study = optuna.create_study(study_name='lenet5_hp_opt_2',
                                storage='sqlite:///lenet5_hp_opt_2.db', load_if_exists=True)
    study.optimize(objective, n_trials=1)
