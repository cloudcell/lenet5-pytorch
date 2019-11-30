import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from config import cfg, load_from_yaml

from train_logger import TrainLogger, plot_history
from data.fashion_mnist import load_mnist, FashionMNISTDataset
from lenet5 import LeNet5

import utils
from utils import model_checker, save_checkpoint, load_checkpoint
from test import test

from tensorboardX import SummaryWriter

import optuna


# --cfg_path=./cfg_files/cfg.yaml
# --hp_optim=False


def bool_string(input_string):
    if input_string not in {"True","False"}:
        raise ValueError("Please Enter a valid Ture/False choice")
    else:
        return input_string == "True"


def parse_args():
    parser = argparse.ArgumentParser(description='Train config.')
    parser.add_argument(
        '--cfg_path', type=str, required=True, help='Path to YAML config file.')
    parser.add_argument(
        '--hp_optim', type=bool_string, required=True, help='If True, perform a hyper-parameter'
                                                            ' optimization using optuna.')
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


def build_dataloaders():
    # Set up the data set.
    images, labels = load_mnist(cfg.PATHS.DATASET, kind='train')
    train_images, val_images, train_labels, val_labels = train_test_split(
        images, labels, test_size=cfg.TRAIN.VAL_SIZE, random_state=0)

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        utils.RandomEraseCrop(p=0.1, size=4),
        utils.RandomGaussNoise(p=0.1, sigma=0.05),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_transofrm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = FashionMNISTDataset(train_images, train_labels, train_transform)
    val_dataset = FashionMNISTDataset(val_images, val_labels, test_transofrm)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=1)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=True, num_workers=1)

    return train_loader, val_loader, train_dataset, val_dataset


def build_model(device):
    model = LeNet5(
        orig_c3=cfg.MODEL.ORIG_C3,
        orig_subsample=cfg.MODEL.ORIG_SUBSAMPLE,
        activation=cfg.MODEL.ACTIVATION,
        dropout=cfg.MODEL.DROPOUT,
        use_bn=cfg.MODEL.BATCHNORM
    )

    model.to(device)
    # Check model dependencies using backprop.
    model_checker(model, train_dataset, device)

    # Load pretrained model if specified.
    if cfg.TRAIN.PRETRAINED_PATH != '':
        load_checkpoint(model, cfg.TRAIN.PRETRAINED_PATH)

    return model


def train_model(model, train_loader, val_loader, train_dataset, optimizer, criterion, scheduler,
                device, cfg, visualize=True):
    # TensorboardX writer.
    tb_writer = SummaryWriter(cfg.TRAIN.LOG_DIR, flush_secs=1)

    # A simple logger is used for the losses.
    logger = TrainLogger()

    # Init a visualizer on a fixed test batch.
    vis = None
    if visualize:
        vis = utils.init_vis(train_dataset, cfg.TRAIN.LOG_DIR)

    # Train.
    val_max_acc = -1.
    for epoch in range(cfg.TRAIN.EPOCHS):
        train_steps = train(model, device, train_loader, optimizer, criterion, epoch,
                            scheduler, visualizer=vis, tb_writer=tb_writer, logger=logger)

        tb_test_idx = (epoch + 1) * train_steps

        val_loss, val_accuracy, targets, preds = test(model, device, val_loader, criterion, cfg.TEST.BATCH_SIZE)

        cm = confusion_matrix(targets, preds)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        if vis is not None:
            vis.add_conf_mat(cm, tb_test_idx)

        tb_writer.add_scalar('val_loss', val_loss, tb_test_idx)
        tb_writer.add_scalar('val_accuracy', val_accuracy, tb_test_idx)
        logger.add_val(val_loss, val_accuracy / 100., tb_test_idx)

        # Save checkpoint.
        if cfg.PATHS.CHECKPOINTS_PATH != '':
            save_checkpoint(model, optimizer, epoch, cfg.PATHS.CHECKPOINTS_PATH)

        if val_accuracy >= val_max_acc:
            val_max_acc = val_accuracy

    plot_history(logger, save_path=cfg.TRAIN.LOG_DIR + '/history.png')
    tb_writer.close()

    return val_max_acc


def train_with_cfg(train_loader, val_loader, train_dataset):
    """Train with the parametrs defined in the YACS global config `cfg`."""
    use_cuda = (not cfg.DEVICE == 'cpu') and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('Available device: ', device)

    model = build_model(device)
    # Save the model with architecture.
    torch.save(model, cfg.PATHS.CHECKPOINTS_PATH + '/model.pt')

    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.TRAIN.LR,
        momentum=cfg.TRAIN.MOMENTUM,
        weight_decay=cfg.TRAIN.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.TRAIN.GAMMA)

    criterion = nn.NLLLoss()

    val_max_acc = train_model(model, train_loader, val_loader, train_dataset, optimizer, criterion,
                              scheduler, device, cfg, visualize=True)

    return val_max_acc


def objective(trial, train_loader, val_loader, train_dataset):
    """An objective function for the optuna optimizer."""
    # Randomly chose a subset of the hyper-parameters.
    cfg.MODEL.DROPOUT = trial.suggest_uniform('dropout', 0.0, 0.5)
    cfg.TRAIN.LR = trial.suggest_loguniform('lr', 0.1, 1.0)
    cfg.TRAIN.WEIGHT_DECAY = trial.suggest_loguniform('weight decay', 1.0e-6, 1.e-5)
    cfg.TRAIN.GAMMA = trial.suggest_uniform('gamma', 0.6, 1.0)

    val_max_acc = train_with_cfg(train_loader, val_loader, train_dataset)
    return val_max_acc


if __name__ == '__main__':
    args = parse_args()

    # Load configuration into a global YACS objects.
    load_from_yaml(args.cfg_path)
    print(cfg)

    torch.manual_seed(0)

    train_loader, val_loader, train_dataset, val_dataset = build_dataloaders()

    if args.hp_optim:
        # Hyper-parameter optimization.
        study = optuna.create_study(
            study_name='lenet5_hp_opt',
            pruner=optuna.pruners.MedianPruner(),
            storage='sqlite:///lenet5_hp_opt_1.db',
            load_if_exists=True
        )
        obj = lambda trial: objective(trial, train_loader, val_loader, train_dataset)
        study.optimize(obj, n_trials=100)
    else:
        # Train with the config specified in `cfg`.
        train_with_cfg(train_loader, val_loader, train_dataset)
