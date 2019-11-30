import argparse

import torch
import torch.nn as nn
from torchvision import transforms

from config import cfg, load_from_yaml
from data.fashion_mnist import load_mnist, FashionMNISTDataset
from utils import load_checkpoint


# --checkpoint_path=./checkpoints/checkpoint_10.pt
# --model_path=./checkpoints/model.pt
# --cfg_path=/home/maors/repos/lenet5-pytorch/cfg_files/cfg.yaml


def parse_args():
    parser = argparse.ArgumentParser(description='Test model test set.')
    parser.add_argument(
        '--model_path', type=str, required=True, help='Path to model .pt file.')
    parser.add_argument(
        '--checkpoint_path', type=str, required=True, help='Path to checkpoint .pt file.')
    parser.add_argument(
        '--cfg_path', type=str, required=True, help='Path to YAML config file.')

    return parser.parse_args()


def test(model, device, test_loader, criterion, batch_size, verbose=True):
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


def main():
    args = parse_args()

    # Load configuration into a global YACS objects.
    load_from_yaml(args.cfg_path)
    print(cfg)

    use_cuda = (not cfg.DEVICE == 'cpu') and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('Available device: ', device)

    model = torch.load(args.model_path)
    load_checkpoint(model, args.checkpoint_path)

    test_images, test_labels = load_mnist(cfg.PATHS.DATASET, kind='t10k')

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_dataset = FashionMNISTDataset(test_images, test_labels, test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=True, num_workers=1)

    criterion = nn.NLLLoss()
    test(model, device, test_loader, criterion, batch_size=cfg.TEST.BATCH_SIZE, verbose=True)


if __name__ == '__main__':
    main()
