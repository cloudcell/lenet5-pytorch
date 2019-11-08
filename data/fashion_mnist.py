import os
import gzip
import numpy as np

from torch.utils import data
import torchvision


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


class FashionMNISTDataset(data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        X = self.images[idx, :].reshape((28, 28, 1))

        if self.transform is not None:
            X = self.transform(X)

        y = self.labels[idx]
        return X, y


if __name__ == '__main__':
    path = '/Users/maorshutman/data/FashionMNIST'
    images, labels = load_mnist(path, kind='train')

    ds = FashionMNISTDataset(images, labels, torchvision.transforms.ToTensor())
    print(len(ds))

    X, y = ds[0]
    import matplotlib.pyplot as plt
    plt.imshow(X.numpy()[0, ::])
    plt.show()
