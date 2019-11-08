import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


# DEBUG
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))

        x = F.max_pool2d(x, 2, 2)
        #x = F.avg_pool2d(x, 2, 2)

        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        # C1 - 156 params.
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)

        # S2 - 12 params.
        self.s2_1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.s2_2 = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=1, groups=6, bias=True)
        #self.s2 = nn.MaxPool2d(2, 2)

        # C3 - 6 * (75 + 1) + 9 * (100 + 1) + 1 * (150 + 1) = 1516 params.
        self.c3_3_in = nn.ModuleList()
        self.c3_4_in = nn.ModuleList()
        self.c3_6_in = nn.Conv2d(in_channels=6, out_channels=1, kernel_size=5, padding=False)

        for i in range(6):
            self.c3_3_in.append(nn.Conv2d(in_channels=3, out_channels=1, kernel_size=5, padding=False))
        for i in range(9):
            self.c3_4_in.append(nn.Conv2d(in_channels=4, out_channels=1, kernel_size=5, padding=False))

        # DEBUG
        # self.c3 = nn.Conv2d(6, 16, 5, 1)

        # See Table 1 in paper.
        self.s2_ch_3_in = [
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [0, 4, 5],
            [0, 1, 5]
        ]
        self.s2_ch_4_in = [
            [0, 1, 2, 3],
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [0, 3, 4, 5],
            [0, 1, 4, 5],
            [0, 1, 2, 5],
            [0, 1, 3, 4],
            [1, 2, 4, 5],
            [0, 2, 3, 5]
        ]

        # S4 - 32 params.
        self.s4_1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.s4_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, groups=16, bias=True)
        #self.s4 = nn.MaxPool2d(2, 2)

        # C5 - 48120 params.
        self.c5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5, 5), bias=True, padding=(1, 1))

        # F6, F7
        self.f6 = nn.Linear(480, 84)  # for input of 28x28
        self.f7 = nn.Linear(84, 10)

        self.activation = nn.Tanh()
        #self.activation = nn.ReLU()

    def forward(self, im):
        # C1
        x = self.c1(im)
        x = self.activation(x)

        # S2
        x = self.s2_1(x)
        x = self.s2_2(x)
        x = self.activation(x)

        # C3
        c3 = []
        for i in range(6):
            c3.append(self.c3_3_in[i](x[:, self.s2_ch_3_in[i], :, :]))
        for i in range(9):
            c3.append(self.c3_4_in[i](x[:, self.s2_ch_4_in[i], :, :]))
        c3.append(self.c3_6_in(x))
        x = torch.cat(c3, dim=1)
        x = self.activation(x)

        # DEBUG
        #x = self.c3(x)
        #x = self.activation(x)

        # S4
        x = self.s4_1(x)
        x = self.s4_2(x)
        x = self.activation(x)

        # C5
        x = self.c5(x)
        x = self.activation(x)

        # F6, F7
        x = x.view(x.shape[0], -1)
        x = self.f6(x)
        x = self.activation(x)
        x = self.f7(x)

        x = nn.LogSoftmax(dim=1)(x)
        return x


if __name__ == '__main__':
    model = LeNet5()
    print(model)

    for name, p in model.named_parameters():
        print(name, p.size(), p.numel())

    # Test forward method.
    from data.fashion_mnist import load_mnist, FashionMNISTDataset
    import torchvision
    path = '/Users/maorshutman/data/FashionMNIST'
    images, labels = load_mnist(path, kind='train')

    ds = FashionMNISTDataset(images, labels, torchvision.transforms.ToTensor())
    print(len(ds))

    loader = torch.utils.data.DataLoader(ds, batch_size=32)

    for batch in loader:
        X, y = batch
        print(X.shape)
        out = model.forward(X)
        break
