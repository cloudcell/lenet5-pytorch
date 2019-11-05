import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        # C1 - 156 params.
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=False)

        # S2 - 12 params.
        # TODO: Put in function.
        self.s2_1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.s2_2 = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=1, groups=6, bias=True)

        # C3 - 6 * (75 + 1) + 9 * (100 + 1) + 1 * (150 + 1) = 1516 params.
        self.c3_3_in = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=5, padding=False)  # x 6
        self.c3_4_in = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=5, padding=False)  # x 9
        self.c3_6_in = nn.Conv2d(in_channels=6, out_channels=1, kernel_size=5, padding=False)  # x 1

    def forward(self, x):
        x = self.c1(x)

        x = self.s1_1(x)
        x = self.s1_2(x)
        x = F.sigmoid(x)
        
        return x


if __name__ == '__main__':
    model = LeNet5()
    print(model)

    for name, p in model.named_parameters():
        print(name, p.size(), p.numel())
