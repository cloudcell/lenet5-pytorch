import torch
import torch.nn as nn


class LeNet5(nn.Module):
    """This implementation follows closely the paper:
    "Gradient-Based Learning Applied to Document Recognition", by LeCun et al.
    The network is adapted to work with images of size 28 x 28.
    """
    def __init__(self, orig_c3=True, orig_subsample=True, activation='tanh', dropout=0.0,
                 use_bn=True):
        super(LeNet5, self).__init__()
        self.orig_c3 = orig_c3
        self.orig_subsample = orig_subsample
        self.dropout = dropout
        self.activation_type = activation
        self.use_bn = use_bn

        # C1
        # We pad the image to get an input size of 32x32 as for the original network.
        self.c1 = nn.Conv2d(1, 6, 5, padding=(2,2))
        self.bn1 = nn.BatchNorm2d(6)

        # S2
        if self.orig_subsample:
            self.s2 = OrigSubSampler(6)
        else:
            self.s2 = nn.MaxPool2d(2, 2)

        if self.orig_c3:
            self.c3 = OrigC3Layer()
        else:
            self.c3 = nn.Conv2d(6, 16, 5, 1)
        self.bn3 = nn.BatchNorm2d(16)

        # S4
        if self.orig_subsample:
            self.s4 = OrigSubSampler(16)
        else:
            self.s4 = nn.MaxPool2d(2, 2)

        # C5
        self.c5 = nn.Conv2d(16, 120, 5, bias=True)
        self.bn5 = nn.BatchNorm2d(120)

        # F6
        self.f6 = nn.Linear(120, 84)
        self.bn6 = nn.BatchNorm1d(84)

        # F7
        self.f7 = nn.Linear(84, 10)
        self.bn7 = nn.BatchNorm1d(10)

        self.drop_layer = nn.Dropout(self.dropout)

        if self.activation_type == 'tanh':
            self.activation = nn.Tanh()
        elif self.activation_type == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif self.activation_type == 'relu':
            self.activation = nn.ReLU()
        elif self.activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError('Unsupported activation.')

    def forward(self, x):
        # C1
        x = self.c1(x)
        x = self.bn1(x)
        x = self.activation(x)

        # S2
        x = self.s2(x)

        # C3
        x = self.c3(x)
        x = self.bn3(x)
        x = self.activation(x)

        # S4
        x = self.s4(x)

        # C5
        x = self.c5(x)
        x = self.bn5(x)
        x = self.activation(x)

        # F6, F7
        x = x.view(x.shape[0], -1)

        x = self.f6(x)
        x = self.bn6(x)
        x = self.activation(x)
        x = self.drop_layer(x)

        x = self.f7(x)
        x = self.bn7(x)

        x = nn.LogSoftmax(dim=1)(x)
        return x


class OrigC3Layer(nn.Module):
    """The original C3 conv. layer as described in "Gradient-Based Learning Applied
    to Document Recognition", by LeCun et al.
    """
    def __init__(self):
        super(OrigC3Layer, self).__init__()
        # The connections are shown in Table 1 in the paper.
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

        # The number of parameters is 6 * (75 + 1) + 9 * (100 + 1) + 1 * (150 + 1) = 1516.
        self.c3_3_in = nn.ModuleList()
        self.c3_4_in = nn.ModuleList()
        self.c3_6_in = nn.Conv2d(6, 1, 5, padding=False)

        for i in range(6):
            self.c3_3_in.append(nn.Conv2d(3, 1, 5, padding=False))
        for i in range(9):
            self.c3_4_in.append(nn.Conv2d(4, 1, 5, padding=False))

    def forward(self, x):
        c3 = []
        for i in range(6):
            c3.append(self.c3_3_in[i](x[:, self.s2_ch_3_in[i], :, :]))
        for i in range(9):
            c3.append(self.c3_4_in[i](x[:, self.s2_ch_4_in[i], :, :]))
        c3.append(self.c3_6_in(x))
        x = torch.cat(c3, dim=1)
        return x


class OrigSubSampler(nn.Module):
    def __init__(self, in_channels=None):
        super(OrigSubSampler, self).__init__()
        self.in_ch = in_channels

        # The number of parameters is 2 * in_channels.
        self.s4_1 = nn.AvgPool2d(2, 2)
        self.s4_2 = nn.Conv2d(self.in_ch, self.in_ch, 1, groups=self.in_ch, bias=True)

    def forward(self, x):
        x = self.s4_1(x)
        x = self.s4_2(x)
        return x
