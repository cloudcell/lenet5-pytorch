import torch
import torch.nn as nn


class LeNet5(nn.Module):
    """This implementation follows closely the paper:
    "Gradient-Based Learning Applied to Document Recognition", by LeCun et al.
    The network is adapted to work with images of size 28 x 28.
    """
    def __init__(
            self,
            c3_non_comp_conn=True,
            activation='tanh',
            dropout=0.0):
        super(LeNet5, self).__init__()
        # This parameter allows to choose the implementation in the paper, or
        # a simple convolution for layer C3.
        self.c3_non_comp_conn = c3_non_comp_conn

        self.dropout = dropout
        self.activation_type = activation

        # C1 - 156 params.
        self.c1 = nn.Conv2d(1, 6, 5)

        # S2 - 12 params.
        self.s2_1 = nn.AvgPool2d(2, 2)
        self.s2_2 = nn.Conv2d(6, 6, 1, groups=6, bias=True)

        if self.c3_non_comp_conn:
            self.c3 = OriginalC3Layer()
        else:
            self.c3 = nn.Conv2d(6, 16, 5, 1)

        # S4 - 32 params.
        self.s4_1 = nn.AvgPool2d(2, 2)
        self.s4_2 = nn.Conv2d(16, 16, 1, groups=16, bias=True)

        # C5 - 48120 params.
        self.c5 = nn.Conv2d(16, 120, 5, bias=True, padding=(1, 1))

        # F6, F7
        self.f6 = nn.Linear(480, 84)  # 480 is for input of size 28x28
        self.f7 = nn.Linear(84, 10)

        self.drop_layer = nn.Dropout(self.dropout)

        if self.activation_type == 'tanh':
            self.activation = nn.Tanh()
        elif self.activation_type == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError('Unsupported activation.')

    def forward(self, im):
        # C1
        x = self.c1(im)
        x = self.activation(x)

        # S2
        x = self.s2_1(x)
        x = self.s2_2(x)
        x = self.activation(x)

        # C3
        x = self.c3(x)
        x = self.activation(x)

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
        x = self.drop_layer(x)
        x = self.f7(x)

        x = nn.LogSoftmax(dim=1)(x)
        return x


class OriginalC3Layer(nn.Module):
    """The original C3 conv. layer as described in "Gradient-Based Learning Applied
    to Document Recognition", by LeCun et al.
    """
    def __init__(self):
        super(OriginalC3Layer, self).__init__()
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

        # C3 - 6 * (75 + 1) + 9 * (100 + 1) + 1 * (150 + 1) = 1516 params.
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
