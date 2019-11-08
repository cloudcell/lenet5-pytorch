import torch
import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        # C1 - 156 params.
        self.c1 = nn.Conv2d(1, 6, 5)

        # S2 - 12 params.
        self.s2_1 = nn.AvgPool2d(2, 2)
        self.s2_2 = nn.Conv2d(6, 6, 1, groups=6, bias=True)

        # C3 - 6 * (75 + 1) + 9 * (100 + 1) + 1 * (150 + 1) = 1516 params.
        self.c3_3_in = nn.ModuleList()
        self.c3_4_in = nn.ModuleList()
        self.c3_6_in = nn.Conv2d(6, 1, 5, padding=False)

        for i in range(6):
            self.c3_3_in.append(nn.Conv2d(3, 1, 5, padding=False))
        for i in range(9):
            self.c3_4_in.append(nn.Conv2d(4, 1, 5, padding=False))

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
        self.s4_1 = nn.AvgPool2d(2, 2)
        self.s4_2 = nn.Conv2d(16, 16, 1, groups=16, bias=True)

        # C5 - 48120 params.
        self.c5 = nn.Conv2d(16, 120, 5, bias=True, padding=(1, 1))

        # F6, F7
        self.f6 = nn.Linear(480, 84)  # for input of size 28x28
        self.f7 = nn.Linear(84, 10)

        self.activation = nn.Tanh()

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
