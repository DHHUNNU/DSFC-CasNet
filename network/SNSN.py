# -----#
# author:HD
# year&month&day:2024:12:22
# -----#
import torch
import torch.nn as nn

class FCN(nn.Module):
    def __init__(self, L, S):
        super(FCN, self).__init__()
        self.conv_layers = nn.ModuleList()

        # 假设输入长度为2048
        input_length = 2048

        for i in range(L):
            input_channels = 1 if i == 0 else 2 ** i
            output_channels = 2 ** (i + 1) if i < L - 1 else S

            conv_layer = nn.Conv1d(input_channels, output_channels, kernel_size=3, stride=2, padding=1)

            batch_norm = nn.BatchNorm1d(output_channels)

            self.conv_layers.append(conv_layer)
            self.conv_layers.append(batch_norm)
            self.conv_layers.append(nn.ReLU(inplace=True))

            input_length = (input_length + 2 * 1 - 3) // 2 + 1

        self.output_length = input_length

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        return x

class SimAM_module(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(SimAM_module, self).__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        batch = x.size()[0]
        channel = x.size()[1]
        x = x.view(batch, channel, 1, -1)
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activation(y)


class SNSN(nn.Module):
    def __init__(self, L, S):
        super(SNSN, self).__init__()
        self.fcn = FCN(L, S)
        self.simam = SimAM_module()
        self.lin = nn.Linear(S * self.fcn.output_length, 2048)

    def forward(self, x, l):
        x = x.float()
        l = l.float()
        batch_size = x.size()[0]
        x = self.fcn(x)
        x = self.simam(x)
        x = x.view(batch_size, -1)
        x = self.lin(x)
        x = x * l
        return x