# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license described in the README.md
# file located in the root directory of this source tree.
#
# Modifications have made by Song-Rong, Lee
#

import torch.nn as nn

from .share import DilatedResConv


class Encoder(nn.Module):
    def __init__(self, n_blocks=3, n_layers=10, hidden_channels=128,
                 latent_d=64, pool_kernel_size=800):
        super(Encoder, self).__init__()

        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.hidden_channels = hidden_channels
        self.latent_d = latent_d
        self.pool_kernel_size = pool_kernel_size

        layers = []
        for _ in range(self.n_blocks):
            for i in range(self.n_layers):
                dilation = 2 ** i
                layers.append(DilatedResConv(self.hidden_channels, dilation))
        self.dilated_convs = nn.Sequential(*layers)

        self.start = nn.Conv1d(1, self.hidden_channels, kernel_size=3, stride=1, padding=1)
        self.conv_1x1 = nn.Conv1d(self.hidden_channels, self.latent_d, kernel_size=1)
        self.pool = nn.AvgPool1d(self.pool_kernel_size)

    def forward(self, x):
        out = x.unsqueeze(1)
        out = out / 255.0 - 0.5
        # print(out.shape)
        # [2, 1, 16000]

        out = self.start(out)
        # print(out.shape)
        # [2, 128, 16000]

        out = self.dilated_convs(out)
        # print(out.shape)
        # [2, 128, 16000]

        out = self.conv_1x1(out)
        # print(out.shape)
        # [2, 64, 16000]

        out = self.pool(out)
        # print(out.shape)
        # [2, 64, 20]

        return out
