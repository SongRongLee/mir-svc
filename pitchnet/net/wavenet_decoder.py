# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license described in the README.md
# file located in the root directory of this source tree.
#
# Modifications have made by Song-Rong, Lee
#

import torch
import torch.nn as nn
import torch.nn.functional as F

from .share import CausalConv1d


class WavenetLayer(nn.Module):
    def __init__(self, residual_channels, skip_channels, cond_channels, kernel_size=2, dilation=1):
        super(WavenetLayer, self).__init__()

        self.causal = CausalConv1d(residual_channels, 2 * residual_channels, kernel_size, dilation=dilation, bias=True)
        self.condition = nn.Conv1d(cond_channels, 2 * residual_channels, kernel_size=1, bias=True)
        self.residual = nn.Conv1d(residual_channels, residual_channels, kernel_size=1, bias=True)
        self.skip = nn.Conv1d(residual_channels, skip_channels, kernel_size=1, bias=True)

    def _condition(self, x, c, f):
        c = f(c)
        x = x + c
        return x

    def forward(self, x, c=None):
        x = self.causal(x)
        if c is not None:
            x = self._condition(x, c, self.condition)

        gate, output = x.chunk(2, 1)
        gate = torch.sigmoid(gate)
        output = torch.tanh(output)
        x = gate * output

        residual = self.residual(x)
        skip = self.skip(x)

        return residual, skip


class WaveNet(nn.Module):
    def __init__(self, blocks=4, layers=10, kernel_size=2, skip_channels=128,
                 residual_channels=128, latent_d=129, shift_input=True):
        super(WaveNet, self).__init__()

        self.blocks = blocks
        self.layer_num = layers
        self.kernel_size = kernel_size
        self.skip_channels = skip_channels
        self.residual_channels = residual_channels
        self.cond_channels = latent_d
        self.classes = 256
        self.shift_input = shift_input

        layers = []
        for _ in range(self.blocks):
            for i in range(self.layer_num):
                dilation = 2 ** i
                layers.append(WavenetLayer(self.residual_channels, self.skip_channels, self.cond_channels,
                                           self.kernel_size, dilation))
        self.layers = nn.ModuleList(layers)

        self.first_conv = CausalConv1d(1, self.residual_channels, kernel_size=self.kernel_size)
        self.skip_conv = nn.Conv1d(self.residual_channels, self.skip_channels, kernel_size=1)
        self.condition = nn.Conv1d(self.cond_channels, self.skip_channels, kernel_size=1)
        self.fc = nn.Conv1d(self.skip_channels, self.skip_channels, kernel_size=1)
        self.logits = nn.Conv1d(self.skip_channels, self.classes, kernel_size=1)

    def _condition(self, x, c, f):
        c = f(c)
        x = x + c
        return x

    def shift_right(self, x):
        x = F.pad(x, (1, 0))
        return x[:, :, :-1]

    def forward(self, x, c=None):
        x = x.unsqueeze(1)
        x = x / 255.0 - 0.5

        if self.shift_input:
            x = self.shift_right(x)
        # print(x.shape, c.shape)
        # [2, 1, 16000], [2, 129, 16000]

        residual = self.first_conv(x)
        skip = self.skip_conv(residual)
        # print(residual.shape, skip.shape)
        # [2, 128, 16000], [2, 128, 16000]

        for layer in self.layers:
            r, s = layer(residual, c)
            residual = residual + r
            skip = skip + s

        skip = F.relu(skip)
        skip = self.fc(skip)
        # print(skip.shape)
        # [2, 128, 16000]

        if c is not None:
            skip = self._condition(skip, c, self.condition)
        skip = F.relu(skip)
        skip = self.logits(skip)
        # print(skip.shape)
        # [2, 256, 16000]

        return skip
