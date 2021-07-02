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
from collections import deque


class DilatedResConv(nn.Module):
    def __init__(self, channels, dilation=1, padding=1, kernel_size=3):
        super(DilatedResConv, self).__init__()
        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU(inplace=True)
        self.dilated_conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, stride=1,
                                      padding=dilation*padding, dilation=dilation, bias=True)
        self.conv_1x1 = nn.Conv1d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x):
        out = x
        out = self.relu_1(out)
        out = self.dilated_conv(out)
        out = self.relu_2(out)
        out = self.conv_1x1(out)

        return x + out


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size=2, dilation=1, **kwargs):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            padding=dilation*(kernel_size-1),
            dilation=dilation,
            **kwargs)

    def forward(self, input):
        out = super(CausalConv1d, self).forward(input)
        return out[:, :, :-self.padding[0]]


class QueuedConv1d(nn.Module):
    def __init__(self, conv, data):
        super(QueuedConv1d, self).__init__()
        if isinstance(conv, nn.Conv1d):
            self.inner_conv = nn.Conv1d(conv.in_channels, conv.out_channels, conv.kernel_size)
            self.init_len = conv.padding[0]
            self.inner_conv.weight.data.copy_(conv.weight.data)
            self.inner_conv.bias.data.copy_(conv.bias.data)

        elif isinstance(conv, QueuedConv1d):
            self.inner_conv = nn.Conv1d(conv.inner_conv.in_channels,
                                        conv.inner_conv.out_channels,
                                        conv.inner_conv.kernel_size)
            self.init_len = conv.init_len
            self.inner_conv.weight.data.copy_(conv.inner_conv.weight.data)
            self.inner_conv.bias.data.copy_(conv.inner_conv.bias.data)

        self.init_queue(data)

    def init_queue(self, data):
        self.queue = deque([data[:, :, 0:1]]*self.init_len, maxlen=self.init_len)

    def forward(self, x):
        y = x
        x = torch.cat([self.queue[0], x], dim=2)
        self.queue.append(y)

        return self.inner_conv(x)
