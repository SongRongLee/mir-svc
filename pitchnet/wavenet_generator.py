# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license described in the README.md
# file located in the root directory of this source tree.
#
# Modifications have made by Song-Rong, Lee
#

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .util.const import MU
from .net.share import QueuedConv1d


class WaveNetGenerator():
    def __init__(self, wavenet, batch_size=1):
        self.wavenet = wavenet
        self.wavenet.shift_input = False
        self.batch_size = batch_size

        # Swap conv layers in WaveNet
        x = torch.zeros(self.batch_size, 1, 1).cuda()
        self.wavenet.first_conv = QueuedConv1d(self.wavenet.first_conv, x)
        x = torch.zeros(self.batch_size, self.wavenet.residual_channels, 1).cuda()
        for layer in self.wavenet.layers:
            layer.causal = QueuedConv1d(layer.causal, x)

        self.wavenet.cuda()

    def reset(self):
        return self.init()

    def init(self, batch_size=None):
        if batch_size is not None:
            self.batch_size = batch_size

        x = torch.zeros(self.batch_size, 1, 1).cuda()
        self.wavenet.first_conv.init_queue(x)
        x = torch.zeros(self.batch_size, self.wavenet.residual_channels, 1).cuda()
        for layer in self.wavenet.layers:
            layer.causal.init_queue(x)

    def get_sample(self, prediction, method):
        if method == 'sample':
            probabilities = F.softmax(prediction, dim=1)
            sample = torch.multinomial(probabilities, 1)
        elif method == 'max':
            _, sample = torch.max(F.softmax(prediction, dim=1), dim=1, keepdim=True)
        else:
            raise ValueError('Method {} not supported.'.format(method))
        return sample

    def generate(self, conditions, method='sample'):
        # Initialize
        batch_size, sample_length = conditions.shape[0], conditions.shape[2]
        self.init(batch_size)
        samples = torch.empty(batch_size, sample_length + 1).fill_((MU+1)/2).cuda()

        # Forward wavenet for each sample point
        self.wavenet.eval()
        with torch.no_grad():
            for t in tqdm(range(sample_length)):
                x = samples[:, t:t+1].clone()
                c = conditions[:, :, t:t+1]

                prediction = self.wavenet(x, c)[:, :, 0]
                sample = self.get_sample(prediction, method).squeeze(1)
                samples[:, t+1] = sample
        return samples[:, 1:]
