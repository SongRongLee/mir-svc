import torch
import torch.nn as nn


class SingerClassifier(nn.Module):
    def __init__(self, singer_num, hidden_channels=32, feature_dim=20):
        super(SingerClassifier, self).__init__()
        self.hidden_channels = hidden_channels
        self.singer_num = singer_num
        self.feature_dim = feature_dim
        self.max_pool_stride = 2
        self.linear_hidden = 16

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(p=0.75)
        self.max_pool = nn.MaxPool2d(2, self.max_pool_stride)
        self.fc = nn.Sequential(
            nn.Linear(int(self.hidden_channels * self.feature_dim / self.max_pool_stride), self.linear_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.linear_hidden, self.singer_num)
        )

    def forward(self, x):
        out = x.unsqueeze(1)
        # print(out.shape)
        # [2, 1, 20, 301]

        out = self.conv_layers(out)
        out = self.dropout(out)
        # print(out.shape)
        # [2, 32, 20, 301]

        out = self.max_pool(out)
        # print(out.shape)
        # [2, 32, 10, 150]

        out = torch.mean(out, dim=3)
        out = out.reshape(out.shape[0], -1)
        # print(out.shape)
        # [2, 320]

        out = self.fc(out)
        # print(out.shape)
        # [2, 12]

        return out
