import torch
import torch.nn as nn


class SingerClassifier(nn.Module):
    def __init__(self, singer_num, input_channels=64, hidden_channels=100):
        super(SingerClassifier, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.singer_num = singer_num

        self.dropout = nn.Dropout(p=0.2)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(self.input_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(self.hidden_channels, self.singer_num)

    def forward(self, x):
        out = self.dropout(x)
        # print(out.shape)
        # [2, 64, 20]

        out = self.conv_layers(out)
        # print(out.shape)
        # [2, 100, 20]

        out = torch.mean(out, dim=2)
        # print(out.shape)
        # [2, 100]

        out = self.fc(out)
        # print(out.shape)
        # [2, 12]

        return out
