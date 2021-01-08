import torch.nn as nn


class MyLeNet(nn.Module):

    def __init__(self):
        super().__init__()
        # Input: 1, 64x64px
        # Output: 6, 30x30 (kernel 5, pooling 2)
        self.conv1 = ConvBlock(1, 6, 5)  # 7 -> 58 -> 29
        # Input: 6, 30x30
        # Output: 16, 13x13
        self.conv2 = ConvBlock(6, 16, 5)  # 6 -> 24 -> 12
        # Input: 16 * 13 * 13  # 13*12*12
        self.mlp = nn.Sequential(
            nn.Linear(2704, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 15),
            nn.LogSoftmax()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        bsz, nch, height, width = x.shape
        x = x.view(bsz, 2704)
        y = self.mlp(x)
        return y


class ConvBlock(nn.Module):

    def __init__(self, num_inp_channels, num_out_fmaps,
                 kernel_size, pool_size=2):
        super().__init__()
        self.conv = nn.Conv2d(num_inp_channels, num_out_fmaps, kernel_size)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(pool_size)

    def forward(self, x):
        return self.maxpool(self.relu(self.conv(x)))
