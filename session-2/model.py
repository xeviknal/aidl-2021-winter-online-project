import torch.nn as nn


class MyLeNet(nn.Module):

    def __init__(self, conv1_out=6, conv2_out=16, kernel=5, pooling=2, fc_hidden1=120, fc_hidden2=84):
        super().__init__()
        # Input: 1, 64x64px
        # Output: 6, 30x30 (kernel 5, pooling 2)
        self.conv1 = ConvBlock(1, conv1_out, kernel, pooling)  # 7 -> 58 -> 29
        conv1_out_size = (64 - kernel + 1) / pooling
        # Input: 6, 30x30
        # Output: 16, 13x13
        self.conv2 = ConvBlock(conv1_out, conv2_out, kernel, pooling)  # 6 -> 24 -> 12
        conv2_out_size = (conv1_out_size - kernel + 1) / pooling
        # Input: 16 * 13 * 13  # 13*12*12
        self.feature_size = int(conv2_out * conv2_out_size * conv2_out_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.feature_size, fc_hidden1),
            nn.ReLU(),
            nn.Linear(fc_hidden1, fc_hidden2),
            nn.ReLU(),
            nn.Linear(fc_hidden2, 15),
            nn.LogSoftmax()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        bsz, nch, height, width = x.shape
        x = x.view(bsz, self.feature_size)
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
