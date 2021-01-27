import torch.nn as nn


class MyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.pipeline = nn.Sequential( # [128, 128]
            nn.Conv2d(3, 32, 3), # [32, 126, 126]
            nn.ReLU(),
            nn.MaxPool2d(2), # [32, 63, 63]
            nn.Conv2d(32, 64, 3), # [64, 61, 61]
            nn.ReLU(),
            nn.MaxPool2d(2), # [64, 30, 30]
            nn.Conv2d(64, 128, 3), # [128, 27, 27]
            nn.ReLU(),
            nn.MaxPool2d(2), # [128, 13, 13]
            nn.Conv2d(128, 256, 3), # [256, 11, 11]
            nn.ReLU(),
            nn.MaxPool2d(2), # [256, 5, 5]
            nn.Flatten(),
            nn.Linear(256*5*5, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.pipeline(x)
