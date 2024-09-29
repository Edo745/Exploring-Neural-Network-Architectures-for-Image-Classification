import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, num_classes=10):
        super(MLP, self).__init__()
         
        self.mlp = nn.Sequential(
            nn.Linear(3*32*32, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.mlp(x)
        return x