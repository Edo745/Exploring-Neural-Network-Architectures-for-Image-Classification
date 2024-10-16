import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size=3*32*32, num_classes=10):
        super(MLP, self).__init__()
        
        self.flatten = nn.Flatten()
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, 4096),
            nn.ReLU(inplace=True),
            
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        return x
