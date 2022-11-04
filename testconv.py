import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    in_channels: int = 3

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, kernel_size=(2,2), stride=1, out_channels=self.in_channels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return x

model = Model()
model.eval()
x = np.random.rand(1, 3, 64, 64)
print(model(x))