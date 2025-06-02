import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(
        self,
        in_channels=3,
        channels=[32, 64],
        kernel_sizes=[3, 1],
        strides=[2, 2],
        hidden_layer=512,
        out_size=64,
        device=torch.device('cuda')):

        super().__init__()
        self.device = device
        self.conv1 = nn.Conv2d(in_channels, channels[0], kernel_sizes[0], strides[0], device=device)
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_sizes[1], strides[1], device=device)
        self.linear1 = nn.Linear(64, hidden_layer, device=device)
        self.linear2 = nn.Linear(hidden_layer, out_size, device=device)

    def forward(self, inputs):
        inputs = inputs.to( self.device)
        x = F.relu(self.conv1(inputs / 255.))
        x = F.relu(self.conv2(x))
        x = x.view(-1,  8 * 8)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)

        return x