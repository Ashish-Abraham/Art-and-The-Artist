import torch
from torch import nn
import torch.nn.functional as F

class SepResidualBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()

        self.blocks = nn.Sequential(
            nn.ReLU(),
            nn.ReflectionPad2d(dilation),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, 
                      padding=0, dilation=dilation, 
                      groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, 
                      padding=0, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.blocks(x)
    
class ResidualHourglass(nn.Module):
    def __init__(self, channels, mult=0.5):
        super().__init__()

        hidden_channels = int(channels * mult)

        self.blocks = nn.Sequential(
            nn.ReLU(),
            # Downsample
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, hidden_channels, kernel_size=3, stride=2, 
                      padding=0, dilation=1, 
                      groups=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            # Bottleneck
            SepResidualBlock(channels=hidden_channels, dilation=1),
            SepResidualBlock(channels=hidden_channels, dilation=2),
            SepResidualBlock(channels=hidden_channels, dilation=1),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_channels, channels, kernel_size=3, stride=1, 
                      padding=0, dilation=1, 
                      groups=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            # Upsample
            nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2, 
                               padding=0, groups=1, bias=True),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.blocks(x)    