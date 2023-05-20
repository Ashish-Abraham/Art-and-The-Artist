from blocks import SepResidualBlock, ResidualHourglass
import torch
from torch import nn

class FinalNet(torch.nn.Module):
    def __init__(self, width=8):
        super().__init__()

        self.blocks = nn.Sequential( 
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, width, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(width, affine=True),
            ResidualHourglass(channels=width),
            ResidualHourglass(channels=width),
            SepResidualBlock(channels=width, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, 3, kernel_size=3, stride=1, padding=1, bias=True)
        )

        # Normalization
        self.blocks[1].weight.data /= 127.5   # -1 to 1
        self.blocks[-1].weight.data *= 127.5 / 8 # 0 to 255
        self.blocks[-1].bias.data.fill_(127.5)  # weights + bias(here bias is set to 127.5 to achieve range [0,255])

    def forward(self, x):
        return self.blocks(x)