from torch import cat
from torch.nn import Sequential, Conv2d, ReLU, Module, MaxPool2d, ConvTranspose2d


class DoubleConv(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = Sequential(
            Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_op(x)


class DownSample(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(x)
        return down, p


class UpSample(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = cat((x2, x1), 1)
        return self.conv(x)
