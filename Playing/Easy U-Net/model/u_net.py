import torch
from torch.nn import Module, Conv2d

from model.layers import DownSample, DoubleConv, UpSample


class UNet(Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.down1 = DownSample(in_channels, 64)
        self.down2 = DownSample(64, 128)
        self.down3 = DownSample(128, 256)
        self.down4 = DownSample(256, 512)

        self.bottle_neck = DoubleConv(512, 1024)

        self.up1 = UpSample(1024, 512)
        self.up2 = UpSample(512, 256)
        self.up3 = UpSample(256, 128)
        self.up4 = UpSample(128, 64)

        self.out = Conv2d(64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        down1, p1 = self.down1(x)
        down2, p2 = self.down2(p1)
        down3, p3 = self.down3(p2)
        down4, p4 = self.down4(p3)

        b = self.bottle_neck(p4)

        up1 = self.up1(b, down4)
        up2 = self.up2(up1, down3)
        up3 = self.up3(up2, down2)
        up4 = self.up4(up3, down1)

        return self.out(up4)


if __name__ == "__main__":
    double_conv = DoubleConv(256, 256)
    print(double_conv)

    input_image = torch.rand((1, 3, 512, 512))
    model = UNet(3, 10)
    print(model(input_image).shape)
