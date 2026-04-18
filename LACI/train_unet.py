# unet_original.py
# Minimal "classic" U-Net (Ronneberger et al., 2015) in PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    # two 3x3 convs with valid padding (no padding), ReLU
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

def center_crop(t, target_h, target_w):
    _, _, h, w = t.shape
    dh = (h - target_h) // 2
    dw = (w - target_w) // 2
    return t[:, :, dh:dh+target_h, dw:dw+target_w]

class UNetClassic(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, base=64):
        super().__init__()
        # encoder
        self.inc   = DoubleConv(in_channels, base)          # 64
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base,   base*2))  # 128
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base*2, base*4))  # 256
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base*4, base*8))  # 512
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base*8, base*16)) # 1024 (bottleneck)

        # decoder (up-conv + concat(cropped skip) + double conv)
        self.up1t  = nn.ConvTranspose2d(base*16, base*8, kernel_size=2, stride=2)
        self.up1c  = DoubleConv(base*16, base*8)

        self.up2t  = nn.ConvTranspose2d(base*8, base*4, kernel_size=2, stride=2)
        self.up2c  = DoubleConv(base*8, base*4)

        self.up3t  = nn.ConvTranspose2d(base*4, base*2, kernel_size=2, stride=2)
        self.up3c  = DoubleConv(base*4, base*2)

        self.up4t  = nn.ConvTranspose2d(base*2, base, kernel_size=2, stride=2)
        self.up4c  = DoubleConv(base*2, base)

        self.outc  = nn.Conv2d(base, num_classes, kernel_size=1)

    def forward(self, x):
        # encode
        x1 = self.inc(x)     # (H-4)
        x2 = self.down1(x1)  # (H/2-4)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)  # bottleneck

        # decode
        u1  = self.up1t(x5)
        x4c = center_crop(x4, u1.size(2), u1.size(3))
        u1  = torch.cat([x4c, u1], dim=1)
        u1  = self.up1c(u1)

        u2  = self.up2t(u1)
        x3c = center_crop(x3, u2.size(2), u2.size(3))
        u2  = torch.cat([x3c, u2], dim=1)
        u2  = self.up2c(u2)

        u3  = self.up3t(u2)
        x2c = center_crop(x2, u3.size(2), u3.size(3))
        u3  = torch.cat([x2c, u3], dim=1)
        u3  = self.up3c(u3)

        u4  = self.up4t(u3)
        x1c = center_crop(x1, u4.size(2), u4.size(3))
        u4  = torch.cat([x1c, u4], dim=1)
        u4  = self.up4c(u4)

        return self.outc(u4)  # logits (no activation)


# quick self-test: 572x572 -> 388x388 (same as the paper)
if __name__ == "__main__":
    x = torch.randn(1, 1, 572, 572)
    net = UNetClassic(in_channels=1, num_classes=4, base=64)
    y = net(x)
    print("in :", x.shape)
    print("out:", y.shape)
