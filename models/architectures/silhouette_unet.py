import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SilhouetteUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1) -> None:
        super().__init__()
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(128, 64)

        self.up0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up0 = DoubleConv(64, 32)

        self.outc = nn.Conv2d(32, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        xb = self.bottleneck(x4)

        u3 = self.up3(xb)
        u3 = torch.cat([x4, u3], dim=1)
        x_up3 = self.conv_up3(u3)

        u2 = self.up2(x_up3)
        u2 = torch.cat([x3, u2], dim=1)
        x_up2 = self.conv_up2(u2)

        u1 = self.up1(x_up2)
        u1 = torch.cat([x2, u1], dim=1)
        x_up1 = self.conv_up1(u1)

        u0 = self.up0(x_up1)
        u0 = torch.cat([x1, u0], dim=1)
        x_up0 = self.conv_up0(u0)

        logits = self.outc(x_up0)
        return self.sigmoid(logits)


if __name__ == "__main__":
    dummy_input = torch.randn(1, 3, 256, 256)
    model = SilhouetteUNet()
    output = model(dummy_input)
    print(
        f"SilhouetteUNet forward shape: {output.shape}, min: {output.min().item():.4f}, max: {output.max().item():.4f}"
    )
