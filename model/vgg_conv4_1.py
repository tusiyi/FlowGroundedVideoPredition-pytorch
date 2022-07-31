import torch
import torch.nn as nn


class Conv4_1(nn.Module):
    """
        VGG-19 encoder construction for flow2rgb, both content and flow(both channels=256)
        for content, in_channels=3; for flow, in_channels=2
    """
    def __init__(self, in_channels, channels):
        super(Conv4_1, self).__init__()
        self.in_c = in_channels
        self.c = channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=channels // 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels // 4, out_channels=channels // 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=channels // 4, out_channels=channels // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels // 2, out_channels=channels // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=channels // 2, out_channels=channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=channels, out_channels=channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class InvertConcatConv4_1(nn.Module):
    """
    Decoder of content and flow features, symmetric to VGG-19 encoder
    out_channels=3 denotes final output is RGB image
    """
    def __init__(self, channels, out_channels):
        super(InvertConcatConv4_1, self).__init__()
        self.c = channels
        self.out_c = out_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=channels * 4, out_channels=channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),  # according to the paper, use nearest upsampling

            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels // 2, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv2d(in_channels=channels // 2, out_channels=channels // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels // 2, out_channels=channels // 4, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv2d(in_channels=channels // 4, out_channels=channels // 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels // 4, out_channels=out_channels, kernel_size=3, padding=1),

        )

    def forward(self, content, flow):
        x = torch.cat((content, flow), dim=1)  # concate in channels
        return self.net(x)
