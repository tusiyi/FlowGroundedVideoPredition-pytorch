import torch
import torch.nn as nn


class ContentConv4_1(nn.Module):
    """
        VGG-19 encoder construction for flow2rgb, both content and flow(both channels=256)
        for content, in_channels=3; for flow, in_channels=2
    """
    def __init__(self, in_channels, channels):
        super(ContentConv4_1, self).__init__()
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
        out = self.net(x)
        return out


class FlowConv4_1(nn.Module):
    """
        VGG-19 encoder construction for flow2rgb, both content and flow(both channels=256)
        for content, in_channels=3; for flow, in_channels=2
    """
    def __init__(self, in_channels, channels):
        super(FlowConv4_1, self).__init__()
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
        out = self.net(x)
        return out


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
        x = torch.cat((content, flow), dim=1)  # concat in channels
        out = self.net(x)
        return out


class FlowToRGB(nn.Module):
    def __init__(self, img_channels, flow_channels, out_channels, common_channels):
        super(FlowToRGB, self).__init__()
        self.netC = ContentConv4_1(in_channels=img_channels, channels=common_channels)
        self.netF = FlowConv4_1(in_channels=flow_channels, channels=common_channels)
        self.netD = InvertConcatConv4_1(out_channels=out_channels, channels=common_channels)

    def forward(self, image, flow):
        image_feat = self.netC(image)
        flow_feat = self.netF(flow)
        x = self.netD(image_feat, flow_feat)

        return x
