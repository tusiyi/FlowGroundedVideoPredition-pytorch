import torch
import torch.nn as nn


class MulConstant(nn.Module):
    """ layer for nn.MulConstant() """
    def __init__(self, constant, **kwargs):
        super(MulConstant, self).__init__(**kwargs)
        self.c = constant

    def forward(self, x):
        return x * self.c


class Exp(nn.Module):
    """ layer for nn.Exp() """
    def __init__(self, **kwargs):
        super(Exp, self).__init__(**kwargs)

    def forward(self, x):
        return torch.exp(x)


class CMulTable(nn.Module):
    """ layer for nn.CMulTable() """
    def __init__(self, **kwargs):
        super(CMulTable, self).__init__(**kwargs)

    def forward(self, x, y):
        return torch.multiply(x, y)


class CAddTable(nn.Module):
    """ layer for nn.CAddTable() """
    def __init__(self, **kwargs):
        super(CAddTable, self).__init__(**kwargs)

    def forward(self, x, y):
        return torch.add(x, y)


class Gaussian(nn.Module):
    """ layer for nn.Gaussian(0, 1) """
    def __init__(self, device, **kwargs):
        super(Gaussian, self).__init__(**kwargs)
        self.device = device

    def forward(self):
        m = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        sample_z = m.sample()
        sample_z = sample_z.to(self.device)
        return sample_z


###############################################################################################

class VolEncoder(nn.Module):
    def __init__(self, channels, naf, z_dim):
        super(VolEncoder, self).__init__()
        self.channels = channels
        self.naf = naf
        self.z_dim = z_dim
        self.mul_constant = MulConstant(0.2)   # lua version: encoder:add(nn.MulConstant(0.2))

        self.net = nn.Sequential(
            self.mul_constant,

            nn.Conv3d(in_channels=channels, out_channels=naf, kernel_size=3, padding=1),
            nn.BatchNorm3d(naf),
            nn.ReLU(inplace=True),  # inplace=True change input directly

            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(in_channels=naf, out_channels=naf, kernel_size=3, padding=1),
            nn.BatchNorm3d(naf),
            nn.ReLU(inplace=True),

            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(in_channels=naf, out_channels=naf * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(naf * 2),
            nn.ReLU(inplace=True),

            nn.MaxPool3d(2),  # 3 dimensions are both 2
            nn.Conv3d(in_channels=naf * 2, out_channels=naf * 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(naf * 4),
            nn.ReLU(inplace=True),

            nn.MaxPool3d(2),
            nn.Conv3d(in_channels=naf * 4, out_channels=naf * 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(naf * 8),
            nn.ReLU(inplace=True),

            nn.MaxPool3d(2),
        )
        self.z1 = nn.Sequential(
            nn.Conv3d(naf * 8, z_dim, kernel_size=(2, 4, 4))
        )
        self.z2 = nn.Sequential(
            nn.Conv3d(naf * 8, z_dim, kernel_size=(2, 4, 4))
        )

    def forward(self, x):
        x = self.net(x)
        out1 = self.z1(x).view(-1, self.z_dim, 1, 1, 1)
        out2 = self.z2(x).view(-1, self.z_dim, 1, 1, 1)
        return out1, out2


class VolSampler(nn.Module):
    def __init__(self, device):
        super(VolSampler, self).__init__()
        self.device = device

        self.mul_constant = MulConstant(0.5)
        self.exp = Exp()
        self.gaussian = Gaussian(self.device)
        self.mul = CMulTable()
        self.add = CAddTable()
        self.std = nn.Sequential(
            self.mul_constant,
            self.exp
        )

    def forward(self, x1, x2):
        """x1:mu,  x2: (1/2)*log(sigma ^ 2)"""
        sigma = self.std(x2)
        z = self.gaussian()
        x = self.add(x1, self.mul(sigma, z))
        return x


class VolDecoder(nn.Module):
    def __init__(self, channels, ngf, z_dim):
        super(VolDecoder, self).__init__()
        self.channels = channels
        self.ngf = ngf
        self.z_dim = z_dim
        self.net = nn.Sequential(
            # 猜测VolumetricFullConvolution是反卷积过程
            nn.ConvTranspose3d(z_dim, ngf * 8, kernel_size=(2, 4, 4)),
            nn.BatchNorm3d(ngf * 8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(ngf * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(ngf * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(ngf),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(ngf, ngf, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(ngf),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(ngf, channels, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=1),
            nn.Tanh(),
        )
        self.mul = CMulTable()
        self.add = CAddTable()
        self.mul_constant = MulConstant(5)  # lua: nn.MulConstant(5)

    def forward(self, im_embed, flow_embed):
        x = self.mul(im_embed, flow_embed)
        x = self.add(x, im_embed)
        x = self.net(x)
        x = self.mul_constant(x)
        return x


class ImEncoder(nn.Module):
    def __init__(self, channels, naf, z_dim):
        super(ImEncoder, self).__init__()
        self.channels = channels
        self.naf = naf
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=naf, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=naf, out_channels=naf, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=naf, out_channels=naf * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=naf * 2, out_channels=naf * 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=naf * 4, out_channels=naf * 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=naf * 8, out_channels=z_dim, kernel_size=4),
        )

    def forward(self, x):
        x = self.net(x)
        # return x
        return x.view(-1, self.z_dim, 1, 1, 1)
