import torch
import torch.nn as nn
from .utils import Conv2dNormRelu


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down_sample=True, norm=None):
        super().__init__()

        if down_sample:
            self.down0 = Conv2dNormRelu(in_channels, out_channels, stride=2, norm=norm, activation=None)
            self.conv0 = Conv2dNormRelu(in_channels, out_channels, kernel_size=3, stride=2, padding=1, norm=norm)
            self.conv1 = Conv2dNormRelu(out_channels, out_channels, kernel_size=3, stride=1, padding=1, norm=norm, activation=None)
        else:
            self.down0 = nn.Identity()
            self.conv0 = Conv2dNormRelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, norm=norm)
            self.conv1 = Conv2dNormRelu(out_channels, out_channels, kernel_size=3, stride=1, padding=1, norm=norm, activation=None)

        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        out = self.conv0(x)
        out = self.conv1(out)
        out = self.relu(out + self.down0(x))
        return out


class FeaturePyramid2D(nn.Module):
    def __init__(self, n_channels, norm=None):
        super().__init__()
        self.pyramid_convs = nn.ModuleList()
        for in_channels, out_channels in zip(n_channels[:-1], n_channels[1:]):
            self.pyramid_convs.append(ResidualBlock(in_channels, out_channels, norm=norm))

    def forward(self, x):
        outputs = []
        for conv in self.pyramid_convs:
            x = conv(x)
            outputs.append(x)
        return outputs


class FlowEstimatorDense2D(nn.Module):
    def __init__(self, n_channels, norm=None, conv_last=True):
        super().__init__()
        self.conv1 = Conv2dNormRelu(
            n_channels[0],
            n_channels[1],
            kernel_size=3, padding=1, norm=norm
        )
        self.conv2 = Conv2dNormRelu(
            n_channels[0] + n_channels[1],
            n_channels[2],
            kernel_size=3, padding=1, norm=norm
        )
        self.conv3 = Conv2dNormRelu(
            n_channels[0] + n_channels[1] + n_channels[2],
            n_channels[3],
            kernel_size=3, padding=1, norm=norm
        )
        self.conv4 = Conv2dNormRelu(
            n_channels[0] + n_channels[1] + n_channels[2] + n_channels[3],
            n_channels[4],
            kernel_size=3, padding=1, norm=norm
        )
        self.conv5 = Conv2dNormRelu(
            n_channels[0] + n_channels[1] + n_channels[2] + n_channels[3] + n_channels[4],
            n_channels[5],
            kernel_size=3, padding=1, norm=norm
        )
        self.flow_feat_dim = sum(n_channels)

        if conv_last:
            self.conv_last = nn.Conv2d(self.flow_feat_dim, 2, kernel_size=3, stride=1, padding=1)
        else:
            self.conv_last = None

    def forward(self, x):
        x1 = torch.cat([self.conv1(x), x], dim=1)
        x2 = torch.cat([self.conv2(x1), x1], dim=1)
        x3 = torch.cat([self.conv3(x2), x2], dim=1)
        x4 = torch.cat([self.conv4(x3), x3], dim=1)
        flow_feat = torch.cat([self.conv5(x4), x4], dim=1)

        if self.conv_last is not None:
            flow = self.conv_last(flow_feat)
            return flow_feat, flow
        else:
            return flow_feat


class ContextNetwork2D(nn.Module):
    def __init__(self, n_channels, dilations, norm=None):
        super().__init__()
        self.convs = nn.ModuleList()
        for in_channels, out_channels, dilation in zip(n_channels[:-1], n_channels[1:], dilations):
            self.convs.append(Conv2dNormRelu(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, norm=norm))
        self.conv_last = nn.Conv2d(n_channels[-1], 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        outputs = self.conv_last(x)
        return x, outputs
