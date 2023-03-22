import torch
import torch.nn as nn


class LayerNormCF1d(nn.Module):
    """LayerNorm that supports the channel_first data format."""

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None] * x + self.bias[:, None]
        return x


class LayerNormCF2d(nn.Module):
    """LayerNorm that supports the channel_first data format."""

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class Conv1dNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, norm=None, act='leaky_relu'):
        super().__init__()

        self.conv_fn = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=norm is None,
        )

        if norm == 'batch_norm':
            self.norm_fn = nn.BatchNorm1d(out_channels, affine=True)
        elif norm == 'instance_norm':
            self.norm_fn = nn.InstanceNorm1d(out_channels)
        elif norm == 'instance_norm_affine':
            self.norm_fn = nn.InstanceNorm1d(out_channels, affine=True)
        elif norm == 'layer_norm':
            self.norm_fn = LayerNormCF1d(out_channels)
        elif norm is None:
            self.norm_fn = nn.Identity()
        else:
            raise NotImplementedError('Unknown normalization function: %s' % norm)

        if act == 'relu':
            self.act_fn = nn.ReLU(inplace=True)
        elif act == 'leaky_relu':
            self.act_fn = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif act == 'sigmoid':
            self.act_fn = nn.Sigmoid()
        elif act is None:
            self.act_fn = nn.Identity()
        else:
            raise NotImplementedError('Unknown activation function: %s' % act)

    def forward(self, x):
        x = self.conv_fn(x)
        x = self.norm_fn(x)
        x = self.act_fn(x)
        return x


class Conv2dNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, norm=None, act='leaky_relu'):
        super().__init__()

        self.conv_fn = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=norm is None,
        )

        if norm == 'batch_norm':
            self.norm_fn = nn.BatchNorm2d(out_channels)
        elif norm == 'instance_norm':
            self.norm_fn = nn.InstanceNorm2d(out_channels)
        elif norm == 'instance_norm_affine':
            self.norm_fn = nn.InstanceNorm2d(out_channels, affine=True)
        elif norm == 'layer_norm':
            self.norm_fn = LayerNormCF2d(out_channels)
        elif norm is None:
            self.norm_fn = nn.Identity()
        else:
            raise NotImplementedError('Unknown normalization function: %s' % norm)

        if act == 'relu':
            self.act_fn = nn.ReLU(inplace=True)
        elif act == 'leaky_relu':
            self.act_fn = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif act == 'sigmoid':
            self.act_fn = nn.Sigmoid()
        elif act is None:
            self.act_fn = nn.Identity()
        else:
            raise NotImplementedError('Unknown activation function: %s' % act)

    def forward(self, x):
        x = self.conv_fn(x)
        x = self.norm_fn(x)
        x = self.act_fn(x)
        return x


class MLP1d(nn.Module):
    def __init__(self, in_channels, mlp_channels, norm=None, act='leaky_relu'):
        super().__init__()
        assert isinstance(in_channels, int)
        assert isinstance(mlp_channels, list)
        n_channels = [in_channels] + mlp_channels

        self.convs = nn.ModuleList()
        for in_channels, out_channels in zip(n_channels[:-1], n_channels[1:]):
            self.convs.append(Conv1dNormRelu(in_channels, out_channels, norm=norm, act=act))

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x


class MLP2d(nn.Module):
    def __init__(self, in_channels, mlp_channels, norm=None, act='leaky_relu'):
        super().__init__()
        assert isinstance(in_channels, int)
        assert isinstance(mlp_channels, list)
        n_channels = [in_channels] + mlp_channels

        self.convs = nn.ModuleList()
        for in_channels, out_channels in zip(n_channels[:-1], n_channels[1:]):
            self.convs.append(Conv2dNormRelu(in_channels, out_channels, norm=norm, act=act))

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x
