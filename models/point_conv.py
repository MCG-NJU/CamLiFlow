import torch
import torch.nn as nn
from .utils import batch_indexing, k_nearest_neighbor
from .mlp import LayerNormCF1d, MLP1d, MLP2d


class PointConv(nn.Module):
    def __init__(self, in_channels, out_channels, norm=None, act='leaky_relu', k=16):
        super().__init__()
        self.k = k

        self.weight_net = MLP2d(3, [8, 16], act=act)
        self.linear = nn.Linear(16 * (in_channels + 3), out_channels)

        if norm == 'batch_norm':
            self.norm_fn = nn.BatchNorm1d(out_channels, affine=True)
        elif norm == 'instance_norm':
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
        elif act is None:
            self.act_fn = nn.Identity()
        else:
            raise NotImplementedError('Unknown activation function: %s' % act)

    def forward(self, xyz, features, sampled_xyz=None, knn_indices=None):
        """
        :param xyz: 3D locations of points, [batch_size, 3, n_points]
        :param features: features of points, [batch_size, in_channels, n_points]
        :param sampled_xyz: 3D locations of sampled points, [batch_size, 3, n_samples]
        :return weighted_features: features of sampled points, [batch_size, out_channels, n_samples]
        """
        if sampled_xyz is None:
            sampled_xyz = xyz

        bs, n_samples = sampled_xyz.shape[0], sampled_xyz.shape[-1]
        features = torch.cat([xyz, features], dim=1)  # [bs, in_channels + 3, n_points]
        features_cl = features.transpose(1, 2)  # [bs, n_points, n_channels + 3]

        # Calculate k nearest neighbors
        if knn_indices is None:
            knn_indices = k_nearest_neighbor(xyz, sampled_xyz, self.k)  # [bs, n_samples, k]
        else:
            assert knn_indices.shape[:2] == torch.Size([bs, n_samples])
            assert knn_indices.shape[2] >= self.k
            knn_indices = knn_indices[:, :, :self.k]

        # Calculate weights
        knn_xyz = batch_indexing(xyz, knn_indices)  # [bs, 3, n_samples, k]
        knn_xyz_norm = knn_xyz - sampled_xyz[:, :, :, None]  # [bs, 3, n_samples, k]
        weights = self.weight_net(knn_xyz_norm)  # [bs, n_weights, n_samples, k]

        # Calculate weighted features
        weights = weights.transpose(1, 2)  # [bs, n_samples, n_weights, k]
        knn_features = batch_indexing(features_cl, knn_indices, layout='channel_last')  # [bs, n_samples, k, 3 + in_channels]
        out = torch.matmul(weights, knn_features)  # [bs, n_samples, n_weights, 3 + in_channels]
        out = out.view(bs, n_samples, -1)  # [bs, n_samples, (3 + in_channels) * n_weights]
        out = self.linear(out)  # [bs, n_samples, out_channels]
        out = self.act_fn(self.norm_fn(out.transpose(1, 2)))  # [bs, out_channels, n_samples]

        return out


class PointNet2(nn.Module):
    def __init__(self, in_channels, mlp_channels, norm=None, act='leaky_relu', k=16):
        super().__init__()
        self.k = k
        self.mlp = MLP2d(in_channels + 3, mlp_channels, norm, act)

    def forward(self, xyz, features, sampled_xyz=None, knn_indices=None):
        if sampled_xyz is None:
            sampled_xyz = xyz

        # Calculate k nearest neighbors
        if knn_indices is None:
            knn_indices = k_nearest_neighbor(xyz, sampled_xyz, self.k)  # [bs, n_samples, k]
        else:
            bs, n_points = sampled_xyz.shape[0], sampled_xyz.shape[-1]
            assert knn_indices.shape[:2] == torch.Size([bs, n_points])
            assert knn_indices.shape[2] >= self.k
            knn_indices = knn_indices[:, :, :self.k]

        knn_xyz = batch_indexing(xyz, knn_indices)  # [bs, 3, n_samples, k]
        knn_offset = knn_xyz - sampled_xyz[:, :, :, None]  # [bs, 3, n_samples, k]

        features = batch_indexing(features, knn_indices)
        features = self.mlp(torch.cat([knn_offset, features], dim=1))  # [bs, out_channels, n_samples, k]
        features = torch.max(features, dim=-1)[0]  # [bs, out_channels, n_samples]

        return features


class PointConvDW(nn.Module):
    def __init__(self, in_channels, out_channels, norm=None, act='leaky_relu', k=16):
        super().__init__()
        self.k = k        
        self.mlp = MLP1d(in_channels, [out_channels], norm, act)
        self.weight_net = MLP2d(3, [8, 32, out_channels], act='relu')

    def forward(self, xyz, features, sampled_xyz=None, knn_indices=None):
        if sampled_xyz is None:
            sampled_xyz = xyz

        # Calculate k nearest neighbors
        if knn_indices is None:
            knn_indices = k_nearest_neighbor(xyz, sampled_xyz, self.k)  # [bs, n_samples, k]
        else:
            bs, n_points = sampled_xyz.shape[0], sampled_xyz.shape[-1]
            assert knn_indices.shape[:2] == torch.Size([bs, n_points])
            assert knn_indices.shape[2] >= self.k
            knn_indices = knn_indices[:, :, :self.k]

        knn_xyz = batch_indexing(xyz, knn_indices)  # [bs, 3, n_samples, k]
        knn_offset = knn_xyz - sampled_xyz[:, :, :, None]  # [bs, 3, n_samples, k]

        features = self.mlp(features)
        features = batch_indexing(features, knn_indices)
        features = features * self.weight_net(knn_offset)
        features = torch.max(features, dim=-1)[0]  # [bs, in_channels, n_samples]

        return features
