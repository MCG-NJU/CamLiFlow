import torch
import torch.nn as nn
from .utils import MLP2d, batch_indexing_channel_first, batch_indexing_channel_last
from .csrc import k_nearest_neighbor


class PointConvDownSampling(nn.Module):
    def __init__(self, in_channels, out_channels, norm=None, activation='leaky_relu', k=16):
        super().__init__()
        self.k = k

        self.weight_net = MLP2d(3, [8, 16], activation=activation)
        self.linear = nn.Linear(16 * (in_channels + 3), out_channels)

        if norm == 'batch_norm':
            self.norm_fn = nn.BatchNorm1d(out_channels)
        elif norm == 'instance_norm':
            self.norm_fn = nn.InstanceNorm1d(out_channels)
        elif norm is None:
            self.norm_fn = nn.Identity()
        else:
            raise NotImplementedError('Unknown normalization function: %s' % norm)

        if activation == 'relu':
            self.activation_fn = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation_fn = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation is None:
            self.activation_fn = nn.Identity()
        else:
            raise NotImplementedError('Unknown activation function: %s' % activation)

    def forward(self, xyz, features, sampled_xyz):
        """
        :param xyz: 3D locations of points, [batch_size, 3, n_points]
        :param features: features of points, [batch_size, in_channels, n_points]
        :param sampled_xyz: 3D locations of sampled points, [batch_size, 3, n_samples]
        :return weighted_features: features of sampled points, [batch_size, out_channels, n_samples]
        """
        batch_size, n_points, n_samples = xyz.shape[0], xyz.shape[2], sampled_xyz.shape[2]

        features = torch.cat([xyz, features], dim=1)  # [bs, in_channels + 3, n_points]
        features_cl = features.transpose(1, 2)  # [bs, n_points, n_channels + 3]

        # Calculate k nearest neighbors
        knn_indices = k_nearest_neighbor(xyz, sampled_xyz, self.k)  # [bs, n_samples, k]

        # Calculate weights
        knn_xyz = batch_indexing_channel_first(xyz, knn_indices)  # [bs, 3, n_samples, k]
        knn_xyz_norm = knn_xyz - sampled_xyz[:, :, :, None]  # [bs, 3, n_samples, k]
        weights = self.weight_net(knn_xyz_norm)  # [bs, n_weights, n_samples, k]

        # Calculate weighted features
        weights = weights.transpose(1, 2)  # [bs, n_samples, n_weights, k]
        knn_features = batch_indexing_channel_last(features_cl, knn_indices)  # [bs, n_samples, k, 3 + in_channels]
        weighted_features = torch.matmul(weights, knn_features)  # [bs, n_samples, n_weights, 3 + in_channels]
        weighted_features = weighted_features.view(batch_size, n_samples, -1)  # [bs, n_samples, (3 + in_channels) * n_weights]
        weighted_features = self.linear(weighted_features)  # [bs, n_samples, out_channels]
        weighted_features = self.activation_fn(self.norm_fn(weighted_features.transpose(1, 2)))  # [bs, out_channels, n_samples]

        return weighted_features


class PointConvNoSampling(nn.Module):
    def __init__(self, in_channels, out_channels, norm=None, activation='leaky_relu', k=16):
        super().__init__()
        self.k = k

        self.weight_net = MLP2d(3, [8, 16], activation=activation)
        self.linear = nn.Linear(16 * (in_channels + 3), out_channels)

        if norm == 'batch_norm':
            self.norm_fn = nn.BatchNorm1d(out_channels)
        elif norm == 'instance_norm':
            self.norm_fn = nn.InstanceNorm1d(out_channels)
        elif norm is None:
            self.norm_fn = nn.Identity()
        else:
            raise NotImplementedError('Unknown normalization function: %s' % norm)

        if activation == 'relu':
            self.activation_fn = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation_fn = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation is None:
            self.activation_fn = nn.Identity()
        else:
            raise NotImplementedError('Unknown activation function: %s' % activation)

    def forward(self, xyz, features, knn_indices=None):
        """
        :param xyz: 3D locations of points, [batch_size, 3, n_points]
        :param features: features of points, [batch_size, in_channels, n_points]
        :param knn_indices: optional pre-computed knn indices, [batch_size, n_points, k]
        :return weighted_features: features of sampled points, [batch_size, out_channels, n_samples]
        """
        batch_size, n_points = xyz.shape[0], xyz.shape[2]
        features = torch.cat([xyz, features], dim=1)  # [bs, in_channels + 3, n_points]
        features_cl = features.transpose(1, 2)  # [bs, n_points, n_channels + 3]

        # Calculate k nearest neighbors
        if knn_indices is not None:  # if knn indices are pre-computed
            assert knn_indices.shape[:2] == torch.Size([batch_size, n_points])
            assert knn_indices.shape[2] >= self.k
            knn_indices = knn_indices[:, :, :self.k]
        else:
            knn_indices = k_nearest_neighbor(xyz, xyz, self.k)  # [bs, n_samples, k]

        # Calculate weights
        knn_xyz = batch_indexing_channel_first(xyz, knn_indices)  # [bs, 3, n_points, k]
        knn_xyz_norm = knn_xyz - xyz[:, :, :, None]  # [bs, 3, n_points, k]
        weights = self.weight_net(knn_xyz_norm)  # [bs, n_weights, n_points, k]

        # Calculate weighted features
        weights = weights.transpose(1, 2)  # [bs, n_points, n_weights, k]
        knn_features = batch_indexing_channel_last(features_cl, knn_indices)  # [bs, n_points, k, 3 + in_channels]
        weighted_features = torch.matmul(weights, knn_features)  # [bs, n_points, n_weights, 3 + in_channels]
        weighted_features = weighted_features.view(batch_size, n_points, -1)  # [bs, n_points, (3 + in_channels) * n_weights]
        weighted_features = self.linear(weighted_features).float()  # [bs, n_points, out_channels]
        weighted_features = self.activation_fn(self.norm_fn(weighted_features.transpose(1, 2)))  # [bs, out_channels, n_points]

        return weighted_features
