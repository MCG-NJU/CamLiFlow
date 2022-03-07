import torch
import torch.nn as nn
from .pointconv import PointConvNoSampling, PointConvDownSampling
from .utils import MLP1d, MLP2d, batch_indexing_channel_first
from .csrc import k_nearest_neighbor, furthest_point_sampling


def build_pc_pyramid(pc1, pc2, n_samples_list):
    batch_size, _, n_points = pc1.shape

    # sub-sampling point cloud
    pc_both = torch.cat([pc1, pc2], dim=0)
    sample_index_both = furthest_point_sampling(pc_both.transpose(1, 2), max(n_samples_list))  # 1/4
    sample_index1 = sample_index_both[:batch_size]
    sample_index2 = sample_index_both[batch_size:]

    # build point cloud pyramid
    lv0_index = torch.arange(n_points, device=pc1.device)
    lv0_index = lv0_index[None, :].expand(batch_size, n_points)
    xyzs1, xyzs2, sample_indices1, sample_indices2 = [pc1], [pc2], [lv0_index], [lv0_index]

    for n_samples in n_samples_list:  # 1/4
        sample_indices1.append(sample_index1[:, :n_samples])
        sample_indices2.append(sample_index2[:, :n_samples])
        xyzs1.append(batch_indexing_channel_first(pc1, sample_index1[:, :n_samples]))
        xyzs2.append(batch_indexing_channel_first(pc2, sample_index2[:, :n_samples]))

    return xyzs1, xyzs2, sample_indices1, sample_indices2


class FeaturePyramid3D(nn.Module):
    def __init__(self, n_channels, norm=None, k=16):
        super().__init__()

        self.level0_mlp = MLP1d(3, [n_channels[0], n_channels[0]])

        self.pyramid_mlps = nn.ModuleList()
        self.pyramid_convs = nn.ModuleList()

        for i in range(len(n_channels) - 1):
            self.pyramid_mlps.append(MLP1d(n_channels[i], [n_channels[i], n_channels[i + 1]]))
            self.pyramid_convs.append(PointConvDownSampling(n_channels[i + 1], n_channels[i + 1], norm=norm, k=k))

    def forward(self, xyzs):
        """
        :param xyzs: pyramid of points
        :return feats: pyramid of features
        """
        assert len(xyzs) == len(self.pyramid_mlps) + 1

        inputs = torch.zeros_like(xyzs[0])  # [bs, 3, n_points]
        feats = [self.level0_mlp(inputs)]  # [bs, 32, n_points]

        for i in range(len(xyzs) - 1):
            feats.append(self.pyramid_convs[i](xyzs[i], self.pyramid_mlps[i](feats[-1]), xyzs[i + 1]))

        return feats


class Correlation3D(nn.Module):
    def __init__(self, in_channels, out_channels, k=16):
        super().__init__()

        self.k = k
        self.cost_mlp = MLP2d(3 + 2 * in_channels, [out_channels, out_channels], activation='leaky_relu')
        self.weight_net1 = MLP2d(3, [8, 8, out_channels], activation='relu')
        self.weight_net2 = MLP2d(3, [8, 8, out_channels], activation='relu')

    def forward(self, xyz1, feat1, xyz2, feat2, knn_indices_1in1=None):
        """
        :param xyz1: [batch_size, 3, n_points]
        :param feat1: [batch_size, in_channels, n_points]
        :param xyz2: [batch_size, 3, n_points]
        :param feat2: [batch_size, in_channels, n_points]
        :param knn_indices_1in1: for each point in xyz1, find its neighbors in xyz1, [batch_size, n_points, k]
        :return cost volume for each point in xyz1: [batch_size, n_cost_channels, n_points]
        """
        batch_size, in_channels, n_points = feat1.shape

        # Step1: for each point in xyz1, find its neighbors in xyz2
        knn_indices_1in2 = k_nearest_neighbor(input_xyz=xyz2, query_xyz=xyz1, k=self.k)
        # knn_xyz2: [bs, 3, n_points, k]
        knn_xyz2 = batch_indexing_channel_first(xyz2, knn_indices_1in2)
        # knn_xyz2_norm: [bs, 3, n_points, k]
        knn_xyz2_norm = knn_xyz2 - xyz1.view(batch_size, 3, n_points, 1)
        # knn_features2: [bs, in_channels, n_points, k]
        knn_features2 = batch_indexing_channel_first(feat2, knn_indices_1in2)
        # features1_expand: [bs, in_channels, n_points, k]
        features1_expand = feat1[:, :, :, None].expand(batch_size, in_channels, n_points, self.k)
        # concatenated_features: [bs, 2 * in_channels + 3, n_points, k]
        concatenated_features = torch.cat([features1_expand, knn_features2, knn_xyz2_norm], dim=1)
        # p2p_cost (point-to-point cost): [bs, out_channels, n_points, k]
        p2p_cost = self.cost_mlp(concatenated_features)

        # weights2: [bs, out_channels, n_points, k]
        weights2 = self.weight_net2(knn_xyz2_norm)
        # p2n_cost (point-to-neighbor cost): [bs, out_channels, n_points]
        p2n_cost = torch.sum(weights2 * p2p_cost, dim=3)

        # Step2: for each point in xyz1, find its neighbors in xyz1
        if knn_indices_1in1 is not None:
            assert knn_indices_1in1.shape == torch.Size([batch_size, n_points, self.k])
        else:
            knn_indices_1in1 = k_nearest_neighbor(input_xyz=xyz1, query_xyz=xyz1, k=self.k)  # [bs, n_points, k]
        # knn_xyz1: [bs, 3, n_points, k]
        knn_xyz1 = batch_indexing_channel_first(xyz1, knn_indices_1in1)
        # knn_xyz1_norm: [bs, 3, n_points, k]
        knn_xyz1_norm = knn_xyz1 - xyz1.view(batch_size, 3, n_points, 1)

        # weights1: [bs, out_channels, n_points, k]
        weights1 = self.weight_net1(knn_xyz1_norm)
        # n2n_cost: [bs, out_channels, n_points, k]
        n2n_cost = batch_indexing_channel_first(p2n_cost, knn_indices_1in1)
        # n2n_cost (neighbor-to-neighbor cost): [bs, out_channels, n_points]
        n2n_cost = torch.sum(weights1 * n2n_cost, dim=3)

        return n2n_cost


class FlowEstimator3D(nn.Module):
    def __init__(self, n_channels, norm=None, conv_last=True, k=16):
        super().__init__()
        self.point_conv1 = PointConvNoSampling(in_channels=n_channels[0], out_channels=n_channels[1], norm=norm, k=k)
        self.point_conv2 = PointConvNoSampling(in_channels=n_channels[1], out_channels=n_channels[2], norm=norm, k=k)
        self.mlp = MLP1d(n_channels[2], [n_channels[2], n_channels[3]])

        if conv_last:
            self.conv_last = nn.Conv1d(n_channels[3], 3, kernel_size=1)
        else:
            self.conv_last = None

    def forward(self, xyz, feat, knn_indices):
        """
        :param xyz: 3D locations of points, [batch_size, 3, n_points]
        :param feat: features of points, [batch_size, in_channels, n_points]
        :param knn_indices: knn indices of points, [batch_size, n_points, k]
        :return flow_feat: [batch_size, 64, n_points]
        :return flow: [batch_size, 3, n_points]
        """
        feat = self.point_conv1.forward(xyz, feat, knn_indices)  # [bs, 128, n_points]
        feat = self.point_conv2.forward(xyz, feat, knn_indices)  # [bs, 128, n_points]
        feat = self.mlp(feat)  # [bs, 64, n_points]

        if self.conv_last is not None:
            flow = self.conv_last(feat)  # [bs, 3, n_points]
            return feat, flow
        else:
            return feat
