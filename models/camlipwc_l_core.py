import torch
import torch.nn as nn
from .point_conv import PointConv
from .utils import backwarp_3d, knn_interpolation, batch_indexing, k_nearest_neighbor, timer
from .mlp import MLP1d, MLP2d, Conv1dNormRelu


class FeaturePyramid3D(nn.Module):
    def __init__(self, n_channels, norm=None, k=16):
        super().__init__()

        self.level0_mlp = MLP1d(3, [n_channels[0], n_channels[0]])

        self.pyramid_mlps = nn.ModuleList()
        self.pyramid_convs = nn.ModuleList()

        for i in range(len(n_channels) - 1):
            self.pyramid_mlps.append(MLP1d(n_channels[i], [n_channels[i], n_channels[i + 1]]))
            self.pyramid_convs.append(PointConv(n_channels[i + 1], n_channels[i + 1], norm=norm, k=k))

    @timer.timer_func
    def forward(self, xyzs):
        """
        :param xyzs: pyramid of points
        :return feats: pyramid of features
        """
        assert len(xyzs) == len(self.pyramid_mlps) + 1

        inputs = xyzs[0] # [bs, 3, n_points]
        feats = [self.level0_mlp(inputs)]

        for i in range(len(xyzs) - 1):
            feat = self.pyramid_mlps[i](feats[-1])
            feats.append(self.pyramid_convs[i](xyzs[i], feat, xyzs[i + 1]))

        return feats


class Correlation3D(nn.Module):
    def __init__(self, in_channels, out_channels, align_channels=None, k=16):
        super().__init__()
        self.k = k

        self.cost_mlp = MLP2d(3 + 2 * in_channels, [out_channels, out_channels], act='leaky_relu')
        self.weight_net1 = MLP2d(3, [8, 8, out_channels], act='relu')
        self.weight_net2 = MLP2d(3, [8, 8, out_channels], act='relu')

        if align_channels is not None:
            self.feat_aligner = Conv1dNormRelu(out_channels, align_channels)
        else:
            self.feat_aligner = nn.Identity()

    @timer.timer_func
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
        knn_xyz2 = batch_indexing(xyz2, knn_indices_1in2)
        # knn_xyz2_norm: [bs, 3, n_points, k]
        knn_xyz2_norm = knn_xyz2 - xyz1.view(batch_size, 3, n_points, 1)
        # knn_features2: [bs, in_channels, n_points, k]
        knn_features2 = batch_indexing(feat2, knn_indices_1in2)
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
            assert knn_indices_1in1.shape[:2] == torch.Size([batch_size, n_points])
            assert knn_indices_1in1.shape[2] >= self.k
            knn_indices_1in1 = knn_indices_1in1[:, :, :self.k]
        else:
            knn_indices_1in1 = k_nearest_neighbor(input_xyz=xyz1, query_xyz=xyz1, k=self.k)  # [bs, n_points, k]
        # knn_xyz1: [bs, 3, n_points, k]
        knn_xyz1 = batch_indexing(xyz1, knn_indices_1in1)
        # knn_xyz1_norm: [bs, 3, n_points, k]
        knn_xyz1_norm = knn_xyz1 - xyz1.view(batch_size, 3, n_points, 1)

        # weights1: [bs, out_channels, n_points, k]
        weights1 = self.weight_net1(knn_xyz1_norm)
        # n2n_cost: [bs, out_channels, n_points, k]
        n2n_cost = batch_indexing(p2n_cost, knn_indices_1in1)
        # n2n_cost (neighbor-to-neighbor cost): [bs, out_channels, n_points]
        n2n_cost = torch.sum(weights1 * n2n_cost, dim=3)
        # align features (optional)
        n2n_cost = self.feat_aligner(n2n_cost)

        return n2n_cost


class FlowEstimator3D(nn.Module):
    def __init__(self, n_channels, norm=None, conv_last=True, k=16):
        super().__init__()
        self.point_conv1 = PointConv(in_channels=n_channels[0], out_channels=n_channels[1], norm=norm, k=k)
        self.point_conv2 = PointConv(in_channels=n_channels[1], out_channels=n_channels[2], norm=norm, k=k)
        self.mlp = MLP1d(n_channels[2], [n_channels[2], n_channels[3]])
        self.flow_feat_dim = n_channels[3]

        if conv_last:
            self.conv_last = nn.Conv1d(n_channels[3], 3, kernel_size=1)
        else:
            self.conv_last = None

    @timer.timer_func
    def forward(self, xyz, feat, knn_indices):
        """
        :param xyz: 3D locations of points, [batch_size, 3, n_points]
        :param feat: features of points, [batch_size, in_channels, n_points]
        :param knn_indices: knn indices of points, [batch_size, n_points, k]
        :return flow_feat: [batch_size, 64, n_points]
        :return flow: [batch_size, 3, n_points]
        """
        feat = self.point_conv1(xyz, feat, knn_indices=knn_indices)
        feat = self.point_conv2(xyz, feat, knn_indices=knn_indices)
        feat = self.mlp(feat)

        if self.conv_last is not None:
            flow = self.conv_last(feat)
            return feat, flow
        else:
            return feat


class CamLiPWC_L_Core(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.cfgs = cfgs

        self.feature_pyramid = FeaturePyramid3D(
            n_channels=[16, 32, 64, 96, 128, 192],
            norm=cfgs.norm.feature_pyramid,
        )
        self.correlations = nn.ModuleList([
            nn.Identity(),
            Correlation3D(32, 32, 64),
            Correlation3D(64, 64, 64),
            Correlation3D(96, 96, 64),
            Correlation3D(128, 128, 64),
            Correlation3D(192, 192, 64),
        ])
        self.pyramid_feat_aligners = nn.ModuleList([
            nn.Identity(),
            Conv1dNormRelu(32, 64),
            Conv1dNormRelu(64, 64),
            Conv1dNormRelu(96, 64),
            Conv1dNormRelu(128, 64),
            Conv1dNormRelu(192, 64),
        ])
        self.flow_estimator = FlowEstimator3D(
            n_channels=[64 + 64 + 3, 128, 128, 64],
            norm=cfgs.norm.flow_estimator,
        )

    def encode(self, xyzs):
        feats_3d = self.feature_pyramid(xyzs)
        return feats_3d

    def decode(self, xyzs1, xyzs2, feats1_3d, feats2_3d):
        flows_3d = []

        for level in range(len(xyzs1) - 1, 0, -1):
            xyz1, feat1_3d = xyzs1[level], feats1_3d[level]
            xyz2, feat2_3d = xyzs2[level], feats2_3d[level]
            knn1 = k_nearest_neighbor(xyz1, xyz1, k=16)

            bs, _, n_points = xyz1.shape

            if level == len(xyzs1) - 1:
                last_flow_3d = torch.zeros([bs, 3, n_points], device=xyz1.device)
                xyz2_warp = xyz2
            else:
                last_flow_3d = knn_interpolation(xyzs1[level + 1], flows_3d[-1], xyz1)
                xyz2_warp = backwarp_3d(xyz1, xyz2, last_flow_3d)

            # estimate scene flow
            x = torch.cat([
                self.pyramid_feat_aligners[level](feat1_3d),
                self.correlations[level](xyz1, feat1_3d, xyz2_warp, feat2_3d, knn1),
                last_flow_3d
            ], dim=1)

            _, flow_delta_3d = self.flow_estimator(xyz1, x, knn1)
            flow_3d = last_flow_3d + flow_delta_3d

            flows_3d.append(flow_3d)
        
        flows_3d = [f.float() for f in flows_3d][::-1]
        
        for i in range(len(flows_3d)):
            flows_3d[i] = knn_interpolation(xyzs1[i + 1], flows_3d[i], xyzs1[i])

        return flows_3d
