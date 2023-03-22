import torch
import torch.nn as nn
from .point_conv import PointConvDW, PointConv
from .utils import backwarp_3d, batch_indexing, knn_interpolation, k_nearest_neighbor, build_pc_pyramid, timer
from .mlp import Conv1dNormRelu, MLP1d, MLP2d


class Encoder3D(nn.Module):
    def __init__(self, n_channels, norm=None, k=16):
        super().__init__()

        self.level0_mlp = MLP1d(3, [n_channels[0], n_channels[0]])

        self.mlps = nn.ModuleList()
        self.convs = nn.ModuleList()

        for i in range(len(n_channels) - 1):
            self.mlps.append(MLP1d(n_channels[i], [n_channels[i], n_channels[i + 1]]))
            self.convs.append(PointConv(n_channels[i + 1], n_channels[i + 1], norm=norm, k=k))

    @timer.timer_func
    def forward(self, xyzs):
        """
        :param xyzs: pyramid of points
        :return feats: pyramid of features
        """
        assert len(xyzs) == len(self.mlps) + 1

        inputs = xyzs[0]  # [bs, 3, n_points]
        feats = [self.level0_mlp(inputs)]

        for i in range(len(xyzs) - 1):
            feat = self.mlps[i](feats[-1])
            feat = self.convs[i](xyzs[i], feat, xyzs[i + 1])
            feats.append(feat)

        return feats


class Correlation3D(nn.Module):
    def __init__(self, out_channels, k=16):
        super().__init__()
        self.k = k

        self.cost_mlp = MLP2d(4, [out_channels // 4, out_channels // 4], act='relu')
        self.merge = Conv1dNormRelu(out_channels, out_channels)  # ?

        # cost volume is built during runtime
        self.cost_volume_pyramid = None

    def build_cost_volume_pyramid(self, feat1, feat2, xyzs2, k=3):
        cost_volume = torch.bmm(feat1.float().transpose(1, 2), feat2.float())  # [B, N, N]
        cost_volume = cost_volume / feat1.shape[1]
        self.cost_volume_pyramid = [cost_volume]  # [B, N, M0]

        for i in range(1, len(xyzs2)):
            knn_indices = k_nearest_neighbor(xyzs2[i - 1], xyzs2[i], k=k)
            knn_corr = batch_indexing(self.cost_volume_pyramid[i - 1], knn_indices)
            avg_corr = torch.mean(knn_corr, dim=-1)
            self.cost_volume_pyramid.append(avg_corr)

    def calc_matching_cost(self, xyz1, xyz2, cost_volume):
        bs, n_points1, n_points2 = cost_volume.shape

        # for each point in xyz1, find its neighbors in xyz2
        knn_indices_cross = k_nearest_neighbor(input_xyz=xyz2, query_xyz=xyz1, k=self.k)  # [bs, n_points, k]
        knn_xyz2 = batch_indexing(xyz2, knn_indices_cross)
        knn_xyz2_norm = knn_xyz2 - xyz1.view(bs, 3, n_points1, 1)
        
        knn_corr = batch_indexing(
            cost_volume.reshape(bs * n_points1, n_points2),
            knn_indices_cross.reshape(bs * n_points1, self.k),
            layout='channel_last'
        ).reshape(bs, 1, n_points1, self.k)

        cost = self.cost_mlp(torch.cat([knn_xyz2_norm, knn_corr], dim=1))
        cost = torch.sum(cost, dim=-1)

        return cost

    @timer.timer_func
    def forward(self, xyz1, xyzs2):
        """
        :param xyz1: [batch_size, 3, n_points]
        :param feat1: [batch_size, in_channels, n_points]
        :param xyz2: [batch_size, 3, n_points]
        :param feat2: [batch_size, in_channels, n_points]
        :param knn_indices_1in1: for each point in xyz1, find its neighbors in xyz1, [batch_size, n_points, k]
        :return cost volume for each point in xyz1: [batch_size, n_cost_channels, n_points]
        """
        # compute single-scale matching cost
        cost0 = self.calc_matching_cost(xyz1, xyzs2[0], self.cost_volume_pyramid[0])
        cost1 = self.calc_matching_cost(xyz1, xyzs2[1], self.cost_volume_pyramid[1])
        cost2 = self.calc_matching_cost(xyz1, xyzs2[2], self.cost_volume_pyramid[2])
        cost3 = self.calc_matching_cost(xyz1, xyzs2[3], self.cost_volume_pyramid[3])

        # merge multi-scale costs
        costs = torch.cat([cost0, cost1, cost2, cost3], dim=1)
        costs = self.merge(costs)

        return costs


class FlowHead3D(nn.Module):
    def __init__(self, input_dim=128):
        super().__init__()
        self.conv1 = PointConvDW(input_dim, 128, k=32)  # ?
        self.conv2 = PointConvDW(128, 64, k=32)  # ?
        self.fc = nn.Conv1d(64, 3, kernel_size=1)

    @timer.timer_func
    def forward(self, xyz, features, knn_indices=None):
        features = features.float()
        features = self.conv1(xyz, features, knn_indices=knn_indices)
        features = self.conv2(xyz, features, knn_indices=knn_indices)
        return self.fc(features)


class GRU3D(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.conv_z = PointConvDW(hidden_dim + input_dim, hidden_dim, act=None, k=4)
        self.conv_r = PointConvDW(hidden_dim + input_dim, hidden_dim, act=None, k=4)
        self.conv_q = PointConvDW(hidden_dim + input_dim, hidden_dim, act=None, k=4)

    @timer.timer_func
    def forward(self, xyz, h, x, knn_indices=None):
        h, x = h.float(), x.float()
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.conv_z(xyz, hx, knn_indices=knn_indices))
        r = torch.sigmoid(self.conv_r(xyz, hx, knn_indices=knn_indices))
        q = torch.tanh(self.conv_q(xyz, torch.cat([r * h, x], dim=1), knn_indices=knn_indices))
        h = (1 - z) * h + z * q
        return h


class MotionEncoder3D(nn.Module):
    def __init__(self, corr_dim=128):
        super(MotionEncoder3D, self).__init__()
        self.conv_c1 = PointConvDW(corr_dim, corr_dim)
        self.conv_f1 = PointConvDW(3, 32, k=32)
        self.conv_f2 = PointConvDW(32, 16, k=16)
        self.conv = PointConvDW(corr_dim + 16, 128 - 3, k=16)  # ?

    @timer.timer_func
    def forward(self, xyz, flow, corr, knn_indices):
        corr, flow = corr.float(), flow.float()
        corr_feat = self.conv_c1(xyz, corr, knn_indices=knn_indices)
        flow_feat = self.conv_f1(xyz, flow, knn_indices=knn_indices)
        flow_feat = self.conv_f2(xyz, flow_feat, knn_indices=knn_indices)

        corr_flow_feat = torch.cat([corr_feat, flow_feat], dim=1)
        out = self.conv(xyz, corr_flow_feat, knn_indices=knn_indices)

        return torch.cat([out, flow], dim=1)


class CamLiRAFT_L_Core(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.cfgs = cfgs

        self.fnet = Encoder3D(n_channels=[64, 96, 128], norm='batch_norm', k=16)
        self.cnet = Encoder3D(n_channels=[64, 96, 128], norm='batch_norm', k=16)
        self.cnet_aligner = nn.Conv1d(128, 256, kernel_size=1)
        self.correlation = Correlation3D(out_channels=128, k=16)
        self.motion_encoder = MotionEncoder3D(corr_dim=128)
        self.gru = GRU3D(input_dim=128 + 128, hidden_dim=128)
        self.flow_head = FlowHead3D(input_dim=128)

    def forward(self, pc1, pc2):
        flow_preds = []

        xyzs1, xyzs2, _, _ = build_pc_pyramid(
            pc1, pc2, [4096, 2048, 1024, 512, 256]
        )

        feat1 = self.fnet(xyzs1[:3])[2]
        feat2 = self.fnet(xyzs2[:3])[2]
        featc = self.cnet(xyzs1[:3])[2]
        featc = self.cnet_aligner(featc)

        xyzs1, xyzs2 = xyzs1[2:], xyzs2[2:]
        xyz1, xyz2 = xyzs1[0], xyzs2[0]  # 2048

        self.correlation.build_cost_volume_pyramid(feat1, feat2, xyzs2)

        h, x = torch.split(featc, [128, 128], dim=1)
        h = torch.tanh(h)
        x = torch.relu(x)
        
        knn_indices = k_nearest_neighbor(xyz1, xyz1, k=32)

        if self.training:
            n_iters = self.cfgs.n_iters_train
        else:
            n_iters = self.cfgs.n_iters_eval

        for it in range(n_iters):
            if it > 0:
                flow_pred = flow_pred.detach()
                xyzs2_warp = [backwarp_3d(xyz1, xyz2_lvl, flow_pred) for xyz2_lvl in xyzs2]
            else:
                flow_pred = torch.zeros_like(xyz1)
                xyzs2_warp = xyzs2

            # correlation
            corr = self.correlation(xyz1, xyzs2_warp)

            # motion feat: corr + flow
            motion_feat = self.motion_encoder(xyz1, flow_pred, corr, knn_indices=knn_indices)

            # GRU
            h = self.gru(xyz1, h=h, x=torch.cat([x, motion_feat], dim=1), knn_indices=knn_indices)

            # predict delta flow
            flow_delta = self.flow_head(xyz1, h, knn_indices)
            flow_pred = flow_pred + flow_delta.float()
            
            flow_preds.append(flow_pred)
        
        for i in range(len(flow_preds)):
            flow_preds[i] = knn_interpolation(xyz1, flow_preds[i], pc1, k=3)

        return flow_preds
