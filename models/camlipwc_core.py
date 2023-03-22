import torch
import torch.nn as nn
from torch.nn.functional import leaky_relu, interpolate
from .clfm import CLFM
from .mlp import Conv1dNormRelu, Conv2dNormRelu
from .pwc_core import FeaturePyramid2D as FeaturePyramid2D
from .pwc_core import FlowEstimatorLite2D as FlowEstimatorLite2D
from .pwc_core import FlowEstimatorDense2D as FlowEstimatorDense2D
from .pwc_core import ContextNetwork2D as ContextNetwork2D
from .camlipwc_l_core import FeaturePyramid3D as FeaturePyramid3D
from .camlipwc_l_core import FlowEstimator3D as FlowEstimator3D
from .camlipwc_l_core import Correlation3D as Correlation3D
from .utils import project_pc2image, backwarp_2d, backwarp_3d, knn_interpolation, convex_upsample, mesh_grid
from .csrc import correlation2d, k_nearest_neighbor


class CamLiPWC_Core(nn.Module):
    def __init__(self, cfgs2d, cfgs3d, cfgs):
        super().__init__()

        self.cfgs, self.cfgs2d, self.cfgs3d = cfgs, cfgs2d, cfgs3d
        corr_channels_2d = (2 * cfgs2d.max_displacement + 1) ** 2

        # PWC-Net 2D
        self.branch_2d_fnet = FeaturePyramid2D(
            [3, 16, 32, 64, 96, 128, 192],
            norm=cfgs2d.norm.feature_pyramid
        )
        self.branch_2d_fnet_aligners = nn.ModuleList([
            nn.Identity(),
            Conv2dNormRelu(32, 64),  # 1/4
            Conv2dNormRelu(64, 64),
            Conv2dNormRelu(96, 64),
            Conv2dNormRelu(128, 64),
            Conv2dNormRelu(192, 64),
        ])
        if cfgs2d.lite_estimator:
            self.branch_2d_flow_estimator = FlowEstimatorLite2D(
                [64 + corr_channels_2d + 2 + 32, 128, 128, 96, 64, 32],
                norm=cfgs2d.norm.flow_estimator,
                conv_last=not cfgs.fuse_estimator,
            )
        else:
            self.branch_2d_flow_estimator = FlowEstimatorDense2D(
                [64 + corr_channels_2d + 2 + 32, 128, 128, 96, 64, 32],
                norm=cfgs2d.norm.flow_estimator,
                conv_last=not cfgs.fuse_estimator,
            )
        self.branch_2d_context_network = ContextNetwork2D(
            [self.branch_2d_flow_estimator.flow_feat_dim + 2, 128, 128, 128, 96, 64, 32],
            dilations=[1, 2, 4, 8, 16, 1],
            norm=cfgs2d.norm.context_network
        )
        self.branch_2d_up_mask_head = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 4 * 4 * 9, kernel_size=1, stride=1, padding=0),  # 1/4
        )

        # PWC-Net 3D
        self.branch_3d_fnet = FeaturePyramid3D(
            n_channels=[16, 32, 64, 96, 128, 192],  # 1/4
            norm=cfgs3d.norm.feature_pyramid,
            k=cfgs3d.k,
        )
        self.branch_3d_fnet_aligners = nn.ModuleList([
            nn.Identity(),
            Conv1dNormRelu(32, 64),  # 1/4
            Conv1dNormRelu(64, 64),
            Conv1dNormRelu(96, 64),
            Conv1dNormRelu(128, 64),
            Conv1dNormRelu(192, 64),
        ])
        self.branch_3d_correlations = nn.ModuleList([
            nn.Identity(),
            Correlation3D(32, 32, k=self.cfgs3d.k),  # 1/4
            Correlation3D(64, 64, k=self.cfgs3d.k),
            Correlation3D(96, 96, k=self.cfgs3d.k),
            Correlation3D(128, 128, k=self.cfgs3d.k),
            Correlation3D(192, 192, k=self.cfgs3d.k),
        ])
        self.branch_3d_correlation_aligners = nn.ModuleList([
            nn.Identity(),
            Conv1dNormRelu(32, 64),  # 1/4
            Conv1dNormRelu(64, 64),
            Conv1dNormRelu(96, 64),
            Conv1dNormRelu(128, 64),
            Conv1dNormRelu(192, 64),
        ])
        self.branch_3d_flow_estimator = FlowEstimator3D(
            [64 + 64 + 3 + 64, 128, 128, 64],
            cfgs3d.norm.flow_estimator,
            conv_last=not cfgs.fuse_estimator,
            k=self.cfgs3d.k,
        )

        if self.cfgs.fuse_pyramid:
            self.pyramid_clfms = nn.ModuleList([
                nn.Identity(),
                CLFM(32, 32, norm=cfgs2d.norm.feature_pyramid),  # 1/4
                CLFM(64, 64, norm=cfgs2d.norm.feature_pyramid),
                CLFM(96, 96, norm=cfgs2d.norm.feature_pyramid),
                CLFM(128, 128, norm=cfgs2d.norm.feature_pyramid),
                CLFM(192, 192, norm=cfgs2d.norm.feature_pyramid),
            ])

        if self.cfgs.fuse_correlation:
            self.corr_clfms = nn.ModuleList([
                nn.Identity(),
                CLFM(corr_channels_2d, 32),  # 1/4
                CLFM(corr_channels_2d, 64),
                CLFM(corr_channels_2d, 96),
                CLFM(corr_channels_2d, 128),
                CLFM(corr_channels_2d, 192),
            ])

        if self.cfgs.fuse_estimator:
            feat_dim_2d = self.branch_2d_flow_estimator.flow_feat_dim
            feat_dim_3d = self.branch_3d_flow_estimator.flow_feat_dim
            self.estimator_clfm = CLFM(feat_dim_2d, feat_dim_3d)
            self.branch_2d_conv_last = nn.Conv2d(feat_dim_2d, 2, kernel_size=3, stride=1, padding=1)
            self.branch_3d_conv_last = nn.Conv1d(feat_dim_3d, 3, kernel_size=1)

    def encode(self, image, xyzs):
        feats_2d = self.branch_2d_fnet(image)  # 1/4
        feats_3d = self.branch_3d_fnet(xyzs)
        return feats_2d, feats_3d

    def decode(self, xyzs1, xyzs2, feats1_2d, feats2_2d, feats1_3d, feats2_3d, camera_info):
        assert len(xyzs1) == len(xyzs2) == len(feats1_2d) == len(feats2_2d) == len(feats1_3d) == len(feats2_3d)

        flows_2d, flows_3d = [], []
        flow_feats_2d, flow_feats_3d = [], []
        
        for level in range(len(xyzs1) - 1, 0, -1):
            xyz1, feat1_2d, feat1_3d = xyzs1[level], feats1_2d[level], feats1_3d[level]
            xyz2, feat2_2d, feat2_3d = xyzs2[level], feats2_2d[level], feats2_3d[level]

            bs, image_h, image_w, n_points = feat1_2d.shape[0], feat1_2d.shape[2], feat1_2d.shape[3], xyz1.shape[-1]

            # project point cloud to image
            uv1 = project_pc2image(xyz1, camera_info)
            uv2 = project_pc2image(xyz2, camera_info)

            sensor_h, sensor_w = camera_info['sensor_h'], camera_info['sensor_w']
            uv1[:, 0] *= (image_w - 1) / (sensor_w - 1)
            uv1[:, 1] *= (image_h - 1) / (sensor_h - 1)
            uv2[:, 0] *= (image_w - 1) / (sensor_w - 1)
            uv2[:, 1] *= (image_h - 1) / (sensor_h - 1)

            # pre-compute knn indices
            grid = mesh_grid(bs, image_h, image_w, uv1.device)  # [B, 2, H, W]
            grid = grid.reshape([bs, 2, -1])  # [B, 2, HW]
            knn_xyz1 = k_nearest_neighbor(xyz1, xyz1, k=self.cfgs3d.k)  # [bs, n_points, k]

            # fuse encoder
            if self.cfgs.fuse_pyramid:
                feat1_2d, feat1_3d = self.pyramid_clfms[level](uv1, feat1_2d, feat1_3d)
                feat2_2d, feat2_3d = self.pyramid_clfms[level](uv2, feat2_2d, feat2_3d)

            if level == len(xyzs1) - 1:
                last_flow_2d = torch.zeros([bs, 2, image_h, image_w], dtype=uv1.dtype, device=uv1.device)
                last_feat_2d = torch.zeros([bs, 32, image_h, image_w], dtype=uv1.dtype, device=uv1.device)
                last_flow_3d = torch.zeros([bs, 3, n_points], dtype=uv1.dtype, device=uv1.device)
                last_feat_3d = torch.zeros([bs, 64, n_points], dtype=uv1.dtype, device=uv1.device)
                xyz2_warp, feat2_2d_warp = xyz2, feat2_2d
            else:
                last_flow_2d = interpolate(flows_2d[-1] * 2, scale_factor=2, mode='bilinear', align_corners=True)
                last_feat_2d = interpolate(flow_feats_2d[-1], scale_factor=2, mode='bilinear', align_corners=True)

                last_flow_3d, last_feat_3d = torch.split(
                    knn_interpolation(
                        xyzs1[level + 1],
                        torch.cat([flows_3d[-1], flow_feats_3d[-1]], dim=1),
                        xyz1
                    ), [3, 64], dim=1)

                feat2_2d_warp = backwarp_2d(feat2_2d, last_flow_2d, padding_mode='border')
                xyz2_warp = backwarp_3d(xyz1, xyz2, last_flow_3d)

            feat_corr_3d = self.branch_3d_correlations[level](xyz1, feat1_3d, xyz2_warp, feat2_3d, knn_xyz1)
            feat_corr_2d = leaky_relu(correlation2d(feat1_2d, feat2_2d_warp, self.cfgs2d.max_displacement), 0.1)

            # fuse correlation
            if self.cfgs.fuse_correlation:
                feat_corr_2d, feat_corr_3d = self.corr_clfms[level](uv1, feat_corr_2d, feat_corr_3d)

            feat1_2d = self.branch_2d_fnet_aligners[level](feat1_2d)
            feat1_3d = self.branch_3d_fnet_aligners[level](feat1_3d)
            feat_corr_3d = self.branch_3d_correlation_aligners[level](feat_corr_3d)

            x_2d = torch.cat([feat_corr_2d, feat1_2d, last_flow_2d, last_feat_2d], dim=1)
            x_3d = torch.cat([feat_corr_3d, feat1_3d, last_flow_3d, last_feat_3d], dim=1)

            # fuse decoder
            if self.cfgs.fuse_estimator:
                flow_feat_2d = self.branch_2d_flow_estimator(x_2d)
                flow_feat_3d = self.branch_3d_flow_estimator(xyz1, x_3d, knn_xyz1)

                flow_feat_2d, flow_feat_3d = self.estimator_clfm(uv1, flow_feat_2d, flow_feat_3d)

                flow_delta_2d = self.branch_2d_conv_last(flow_feat_2d)
                flow_delta_3d = self.branch_3d_conv_last(flow_feat_3d)
            else:
                flow_feat_2d, flow_delta_2d = self.branch_2d_flow_estimator(x_2d)
                flow_feat_3d, flow_delta_3d = self.branch_3d_flow_estimator(xyz1, x_3d, knn_xyz1)

            # residual connection
            flow_2d = last_flow_2d + flow_delta_2d
            flow_3d = last_flow_3d + flow_delta_3d

            # context network (2D only)
            flow_feat_2d, flow_delta_2d = self.branch_2d_context_network(torch.cat([flow_feat_2d, flow_2d], dim=1))
            flow_2d = flow_delta_2d + flow_2d

            # clip
            flow_2d = torch.clip(flow_2d, min=-1000, max=1000)
            flow_3d = torch.clip(flow_3d, min=-100, max=100)

            # save results
            flows_2d.append(flow_2d)
            flows_3d.append(flow_3d)
            flow_feats_2d.append(flow_feat_2d)
            flow_feats_3d.append(flow_feat_3d)

        flows_2d = [f.float() for f in flows_2d][::-1]
        flows_3d = [f.float() for f in flows_3d][::-1]

        flows_2d[0] = convex_upsample(flows_2d[0], self.branch_2d_up_mask_head(flow_feat_2d), scale_factor=4)  # 1/4

        for i in range(1, len(flows_2d)):
            flows_2d[i] = interpolate(flows_2d[i] * 4, scale_factor=4, mode='bilinear', align_corners=True)  # 1/4

        for i in range(len(flows_3d)):
            flows_3d[i] = knn_interpolation(xyzs1[i + 1], flows_3d[i], xyzs1[i])

        return flows_2d, flows_3d
