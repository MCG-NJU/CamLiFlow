import torch
import torch.nn as nn
from .utils import grid_sample_wrapper, mesh_grid, k_nearest_neighbor, batch_indexing, softmax, timer
from .mlp import Conv1dNormRelu, Conv2dNormRelu


class CLFM(nn.Module):
    def __init__(self, in_channels_2d, in_channels_3d, fusion_fn='sk', norm=None):
        super().__init__()

        self.interp = FusionAwareInterp(in_channels_3d, k=1, norm=norm)
        self.mlps3d = Conv1dNormRelu(in_channels_2d, in_channels_2d, norm=norm)

        if fusion_fn == 'add':
            self.fuse2d = AddFusion(in_channels_2d, in_channels_3d, in_channels_2d, 'nchw', norm)
            self.fuse3d = AddFusion(in_channels_2d, in_channels_3d, in_channels_3d, 'ncm', norm)
        elif fusion_fn == 'concat':
            self.fuse2d = ConcatFusion(in_channels_2d, in_channels_3d, in_channels_2d, 'nchw', norm)
            self.fuse3d = ConcatFusion(in_channels_2d, in_channels_3d, in_channels_3d, 'ncm', norm)
        elif fusion_fn == 'gated':
            self.fuse2d = GatedFusion(in_channels_2d, in_channels_3d, in_channels_2d, 'nchw', norm)
            self.fuse3d = GatedFusion(in_channels_2d, in_channels_3d, in_channels_3d, 'ncm', norm)
        elif fusion_fn == 'sk':
            self.fuse2d = SKFusion(in_channels_2d, in_channels_3d, in_channels_2d, 'nchw', norm, reduction=2)
            self.fuse3d = SKFusion(in_channels_2d, in_channels_3d, in_channels_3d, 'ncm', norm, reduction=2)
        else:
            raise ValueError

    @timer.timer_func
    def forward(self, uv, feat_2d, feat_3d):
        feat_2d = feat_2d.float()
        feat_3d = feat_3d.float()

        feat_3d_interp = self.interp(uv, feat_2d.detach(), feat_3d.detach())
        out2d = self.fuse2d(feat_2d, feat_3d_interp)

        feat_2d_sampled = grid_sample_wrapper(feat_2d.detach(), uv)
        out3d = self.fuse3d(self.mlps3d(feat_2d_sampled.detach()), feat_3d)

        return out2d, out3d


class FusionAwareInterp(nn.Module):
    def __init__(self, n_channels_3d, k=1, norm=None):
        super().__init__()
        self.k = k
        self.out_conv = Conv2dNormRelu(n_channels_3d, n_channels_3d, norm=norm)
        self.score_net = nn.Sequential(
            Conv2dNormRelu(3, 16),  # [dx, dy, |dx, dy|_2, sim]
            Conv2dNormRelu(16, n_channels_3d, act='sigmoid'),
        )

    def forward(self, uv, feat_2d, feat_3d):
        bs, _, image_h, image_w = feat_2d.shape
        n_channels_3d = feat_3d.shape[1]

        grid = mesh_grid(bs, image_h, image_w, uv.device)  # [B, 2, H, W]
        grid = grid.reshape([bs, 2, -1])  # [B, 2, HW]

        knn_indices = k_nearest_neighbor(uv, grid, self.k)  # [B, HW, k]

        knn_uv, knn_feat3d = torch.split(
            batch_indexing(
                torch.cat([uv, feat_3d], dim=1),
                knn_indices
            ), [2, n_channels_3d], dim=1)

        knn_offset = knn_uv - grid[..., None]  # [B, 2, HW, k]
        knn_offset_norm = torch.linalg.norm(knn_offset, dim=1, keepdim=True)  # [B, 1, HW, k]

        score_input = torch.cat([knn_offset, knn_offset_norm], dim=1)  # [B, 4, HW, K]
        score = self.score_net(score_input)  # [B, n_channels_3d, HW, k]
        # score = softmax(score, dim=-1)  # [B, n_channels_3d, HW, k]

        final = score * knn_feat3d  # [B, n_channels_3d, HW, k]
        final = final.sum(dim=-1).reshape(bs, -1, image_h, image_w)  # [B, n_channels_3d, H, W]
        final = self.out_conv(final)

        return final


class FusionAwareInterpCVPR(nn.Module):
    def __init__(self, n_channels_2d, n_channels_3d, k=3, norm=None) -> None:
        super().__init__()

        self.mlps = nn.Sequential(
            Conv2dNormRelu(n_channels_3d + 3, n_channels_3d, norm=norm),
            Conv2dNormRelu(n_channels_3d, n_channels_3d, norm=norm),
            Conv2dNormRelu(n_channels_3d, n_channels_3d, norm=norm),
        )

    def forward(self, uv, feat_2d, feat_3d):
        bs, _, h, w = feat_2d.shape

        grid = mesh_grid(bs, h, w, uv.device)  # [B, 2, H, W]
        grid = grid.reshape([bs, 2, -1])  # [B, 2, HW]

        with torch.no_grad():
            nn_indices = k_nearest_neighbor(uv, grid, k=1)[..., 0]  # [B, HW]
            nn_feat2d = batch_indexing(grid_sample_wrapper(feat_2d, uv), nn_indices)  # [B, n_channels_2d, HW]
            nn_feat3d = batch_indexing(feat_3d, nn_indices)  # [B, n_channels_3d, HW]
            nn_offset = batch_indexing(uv, nn_indices) - grid  # [B, 2, HW]
            nn_corr = torch.mean(nn_feat2d * feat_2d.reshape(bs, -1, h * w), dim=1, keepdim=True)  # [B, 1, HW]

        feat = torch.cat([nn_offset, nn_corr, nn_feat3d], dim=1)  # [B, n_channels_3d + 3, HW]
        feat = feat.reshape([bs, -1, h, w])  # [B, n_channels_3d + 3, H, W]
        final = self.mlps(feat)  # [B, n_channels_3d, H, W]

        return final


class AddFusion(nn.Module):
    def __init__(self, in_channels_2d, in_channels_3d, out_channels, feat_format, norm=None):
        super().__init__()

        if feat_format == 'nchw':
            self.align1 = Conv2dNormRelu(in_channels_2d, out_channels, norm=norm)
            self.align2 = Conv2dNormRelu(in_channels_3d, out_channels, norm=norm)
        elif feat_format == 'ncm':
            self.align1 = Conv1dNormRelu(in_channels_2d, out_channels, norm=norm)
            self.align2 = Conv1dNormRelu(in_channels_3d, out_channels, norm=norm)
        else:
            raise ValueError

        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, feat_2d, feat_3d):
        return self.relu(self.align1(feat_2d) + self.align2(feat_3d))


class ConcatFusion(nn.Module):
    def __init__(self, in_channels_2d, in_channels_3d, out_channels, feat_format, norm=None):
        super().__init__()

        if feat_format == 'nchw':
            self.mlp = Conv2dNormRelu(in_channels_2d + in_channels_3d, out_channels, norm=norm)
        elif feat_format == 'ncm':
            self.mlp = Conv1dNormRelu(in_channels_2d + in_channels_3d, out_channels, norm=norm)
        else:
            raise ValueError

    def forward(self, feat_2d, feat_3d):
        return self.mlp(torch.cat([feat_2d, feat_3d], dim=1))


class GatedFusion(nn.Module):
    def __init__(self, in_channels_2d, in_channels_3d, out_channels, feat_format, norm=None):
        super().__init__()

        if feat_format == 'nchw':
            self.align1 = Conv2dNormRelu(in_channels_2d, out_channels, norm=norm)
            self.align2 = Conv2dNormRelu(in_channels_3d, out_channels, norm=norm)
            self.mlp1 = Conv2dNormRelu(out_channels, 2, norm=None, act='sigmoid')
            self.mlp2 = Conv2dNormRelu(out_channels, 2, norm=None, act='sigmoid')
        elif feat_format == 'ncm':
            self.align1 = Conv1dNormRelu(in_channels_2d, out_channels, norm=norm)
            self.align2 = Conv1dNormRelu(in_channels_3d, out_channels, norm=norm)
            self.mlp1 = Conv1dNormRelu(out_channels, 2, norm=None, act='sigmoid')
            self.mlp2 = Conv1dNormRelu(out_channels, 2, norm=None, act='sigmoid')
        else:
            raise ValueError

    def forward(self, feat_2d, feat_3d):
        feat_2d = self.align1(feat_2d)
        feat_3d = self.align2(feat_3d)
        weight = self.mlp1(feat_2d) + self.mlp2(feat_3d)  # [N, 2, H, W]
        weight = softmax(weight, dim=1)  # [N, 2, H, W]
        return feat_2d * weight[:, 0:1] + feat_3d * weight[:, 1:2]


class SKFusion(nn.Module):
    def __init__(self, in_channels_2d, in_channels_3d, out_channels, feat_format, norm=None, reduction=1):
        super().__init__()

        if feat_format == 'nchw':
            self.align1 = Conv2dNormRelu(in_channels_2d, out_channels, norm=norm)
            self.align2 = Conv2dNormRelu(in_channels_3d, out_channels, norm=norm)
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        elif feat_format == 'ncm':
            self.align1 = Conv1dNormRelu(in_channels_2d, out_channels, norm=norm)
            self.align2 = Conv1dNormRelu(in_channels_3d, out_channels, norm=norm)
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
        else:
            raise ValueError

        self.fc_mid = nn.Sequential(
            nn.Linear(out_channels, out_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
        )
        self.fc_out = nn.Sequential(
            nn.Linear(out_channels // reduction, out_channels * 2, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, feat_2d, feat_3d):
        bs = feat_2d.shape[0]

        feat_2d = self.align1(feat_2d)
        feat_3d = self.align2(feat_3d)

        weight = self.avg_pool(feat_2d + feat_3d).reshape([bs, -1])  # [bs, C]
        weight = self.fc_mid(weight)  # [bs, C / r]
        weight = self.fc_out(weight).reshape([bs, -1, 2])  # [bs, C, 2]
        weight = softmax(weight, dim=-1)
        w1, w2 = weight[..., 0], weight[..., 1]  # [bs, C]

        if len(feat_2d.shape) == 4:
            w1 = w1.reshape([bs, -1, 1, 1])
            w2 = w2.reshape([bs, -1, 1, 1])
        else:
            w1 = w1.reshape([bs, -1, 1])
            w2 = w2.reshape([bs, -1, 1])

        return feat_2d * w1 + feat_3d * w2
