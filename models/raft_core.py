import torch
import logging
import torch.nn as nn
from torch.nn.functional import grid_sample, avg_pool2d
from mmdet.models.backbones import ResNet
from .utils import convex_upsample, mesh_grid, timer
from .mlp import Conv2dNormRelu


class Encoder2D(ResNet):
    def __init__(self, depth=50, pretrained=None):
        super().__init__(
            depth=depth,
            num_stages=2,
            strides=(1, 2),
            dilations=(1, 1),
            out_indices=(1,),
            norm_eval=True,
            with_cp=False,
            init_cfg=dict(
                type='Pretrained',
                checkpoint=pretrained
            )
        )
        self.align = Conv2dNormRelu(self.feat_dim, 128)

        # MMCV, please shut up
        from mmcv.utils.logging import get_logger
        get_logger('root').setLevel(logging.ERROR)
        get_logger('mmcv').setLevel(logging.ERROR)

        self.init_weights()
    
    @timer.timer_func
    def forward(self, x):
        x = super().forward(x)[0]
        x = self.align(x)
        return x


class Correlation2D(nn.Module):
    def __init__(self, num_levels=4, radius=4):
        super().__init__()

        self.num_levels = num_levels
        self.radius = radius
        self.fnet_aligner = nn.Conv2d(128, 256, kernel_size=1)
        
        # cost volume pyramid is built during runtime
        self.cost_volume_pyramid = None

    def build_cost_volume_pyramid(self, fmap1, fmap2):
        fmap1 = self.fnet_aligner(fmap1.float())
        fmap2 = self.fnet_aligner(fmap2.float())
        
        # all pairs correlation
        bs, dim, h, w = fmap1.shape
        fmap1 = fmap1.view(bs, dim, h * w)
        fmap2 = fmap2.view(bs, dim, h * w)

        cost_volume = torch.matmul(fmap1.transpose(1, 2), fmap2)
        cost_volume = cost_volume / torch.sqrt(torch.tensor(dim))
        cost_volume = cost_volume.reshape(bs * h * w, 1, h, w)

        self.cost_volume_pyramid = [cost_volume]
        for _ in range(self.num_levels-1):
            cost_volume = avg_pool2d(cost_volume, 2, stride=2)
            self.cost_volume_pyramid.append(cost_volume)

    @timer.timer_func
    def forward(self, coords):
        coords = coords.permute(0, 2, 3, 1).float()
        bs, h, w, _ = coords.shape
        r = self.radius

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.cost_volume_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1, device=coords.device)
            dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing='ij'), axis=-1)

            centroid_lvl = coords.reshape(bs*h*w, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = self.bilinear_sampler(corr, coords_lvl)
            corr = corr.view(bs, h, w, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        out = out.permute(0, 3, 1, 2).contiguous()

        return out
    
    @staticmethod
    def bilinear_sampler(feat, coords):
        """ Wrapper for grid_sample, uses pixel coordinates """
        h, w = feat.shape[-2:]
        xgrid, ygrid = coords.split([1, 1], dim=-1)
        xgrid = 2 * xgrid / (w - 1) - 1
        ygrid = 2 * ygrid / (h - 1) - 1

        grid = torch.cat([xgrid, ygrid], dim=-1)
        feat = grid_sample(feat, grid, align_corners=True)

        return feat


class GRU2D(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(GRU2D, self).__init__()

        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1, 5), padding=(0, 2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5, 1), padding=(2, 0))

    @timer.timer_func
    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        h = torch.nan_to_num(h)
        return h


class MotionEncoder2D(nn.Module):
    def __init__(self, corr_levels, corr_radius):
        super(MotionEncoder2D, self).__init__()
        corr_planes = corr_levels * (2 * corr_radius + 1) ** 2

        self.conv_c1 = nn.Conv2d(corr_planes, 256, kernel_size=1, padding=0)
        self.conv_c2 = nn.Conv2d(256, 192, kernel_size=3, padding=1)
        self.conv_f1 = nn.Conv2d(2, 128, kernel_size=7, padding=3)
        self.conv_f2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv = nn.Conv2d(64 + 192, 128 - 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    @timer.timer_func
    def forward(self, flow, corr):
        corr_feat = self.relu(self.conv_c1(corr))
        corr_feat = self.relu(self.conv_c2(corr_feat))
        flow_feat = self.relu(self.conv_f1(flow))
        flow_feat = self.relu(self.conv_f2(flow_feat))

        out = self.relu(self.conv(
            torch.cat([corr_feat, flow_feat], dim=1)
        ))
        out = torch.nan_to_num(out)

        return torch.cat([out, flow], dim=1)


class FlowHead2D(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead2D, self).__init__()

        self.conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    @timer.timer_func
    def forward(self, x):
        out = self.conv2(self.relu(self.conv1(x))).float()
        out = torch.nan_to_num(out)
        return out


class ConvexUpsampler2D(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mask = nn.Sequential(
            nn.Conv2d(input_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0)
        )

    @timer.timer_func
    def forward(self, h, flow):
        # scale mask to balance gradients
        up_mask = 0.25 * self.mask(h.float())
        return convex_upsample(flow, up_mask)


class RAFTCore(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.cfgs = cfgs

        self.hidden_dim = 128
        self.context_dim = 128
        self.corr_levels = 4
        self.corr_radius = 4

        # feature network, context network, and update block
        self.fnet = Encoder2D(cfgs.backbone.depth, cfgs.backbone.pretrained)
        self.cnet = Encoder2D(cfgs.backbone.depth, cfgs.backbone.pretrained)

        self.cnet_aligner = nn.Conv2d(128, 256, kernel_size=1)
        self.correlation = Correlation2D(self.corr_levels, self.corr_radius)
        self.motion_encoder = MotionEncoder2D(self.corr_levels, self.corr_radius)
        self.gru = GRU2D(hidden_dim=self.hidden_dim, input_dim=self.hidden_dim + 128)
        self.flow_head = FlowHead2D(self.hidden_dim)
        self.convex_upsampler = ConvexUpsampler2D(self.hidden_dim)

    def forward(self, image1, image2):
        # run the feature network
        fmap1 = self.fnet(image1)
        fmap2 = self.fnet(image2)

        # all-pair correlation
        self.correlation.build_cost_volume_pyramid(fmap1, fmap2)

        # run the context network
        cnet = self.cnet(image1)
        cnet = self.cnet_aligner(cnet)
        h, x = torch.split(cnet, [self.hidden_dim, self.context_dim], dim=1)
        h = torch.tanh(h)
        x = torch.relu(x)

        bs, _, image_h, image_w = image1.shape
        grid_coords = mesh_grid(bs, image_h//8, image_w//8, device=image1.device)

        flow_preds = []
        flow_pred = torch.zeros_like(grid_coords)

        if self.training:
            n_iters = self.cfgs.n_iters_train
        else:
            n_iters = self.cfgs.n_iters_eval

        for _ in range(n_iters):
            flow_pred = flow_pred.detach()
            
            # index correlation volume
            corr = self.correlation(grid_coords + flow_pred)

            # motion features: previous flow with current correlation
            motion_features = self.motion_encoder(flow_pred, corr)

            # run GRU
            h = self.gru(h, torch.cat([x, motion_features], dim=1))

            # predict delta flow
            delta_flow = self.flow_head(h)

            # F(t+1) = F(t) + \Delta(t)
            flow_pred = flow_pred + delta_flow

            # upsample predictions
            flow_up = self.convex_upsampler(h, flow_pred)

            flow_preds.append(flow_up)
        
        return flow_preds
