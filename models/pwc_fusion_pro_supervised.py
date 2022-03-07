import torch
import torch.nn as nn
from .pwc3d_core import build_pc_pyramid
from .pwc_fusion_core import PWCFusionCore
from .losses2d import calc_supervised_loss_2d
from .losses3d import calc_supervised_loss_3d
from .utils import resize_to_64x, resize_flow2d, perspect2parallel, parallel2perspect


class PWCFusionProSupervised(nn.Module):
    def __init__(self, cfgs):
        super(PWCFusionProSupervised, self).__init__()
        self.cfgs = cfgs
        self.pwc_fusion_core = PWCFusionCore(cfgs.pwc2d, cfgs.pwc3d)
        self.loss = None
        self.scalar_summary, self.image_summary = {}, {}

    def train(self, mode=True):
        self.training = mode

        for module in self.children():
            module.train(mode)

        if self.cfgs.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()

        return self

    def eval(self):
        return self.train(False)

    def forward(self, inputs):
        images = inputs['images'].float() / 255.0
        pc1, pc2 = inputs['pcs'][:, :3], inputs['pcs'][:, 3:]
        intrinsics = inputs['intrinsics']

        # assert images.shape[2] % 64 == 0 and images.shape[3] % 64 == 0
        origin_h, origin_w = images.shape[2:]
        images = resize_to_64x(images, None)[0]
        image1, image2 = images[:, :3], images[:, 3:]

        persp_cam_info = {
            'projection_mode': 'perspective',
            'sensor_h': origin_h,
            'sensor_w': origin_w,
            'f': intrinsics[:, 0],
            'cx': intrinsics[:, 1],
            'cy': intrinsics[:, 2],
        }

        if self.cfgs.ids.enabled:
            parallel_sensor_size = (
                images.shape[2] // self.cfgs.ids.sensor_size_divisor,
                images.shape[3] // self.cfgs.ids.sensor_size_divisor,
            )
            paral_cam_info = {
                'projection_mode': 'parallel',
                'sensor_h': parallel_sensor_size[0],
                'sensor_w': parallel_sensor_size[1],
                'cx': (parallel_sensor_size[1] - 1) / 2,
                'cy': (parallel_sensor_size[0] - 1) / 2,
            }
            pc1 = perspect2parallel(pc1, persp_cam_info, paral_cam_info)
            pc2 = perspect2parallel(pc2, persp_cam_info, paral_cam_info)
        else:
            paral_cam_info = None

        # encode features
        xyzs1, xyzs2, sample_indices1, _ = build_pc_pyramid(pc1, pc2, [4096, 2048, 1024, 512, 256])
        feats1_2d, feats1_3d = self.pwc_fusion_core.encode(image1, xyzs1)
        feats2_2d, feats2_3d = self.pwc_fusion_core.encode(image2, xyzs2)

        # predict flows (1->2)
        flows_2d, flows_3d = self.pwc_fusion_core.decode(
            xyzs1, xyzs2, feats1_2d, feats2_2d, feats1_3d, feats2_3d,
            paral_cam_info if self.cfgs.ids.enabled else persp_cam_info
        )

        if self.cfgs.ids.enabled:
            pc1 = parallel2perspect(pc1, persp_cam_info, paral_cam_info)
            pc2 = parallel2perspect(pc2, persp_cam_info, paral_cam_info)
            for idx, (xyz1, flow12_3d) in enumerate(zip(xyzs1, flows_3d)):
                flows_3d[idx] = parallel2perspect(xyz1 + flow12_3d, persp_cam_info, paral_cam_info) - \
                                parallel2perspect(xyz1, persp_cam_info, paral_cam_info)

        final_flow_2d = resize_flow2d(flows_2d[0], origin_h, origin_w)
        final_flow_3d = flows_3d[0]

        if 'flow_2d' not in inputs or 'flow_3d' not in inputs:
            return {
                'flow_2d': final_flow_2d,
                'flow_3d': final_flow_3d
            }

        # calculate losses
        target_2d = inputs['flow_2d'].float()
        target_3d = inputs['flow_3d'].float()
        final_loss_2d = calc_supervised_loss_2d(flows_2d, target_2d, self.cfgs.loss2d)
        final_loss_3d = calc_supervised_loss_3d(flows_3d, target_3d, self.cfgs.loss3d, sample_indices1)
        self.loss = final_loss_2d + final_loss_3d

        # prepare scalar summary
        self.scalar_summary = {'loss': self.loss}

        with torch.no_grad():
            if target_2d.shape[1] == 3:  # sparse evaluation
                target_2d_mask = target_2d[:, 2, :, :] > 0
                target_2d = target_2d[:, :2, :, :]
            else:  # dense evaluation
                target_2d_mask = torch.ones_like(target_2d)[:, 0, :, :] > 0

            # compute endpoint error
            diff = final_flow_2d - target_2d
            epe2d_map = torch.linalg.norm(diff, dim=1) * target_2d_mask
            epe2d_bat = epe2d_map.sum(dim=[1, 2]) / target_2d_mask.sum(dim=[1, 2])
            self.scalar_summary['epe2d'] = epe2d_bat.mean()

            # compute 1px accuracy
            acc1_2d_map = (epe2d_map < 1.0).float() * target_2d_mask
            acc1_2d_bat = acc1_2d_map.sum(dim=[1, 2]) / target_2d_mask.sum(dim=[1, 2])
            self.scalar_summary['acc2d_1px'] = acc1_2d_bat.mean()

            # compute flow outliers
            target_2d_mag = torch.linalg.norm(target_2d, dim=1) + 1e-5
            outlier_2d_map = torch.logical_and(epe2d_map > 3.0, epe2d_map / target_2d_mag > 0.05).float() * target_2d_mask
            outlier_2d_bat = outlier_2d_map.sum(dim=[1, 2]) / target_2d_mask.sum(dim=[1, 2])
            self.scalar_summary['outlier2d'] = outlier_2d_bat.mean()

        with torch.no_grad():
            if target_3d.shape[1] == 4:
                target_3d_mask = target_3d[:, 3, :] > 0
                target_3d = target_3d[:, :3, :]
            else:
                target_3d_mask = torch.ones_like(target_3d)[:, 0, :] > 0

            # compute endpoint error
            diff = final_flow_3d - target_3d
            epe3d_map = torch.linalg.norm(diff, dim=1) * target_3d_mask
            epe3d_bat = epe3d_map.sum(dim=1) / target_3d_mask.sum(dim=1)
            self.scalar_summary['epe3d'] = epe3d_bat.mean()

            # compute 5cm accuracy
            acc5_3d_map = (epe3d_map < 0.05).float() * target_3d_mask
            acc5_3d_bat = acc5_3d_map.sum(dim=1) / target_3d_mask.sum(dim=1)
            self.scalar_summary['acc3d_5cm'] = acc5_3d_bat.mean()

        return {
            'flow_2d': final_flow_2d,
            'flow_3d': final_flow_3d
        }

    def get_loss(self):
        return self.loss

    def get_scalar_summary(self):
        return self.scalar_summary

    def get_image_summary(self):
        return self.image_summary

    def get_log_string(self, scalar_summary=None):
        if scalar_summary is None:
            scalar_summary = self.get_scalar_summary()
        log_strings = [
            'loss: %.1f' % scalar_summary['loss'],
            'epe2d: %.3f' % scalar_summary['epe2d'],
            'epe3d: %.3f' % scalar_summary['epe3d'],
        ]
        return ', '.join(log_strings)

    @staticmethod
    def is_better(curr_summary, best_summary):
        if best_summary is None:
            return True
        return curr_summary['outlier2d'] < best_summary['outlier2d']
