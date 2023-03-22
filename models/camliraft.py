import torch
import torch.nn as nn
from .base import FlowModel
from .camliraft_core import CamLiRAFT_Core
from .losses import calc_sequence_loss_2d, calc_sequence_loss_3d
from .utils import InputPadder
from .ids import paral2persp, persp2paral


class CamLiRAFT(FlowModel):
    def __init__(self, cfgs):
        super(CamLiRAFT, self).__init__()
        self.cfgs = cfgs
        self.core = CamLiRAFT_Core(cfgs)

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
        images = inputs['images'].float()
        pc1, pc2 = inputs['pcs'][:, :3], inputs['pcs'][:, 3:]
        intrinsics = inputs['intrinsics']

        # pad input shape to a multiple of 8
        padder = InputPadder(images.shape, x=8)
        image1, image2 = padder.pad(images[:, :3], images[:, 3:])

        norm_mean = torch.tensor([123.675, 116.280, 103.530], device=images.device)
        norm_std = torch.tensor([58.395, 57.120, 57.375], device=images.device)
        image1 = image1 - norm_mean.reshape(1, 3, 1, 1)
        image2 = image2 - norm_mean.reshape(1, 3, 1, 1)
        image1 = image1 / norm_std.reshape(1, 3, 1, 1)
        image2 = image2 / norm_std.reshape(1, 3, 1, 1)

        persp_cam_info = {
            'projection_mode': 'perspective',
            'sensor_h': image1.shape[-2],  # 544
            'sensor_w': image1.shape[-1],  # 960
            'f': intrinsics[:, 0],
            'cx': intrinsics[:, 1],
            'cy': intrinsics[:, 2],
        }
        paral_cam_info = {
            'projection_mode': 'parallel',
            'sensor_h': round(image1.shape[-2] / 32),
            'sensor_w': round(image1.shape[-1] / 32),
            'cx': (round(image1.shape[-1] / 32) - 1) / 2,
            'cy': (round(image1.shape[-2] / 32) - 1) / 2,
        }
        pc1 = persp2paral(pc1, persp_cam_info, paral_cam_info)
        pc2 = persp2paral(pc2, persp_cam_info, paral_cam_info)

        flow_2d_preds, flow_3d_preds = self.core(image1, image2, pc1, pc2, paral_cam_info)

        for i in range(len(flow_2d_preds)):
            flow_2d_preds[i] = padder.unpad(flow_2d_preds[i])

        for i in range(len(flow_3d_preds)):
            flow_3d_preds[i] = paral2persp(pc1 + flow_3d_preds[i], persp_cam_info, paral_cam_info) -\
                               paral2persp(pc1, persp_cam_info, paral_cam_info)

        final_flow_2d = flow_2d_preds[-1]
        final_flow_3d = flow_3d_preds[-1]

        if 'flow_2d' not in inputs or 'flow_3d' not in inputs:
            return {'flow_2d': final_flow_2d, 'flow_3d': final_flow_3d}

        target_2d = inputs['flow_2d'].float()
        target_3d = inputs['flow_3d'].float()

        # calculate losses
        loss_2d = calc_sequence_loss_2d(flow_2d_preds, target_2d, cfgs=self.cfgs.loss2d)
        loss_3d = calc_sequence_loss_3d(flow_3d_preds, target_3d, cfgs=self.cfgs.loss3d)
        self.loss = loss_2d + loss_3d

        self.update_metrics('loss', self.loss)
        self.update_metrics('loss2d', loss_2d)
        self.update_metrics('loss3d', loss_3d)
        self.update_2d_metrics(final_flow_2d, target_2d)
        self.update_3d_metrics(final_flow_3d, target_3d)

        if 'occ_mask_3d' in inputs:
            self.update_3d_metrics(final_flow_3d, target_3d, inputs['occ_mask_3d'])

        return {'flow_2d': final_flow_2d, 'flow_3d': final_flow_3d}

    @staticmethod
    def is_better(curr_metrics, best_metrics):
        if best_metrics is None:
            return True
        return curr_metrics['epe2d'] < best_metrics['epe2d']
