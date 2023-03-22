import torch.nn as nn
from .base import FlowModel
from .camlipwc_core import CamLiPWC_Core
from .losses import calc_pyramid_loss_2d, calc_pyramid_loss_3d
from .utils import resize_to_64x, resize_flow2d, build_pc_pyramid
from .ids import persp2paral, paral2persp


class CamLiPWC(FlowModel):
    def __init__(self, cfgs):
        super(CamLiPWC, self).__init__()
        self.cfgs = cfgs
        self.core = CamLiPWC_Core(cfgs.pwc2d, cfgs.pwc3d, cfgs.fusion)
        
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

        # inverse depth scaling
        persp_cam_info = {
            'projection_mode': 'perspective',
            'sensor_h': origin_h,
            'sensor_w': origin_w,
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

        # encode features
        xyzs1, xyzs2, sample_indices1, _ = build_pc_pyramid(
            pc1, pc2, [4096, 2048, 1024, 512, 256]  # 1/4
        )
        feats1_2d, feats1_3d = self.core.encode(image1, xyzs1)
        feats2_2d, feats2_3d = self.core.encode(image2, xyzs2)

        # predict flows (1->2)
        flows_2d, flows_3d = self.core.decode(
            xyzs1, xyzs2, feats1_2d, feats2_2d, feats1_3d, feats2_3d, paral_cam_info
        )

        # inverse depth scaling
        for idx, (xyz1, flow_3d) in enumerate(zip(xyzs1, flows_3d)):
            flows_3d[idx] = paral2persp(xyz1 + flow_3d, persp_cam_info, paral_cam_info) - \
                            paral2persp(xyz1, persp_cam_info, paral_cam_info)

        final_flow_2d = resize_flow2d(flows_2d[0], origin_h, origin_w)
        final_flow_3d = flows_3d[0]

        if 'flow_2d' not in inputs or 'flow_3d' not in inputs:
            return {'flow_2d': final_flow_2d, 'flow_3d': final_flow_3d}

        target_2d = inputs['flow_2d'].float()
        target_3d = inputs['flow_3d'].float()

        # calculate losses
        loss_2d = calc_pyramid_loss_2d(flows_2d, target_2d, self.cfgs.loss2d)
        loss_3d = calc_pyramid_loss_3d(flows_3d, target_3d, self.cfgs.loss3d, sample_indices1)
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
