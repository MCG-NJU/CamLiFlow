from .base import FlowModel
from .camlipwc_l_core import CamLiPWC_L_Core
from .losses import calc_pyramid_loss_3d
from .utils import build_pc_pyramid
from .ids import persp2paral, paral2persp


class CamLiPWC_L(FlowModel):
    def __init__(self, cfgs):
        super(CamLiPWC_L, self).__init__()
        self.cfgs = cfgs
        self.core = CamLiPWC_L_Core(cfgs)

    def forward(self, inputs):
        pc1, pc2 = inputs['pcs'][:, :3], inputs['pcs'][:, 3:]
        intrinsics = inputs['intrinsics']

        persp_cam_info = {
            'projection_mode': 'perspective',
            'sensor_h': 540,
            'sensor_w': 960,
            'f': intrinsics[:, 0],
            'cx': intrinsics[:, 1],
            'cy': intrinsics[:, 2],
        }

        if self.cfgs.ids.enabled:
            paral_cam_info = {
                'projection_mode': 'parallel',
                'sensor_h': round(540 / 32),
                'sensor_w': round(960 / 32),
                'cx': (round(960 / 32) - 1) / 2,
                'cy': (round(540 / 32) - 1) / 2,
            }
            pc1 = persp2paral(pc1, persp_cam_info, paral_cam_info)
            pc2 = persp2paral(pc2, persp_cam_info, paral_cam_info)
        else:
            paral_cam_info = None

        xyzs1, xyzs2, sample_indices1, _ = build_pc_pyramid(
            pc1, pc2, n_samples_list=[4096, 2048, 1024, 512, 256]
        )

        feats1_3d = self.core.encode(xyzs1)
        feats2_3d = self.core.encode(xyzs2)

        # predict flows_3d
        flows_3d = self.core.decode(xyzs1, xyzs2, feats1_3d, feats2_3d)

        if self.cfgs.ids.enabled:
            for idx, (xyz1, flow_3d) in enumerate(zip(xyzs1, flows_3d)):
                flows_3d[idx] = paral2persp(xyz1 + flow_3d, persp_cam_info, paral_cam_info) - \
                                paral2persp(xyz1, persp_cam_info, paral_cam_info)

        final_flow_3d = flows_3d[0]

        if 'flow_3d' not in inputs:
            return {'flow_3d': final_flow_3d}

        # calculate loss
        target_3d = inputs['flow_3d']
        self.loss = calc_pyramid_loss_3d(flows_3d, target_3d, self.cfgs.loss, sample_indices1)

        # prepare scalar summary
        self.update_metrics('loss3d', self.loss)
        self.update_3d_metrics(final_flow_3d, target_3d)

        return {'flow_3d': final_flow_3d}

    @staticmethod
    def is_better(curr_summary, best_summary):
        if best_summary is None:
            return True
        return curr_summary['epe3d'] < best_summary['epe3d']
