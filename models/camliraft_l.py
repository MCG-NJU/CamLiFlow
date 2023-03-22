from .base import FlowModel
from .camliraft_l_core import CamLiRAFT_L_Core
from .losses import calc_sequence_loss_3d
from .ids import persp2paral, paral2persp


class CamLiRAFT_L(FlowModel):
    def __init__(self, cfgs):
        super(CamLiRAFT_L, self).__init__()
        self.cfgs = cfgs
        self.core = CamLiRAFT_L_Core(cfgs)

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

        flow_preds = self.core.forward(pc1, pc2)

        if self.cfgs.ids.enabled:
            for i in range(len(flow_preds)):
                flow_preds[i] = paral2persp(pc1 + flow_preds[i], persp_cam_info, paral_cam_info) - \
                                paral2persp(pc1, persp_cam_info, paral_cam_info)

        final_flow_3d = flow_preds[-1]

        target_3d = inputs['flow_3d'][:, :3]
        self.loss = calc_sequence_loss_3d(flow_preds, target_3d, self.cfgs.loss)

        # prepare scalar summary
        self.update_metrics('loss3d', self.loss)
        self.update_3d_metrics(final_flow_3d, target_3d)

        return {'flow_3d': final_flow_3d}

    @staticmethod
    def is_better(curr_summary, best_summary):
        if best_summary is None:
            return True
        return curr_summary['epe3d'] < best_summary['epe3d']
