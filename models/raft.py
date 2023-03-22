from .base import FlowModel
from .raft_core import RAFTCore
from .utils import InputPadder
from .losses import calc_sequence_loss_2d


class RAFT(FlowModel):
    def __init__(self, cfgs):
        super(RAFT, self).__init__()
        self.cfgs = cfgs
        self.core = RAFTCore(cfgs)

    def forward(self, inputs):
        images = 2 * (inputs['images'].float() / 255.0) - 1.0

        # pad input shape to a multiple of 8
        padder = InputPadder(images.shape, x=8)
        image1, image2 = padder.pad(images[:, :3], images[:, 3:])

        # core
        flow_preds = self.core(image1, image2)

        # unresize
        for i in range(len(flow_preds)):
            flow_preds[i] = padder.unpad(flow_preds[i])

        final_flow = flow_preds[-1]

        if 'flow_2d' not in inputs:
            return {'flow_2d': final_flow}

        # calculate supervised loss
        target_2d = inputs['flow_2d'].float()

        self.loss = calc_sequence_loss_2d(flow_preds, target_2d, self.cfgs.loss)
        self.update_metrics('loss2d', self.loss)
        self.update_2d_metrics(final_flow, target_2d)

        return {'flow_2d': final_flow}

    @staticmethod
    def is_better(curr_metrics, best_metrics):
        if best_metrics is None:
            return True
        return curr_metrics['epe2d'] < best_metrics['epe2d']
