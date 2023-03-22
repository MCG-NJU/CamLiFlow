from .base import FlowModel
from .pwc_core import PWCCore
from .utils import resize_to_64x, resize_flow2d
from .losses import calc_pyramid_loss_2d


class PWC(FlowModel):
    def __init__(self, cfgs):
        super(PWC, self).__init__()
        self.cfgs = cfgs
        self.core = PWCCore(cfgs)

    def forward(self, inputs):
        images = inputs['images'].float() / 255.0

        # assert images.shape[2] % 64 == 0 and images.shape[3] % 64 == 0
        origin_h, origin_w = images.shape[2:]
        images = resize_to_64x(images, None)[0]
        image1, image2 = images[:, :3], images[:, 3:]

        # image1->image2 and image2->image1
        feats1 = self.core.encode(image1)
        feats2 = self.core.encode(image2)
        flows = self.core.decode(feats1, feats2)
        final_flow = resize_flow2d(flows[0], origin_h, origin_w)

        if 'flow_2d' not in inputs:
            return {'flow_2d': final_flow}

        # calculate supervised loss
        target_2d = inputs['flow_2d'].float()
        self.loss = calc_pyramid_loss_2d(flows, target_2d, self.cfgs.loss)

        self.update_metrics('loss2d', self.loss)
        self.update_2d_metrics(final_flow, target_2d)

        return {'flow_2d': final_flow}

    @staticmethod
    def is_better(curr_metrics, best_metrics):
        if best_metrics is None:
            return True
        return curr_metrics['epe2d'] < best_metrics['epe2d']
