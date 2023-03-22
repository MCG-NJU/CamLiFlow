import torch
import torch.nn as nn
from .utils import dist_reduce_sum


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = None
        self.metrics = {}

    def clear_metrics(self):
        self.metrics = {}

    @torch.no_grad()
    def update_metrics(self, name, var):
        if isinstance(var, torch.Tensor):
            var = var.reshape(-1)
            count = var.shape[0]
            var = var.float().sum().item()

        var = dist_reduce_sum(var)
        count = dist_reduce_sum(count)

        if count <= 0:
            return

        if name not in self.metrics.keys():
            self.metrics[name] = [0, 0]  # [var, count]

        self.metrics[name][0] += var
        self.metrics[name][1] += count

    def get_metrics(self):
        results = {}
        for name, (var, count) in self.metrics.items():
            results[name] = var / count
        return results

    def get_loss(self):
        if self.loss is None:
            raise ValueError('Loss is empty.')
        return self.loss

    @staticmethod
    def is_better(curr_metrics, best_metrics):
        raise RuntimeError('Function `is_better` must be implemented.')


class FlowModel(BaseModel):
    def __init__(self):
        super(FlowModel, self).__init__()

    @torch.no_grad()
    def update_2d_metrics(self, pred, target):
        if target.shape[1] == 3:  # sparse evaluation
            mask = target[:, 2, :, :] > 0
            target = target[:, :2, :, :]
        else:  # dense evaluation
            mask = torch.ones_like(target)[:, 0, :, :] > 0

        # compute endpoint error
        diff = pred - target
        epe2d_map = torch.linalg.norm(diff, dim=1)
        self.update_metrics('epe2d', epe2d_map[mask])

        # compute 1px accuracy
        acc2d_map = epe2d_map < 1.0
        self.update_metrics('acc2d_1px', acc2d_map[mask])
        
        # compute flow outliers
        mag = torch.linalg.norm(target, dim=1) + 1e-5
        out2d_map = torch.logical_and(epe2d_map > 3.0, epe2d_map / mag > 0.05)
        self.update_metrics('outlier2d', out2d_map[mask])

    @torch.no_grad()
    def update_3d_metrics(self, pred, target, occ_mask=None):
        if target.shape[1] == 4:  # sparse evaluation
            mask = target[:, 3, :] > 0
            target = target[:, :3, :]
        else:
            mask = torch.ones_like(target)[:, 0, :] > 0

        diff = pred - target
        epe3d_map = torch.linalg.norm(diff, dim=1)  # compute endpoint error
        acc3d_map = epe3d_map < 0.05  # compute 5cm accuracy

        if occ_mask is not None:
            mask = torch.logical_and(occ_mask == 0, mask)
            self.update_metrics('epe3d_noc', epe3d_map[mask])
            self.update_metrics('acc3d_5cm_noc', acc3d_map[mask])
        else:
            self.update_metrics('epe3d', epe3d_map[mask])
            self.update_metrics('acc3d_5cm', acc3d_map[mask])
