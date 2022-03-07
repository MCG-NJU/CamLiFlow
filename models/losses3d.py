import torch
from .csrc import k_nearest_neighbor
from .utils import batch_indexing_channel_first


def calc_supervised_loss_3d(flows, target, cfgs, indices):
    assert len(flows) <= len(cfgs.level_weights)

    total_loss = 0
    for idx, (flow, level_weight) in enumerate(zip(flows, cfgs.level_weights)):
        level_target = batch_indexing_channel_first(target, indices[idx])

        if level_target.shape[1] == 4:
            flow_mask = level_target[:, 3, :] > 0
            diff = flow - level_target[:, :3, :]
            epe_l1 = torch.pow(diff.abs().sum(dim=1) + 0.01, 0.4)[flow_mask].mean()
            epe_l2 = torch.linalg.norm(diff, dim=1)[flow_mask].mean()
        else:
            diff = flow - level_target
            epe_l1 = torch.pow(diff.abs().sum(dim=1) + 0.01, 0.4).mean()
            epe_l2 = torch.linalg.norm(diff, dim=1).mean()

        if cfgs.order == 'l1':
            total_loss += level_weight * epe_l1
        elif cfgs.order == 'l2':
            total_loss += level_weight * epe_l2
        else:
            raise NotImplementedError

    return total_loss


def calc_chamfer_loss_3d(xyzs1, xyzs2, flows, level_weights, occ_masks1=None, occ_masks2=None):
    def _calc_chamfer_loss_single_level(pc1, pc2, occ_mask1=None, occ_mask2=None):
        batch_size, n_points_1, n_points_2 = pc1.shape[0], pc1.shape[2], pc2.shape[2]

        occ_mask1 = torch.zeros([batch_size, n_points_1], device=pc1.device) if occ_mask1 is None else occ_mask1
        occ_mask2 = torch.zeros([batch_size, n_points_2], device=pc2.device) if occ_mask2 is None else occ_mask2
        noc_mask1, noc_mask2 = 1 - occ_mask1, 1 - occ_mask2

        dists1, dists2 = chamfer_distance(pc1.transpose(1, 2), pc2.transpose(1, 2))
        loss1 = (dists1 * noc_mask1).mean() / (noc_mask1.mean() + 1e-7)
        loss2 = (dists2 * noc_mask2).mean() / (noc_mask2.mean() + 1e-7)
        return loss1 + loss2

    occ_masks1 = [None] * len(xyzs1) if occ_masks1 is None else occ_masks1
    occ_masks2 = [None] * len(xyzs2) if occ_masks2 is None else occ_masks2

    chamfer_loss = 0
    for xyz1, xyz2, flow, level_weight, occ_mask1, occ_mask2 in zip(xyzs1, xyzs2, flows, level_weights, occ_masks1, occ_masks2):
        chamfer_loss += level_weight * _calc_chamfer_loss_single_level(xyz1 + flow, xyz2, occ_mask1, occ_mask2)
    return chamfer_loss


def calc_smooth_loss_3d(xyzs1, flows, knns1, level_weights):
    def _calc_smooth_loss_single_level(pc, flow, k=9, knn_indices=None):
        """
        :param pc: [batch_size, 3, n_points]
        :param flow: [batch_size, 3, n_points]
        :param k: k-nearest-neighbor, int
        :param knn_indices: [batch_size, n_points]
        """
        batch_size, n_points = pc.shape[0], pc.shape[2]

        # Calculate k nearest neighbors
        if knn_indices is not None:  # knn indices are pre-computed
            assert knn_indices.shape[:2] == torch.Size([batch_size, n_points])
            assert knn_indices.shape[2] >= k
            knn_indices = knn_indices[:, :, :k]
        else:
            knn_indices = k_nearest_neighbor(pc, pc, k)  # [bs, n_points, k]

        knn_flow = batch_indexing_channel_first(flow, knn_indices)  # [bs, 3, n_points, k]
        diff = torch.norm(knn_flow - flow.unsqueeze(3), dim=1).sum(dim=-1) / (k - 1)  # [bs, n_points]
        loss = diff.mean()
        return loss

    smooth_loss = 0
    for xyz1, flow, knn1, level_weight in zip(xyzs1, flows, knns1, level_weights):
        smooth_loss += level_weight * _calc_smooth_loss_single_level(xyz1, flow, knn_indices=knn1)
    return smooth_loss


def calc_unsupervised_loss_3d_bidirection(xyzs1, xyzs2, flows12_3d, flows21_3d, knns_1, knns_2, cfgs, occ_masks1=None, occ_masks2=None):
    assert len(xyzs1) == len(xyzs2) == len(flows12_3d) == len(flows21_3d) == len(cfgs.level_weights)

    chamfer_loss1 = calc_chamfer_loss_3d(xyzs1, xyzs2, flows12_3d, cfgs.level_weights, occ_masks1, occ_masks2)
    chamfer_loss2 = calc_chamfer_loss_3d(xyzs2, xyzs1, flows21_3d, cfgs.level_weights, occ_masks2, occ_masks1)
    chamfer_loss = (chamfer_loss1 + chamfer_loss2) / 2.0

    smooth_loss1 = calc_smooth_loss_3d(xyzs1, flows12_3d, knns_1, cfgs.level_weights)
    smooth_loss2 = calc_smooth_loss_3d(xyzs2, flows21_3d, knns_2, cfgs.level_weights)
    smooth_loss = (smooth_loss1 + smooth_loss2) / 2.0

    return chamfer_loss * cfgs.chamfer_weight, smooth_loss * cfgs.smooth_weight


def calc_unsupervised_loss_3d(xyzs1, xyzs2, flows12_3d, knns_1, cfgs, occ_masks1=None, occ_masks2=None):
    chamfer_loss = calc_chamfer_loss_3d(xyzs1, xyzs2, flows12_3d, cfgs.level_weights, occ_masks1, occ_masks2)
    smooth_loss = calc_smooth_loss_3d(xyzs1, flows12_3d, knns_1, cfgs.level_weights)
    return chamfer_loss * cfgs.chamfer_weight, smooth_loss * cfgs.smooth_weight
