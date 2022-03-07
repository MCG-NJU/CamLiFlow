import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from .utils import backwarp_2d, resize_flow2d


def calc_supervised_loss_2d(flows, target, cfgs):
    assert len(flows) <= len(cfgs.level_weights)

    total_loss = 0
    for pred, level_weight in zip(flows, cfgs.level_weights):
        assert pred.shape[1] == 2  # [B, 2, H, W]

        if target.shape[1] == 3:
            flow_mask = target[:, 2] > 0
        else:
            flow_mask = torch.ones_like(target)[:, 0] > 0

        diff = torch.abs(resize_flow2d(pred, target.shape[2], target.shape[3]) - target[:, :2])

        if cfgs.order == 'l1':
            loss_l1_map = torch.pow(diff.sum(dim=1) + 0.01, 0.4)
            loss_l1 = loss_l1_map[flow_mask].mean()
            total_loss += level_weight * loss_l1
        elif cfgs.order == 'l2':
            loss_l2_map = torch.linalg.norm(diff, dim=1)
            loss_l2 = loss_l2_map[flow_mask].mean()
            total_loss += level_weight * loss_l2
        else:
            raise NotImplementedError

    return total_loss


def calc_census_loss_2d(image1, image2, noc_mask=None, max_distance=1):
    """
    Calculate photometric loss based on census transform.
    :param image1: [N, 3, H, W] float tensor, ranging from 0 to 1, RGB
    :param image2: [N, 3, H, W] float tensor, ranging from 0 to 1, RGB
    :param noc_mask: [N, 1, H, W] float tensor, ranging from 0 to 1
    :param max_distance: int
    """
    def rgb_to_grayscale(image):
        grayscale = image[:, 0, :, :] * 0.2989 + \
                    image[:, 1, :, :] * 0.5870 + \
                    image[:, 2, :, :] * 0.1140
        return grayscale.unsqueeze(1) * 255.0

    def census_transform(gray_image):
        patch_size = 2 * max_distance + 1
        out_channels = patch_size * patch_size  # 9
        weights = torch.eye(out_channels, dtype=gray_image.dtype, device=gray_image.device)
        weights = weights.view([out_channels, 1, patch_size, patch_size])  # [9, 1, 3, 3]
        patches = nn.functional.conv2d(gray_image, weights, padding=max_distance)
        result = patches - gray_image
        result = result / torch.sqrt(0.81 + torch.pow(result, 2))
        return result

    if noc_mask is not None:
        image1 = noc_mask * image1
        image2 = noc_mask * image2

    gray_image1 = rgb_to_grayscale(image1)
    gray_image2 = rgb_to_grayscale(image2)

    t1 = census_transform(gray_image1)
    t2 = census_transform(gray_image2)

    dist = torch.pow(t1 - t2, 2)
    dist_norm = dist / (0.1 + dist)
    dist_mean = torch.mean(dist_norm, 1, keepdim=True)  # instead of sum

    n, _, h, w = image1.shape
    inner = torch.ones([n, 1, h - 2 * max_distance, w - 2 * max_distance], dtype=image1.dtype, device=image1.device)
    inner_mask = nn.functional.pad(inner, [max_distance] * 4)
    loss = dist_mean * inner_mask

    if noc_mask is not None:
        return loss.mean() / (noc_mask.mean() + 1e-7)
    else:
        return loss.mean()


@torch.cuda.amp.autocast(enabled=False)
def calc_smooth_loss_2d(image, flow, derivative='first'):
    """
    :param image: [N, 3, H, W] float tensor, ranging from 0 to 1, RGB
    :param flow: [N, 2, H, W] float tensor
    :param derivative: 'first' or 'second'
    """
    def gradient(inputs):
        dy = inputs[:, :, 1:, :] - inputs[:, :, :-1, :]
        dx = inputs[:, :, :, 1:] - inputs[:, :, :, :-1]
        return dx, dy

    image_dx, image_dy = gradient(image)
    flow_dx, flow_dy = gradient(flow)

    weights_x = torch.exp(-torch.mean(image_dx.abs(), 1, keepdim=True) * 10)
    weights_y = torch.exp(-torch.mean(image_dy.abs(), 1, keepdim=True) * 10)

    if derivative == 'first':
        loss_x = weights_x * flow_dx.abs() / 2.0
        loss_y = weights_y * flow_dy.abs() / 2.0
    elif derivative == 'second':
        flow_dx2 = flow_dx[:, :, :, 1:] - flow_dx[:, :, :, :-1]
        flow_dy2 = flow_dy[:, :, 1:, :] - flow_dy[:, :, :-1, :]
        loss_x = weights_x[:, :, :, 1:] * flow_dx2.abs()
        loss_y = weights_y[:, :, 1:, :] * flow_dy2.abs()
    else:
        raise NotImplementedError('Unknown derivative: %s' % derivative)

    return loss_x.mean() / 2 + loss_y.mean() / 2


def calc_ssim_loss_2d(image1, image2, noc_mask=None, max_distance=1):
    """
    Calculate photometric loss based on SSIM.
    :param image1: [N, 3, H, W] float tensor, ranging from 0 to 1, RGB
    :param image2: [N, 3, H, W] float tensor, ranging from 0 to 1, RGB
    :param noc_mask: [N, 1, H, W] float tensor, ranging from 0 to 1
    :param max_distance: int
    """
    patch_size = 2 * max_distance + 1
    c1, c2 = 0.01 ** 2, 0.03 ** 2

    if noc_mask is not None:
        image1 = noc_mask * image1
        image2 = noc_mask * image2

    mu_x = nn.AvgPool2d(patch_size, 1, 0)(image1)
    mu_y = nn.AvgPool2d(patch_size, 1, 0)(image2)
    mu_x_square, mu_y_square = mu_x.pow(2), mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x = nn.AvgPool2d(patch_size, 1, 0)(image1 * image1) - mu_x_square
    sigma_y = nn.AvgPool2d(patch_size, 1, 0)(image2 * image2) - mu_y_square
    sigma_xy = nn.AvgPool2d(patch_size, 1, 0)(image1 * image2) - mu_xy

    ssim_n = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    ssim_d = (mu_x_square + mu_y_square + c1) * (sigma_x + sigma_y + c2)
    ssim = ssim_n / ssim_d
    loss = torch.clamp((1 - ssim) / 2, min=0.0, max=1.0)

    if noc_mask is not None:
        return loss.mean() / (noc_mask.mean() + 1e-7)
    else:
        return loss.mean()


def calc_unsupervised_loss_2d(pyramid_flows12, pyramid_flows21, image1, image2, occ_mask1, occ_mask2, cfgs):
    photo_loss = smooth_loss = 0
    for lv, (pyramid_flow12, pyramid_flow21) in enumerate(zip(pyramid_flows12, pyramid_flows21)):
        if lv == 0:
            image1_scaled, noc_mask1_scaled = image1, 1 - occ_mask1[:, None, :, :]
            image2_scaled, noc_mask2_scaled = image2, 1 - occ_mask2[:, None, :, :]
        else:
            curr_h, curr_w = pyramid_flow12.shape[2], pyramid_flow12.shape[3]
            image1_scaled = interpolate(image1, (curr_h, curr_w), mode='area')
            image2_scaled = interpolate(image2, (curr_h, curr_w), mode='area')
            noc_mask1_scaled = 1 - interpolate(occ_mask1[:, None, :, :], (curr_h, curr_w), mode='nearest')
            noc_mask2_scaled = 1 - interpolate(occ_mask2[:, None, :, :], (curr_h, curr_w), mode='nearest')

        image1_scaled_warp = backwarp_2d(image1_scaled, pyramid_flow21, padding_mode='border')
        image2_scaled_warp = backwarp_2d(image2_scaled, pyramid_flow12, padding_mode='border')

        # calculate photometric loss
        if cfgs.photometric_loss == 'ssim':
            photo_loss1 = calc_ssim_loss_2d(image1_scaled, image2_scaled_warp, noc_mask1_scaled)
            photo_loss2 = calc_ssim_loss_2d(image2_scaled, image1_scaled_warp, noc_mask2_scaled)
        elif cfgs.photometric_loss == 'census':
            photo_loss1 = calc_census_loss_2d(image1_scaled, image2_scaled_warp, noc_mask1_scaled)
            photo_loss2 = calc_census_loss_2d(image2_scaled, image1_scaled_warp, noc_mask2_scaled)
        else:
            raise NotImplementedError('Unknown photometric loss: %s' % cfgs.photometric_loss)
        photo_loss += cfgs.photometric_weights[lv] * (photo_loss1 + photo_loss2) / 2

        # calculate smooth loss
        scale = min(pyramid_flows12[0].shape[2], pyramid_flows12[0].shape[3])
        smooth_loss1 = calc_smooth_loss_2d(image1_scaled, pyramid_flow12 / scale, cfgs.smooth_derivative)
        smooth_loss2 = calc_smooth_loss_2d(image2_scaled, pyramid_flow21 / scale, cfgs.smooth_derivative)
        smooth_loss += cfgs.smooth_weights[lv] * (smooth_loss1 + smooth_loss2) / 2

    return photo_loss, smooth_loss
