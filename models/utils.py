import time
import torch
import torch.nn as nn
from torch.nn.functional import grid_sample, interpolate, pad
from .csrc import k_nearest_neighbor


class Conv1dNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, norm=None, activation='leaky_relu'):
        super().__init__()
        self.conv_fn = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups)

        if norm == 'batch_norm':
            self.norm_fn = nn.BatchNorm1d(out_channels)
        elif norm == 'instance_norm':
            self.norm_fn = nn.InstanceNorm1d(out_channels)
        elif norm is None:
            self.norm_fn = nn.Identity()
        else:
            raise NotImplementedError('Unknown normalization function: %s' % norm)

        if activation == 'relu':
            self.relu_fn = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.relu_fn = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation is None:
            self.relu_fn = nn.Identity()
        else:
            raise NotImplementedError('Unknown activation function: %s' % activation)

    def forward(self, x):
        x = self.conv_fn(x)
        x = self.norm_fn(x)
        x = self.relu_fn(x)
        return x


class Conv2dNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, norm=None, activation='leaky_relu'):
        super().__init__()
        self.conv_fn = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups)

        if norm == 'batch_norm':
            self.norm_fn = nn.BatchNorm2d(out_channels)
        elif norm == 'instance_norm':
            self.norm_fn = nn.InstanceNorm2d(out_channels)
        elif norm is None:
            self.norm_fn = nn.Identity()
        else:
            raise NotImplementedError('Unknown normalization function: %s' % norm)

        if activation == 'relu':
            self.relu_fn = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.relu_fn = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation is None:
            self.relu_fn = nn.Identity()
        else:
            raise NotImplementedError('Unknown activation function: %s' % activation)

    def forward(self, x):
        x = self.conv_fn(x)
        x = self.norm_fn(x)
        x = self.relu_fn(x)
        return x


class MLP1d(nn.Module):
    def __init__(self, in_channels, mlps, norm=None, activation='leaky_relu'):
        super().__init__()
        assert isinstance(in_channels, int)
        assert isinstance(mlps, list)
        n_channels = [in_channels] + mlps

        self.convs = nn.ModuleList()
        for in_channels, out_channels in zip(n_channels[:-1], n_channels[1:]):
            self.convs.append(Conv1dNormRelu(in_channels, out_channels, norm=norm, activation=activation))

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x


class MLP2d(nn.Module):
    def __init__(self, in_channels, mlps, norm=None, activation='leaky_relu'):
        super().__init__()
        assert isinstance(in_channels, int)
        assert isinstance(mlps, list)
        n_channels = [in_channels] + mlps

        self.convs = nn.ModuleList()
        for in_channels, out_channels in zip(n_channels[:-1], n_channels[1:]):
            self.convs.append(Conv2dNormRelu(in_channels, out_channels, norm=norm, activation=activation))

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x


def batch_indexing_channel_last(batched_data: torch.Tensor, batched_indices: torch.Tensor):
    """
    :param batched_data: [batch_size, N, C]
    :param batched_indices: [batch_size, I1, I2, ..., Im]
    :return: indexed data: [batch_size, I1, I2, ..., Im, C]
    """
    assert batched_data.shape[0] == batched_indices.shape[0]
    batch_size = batched_data.shape[0]
    view_shape = [batch_size] + [1] * (len(batched_indices.shape) - 1)
    expand_shape = [batch_size] + list(batched_indices.shape)[1:]
    indices_of_batch = torch.arange(batch_size, dtype=torch.long, device=batched_data.device)
    indices_of_batch = indices_of_batch.view(view_shape).expand(expand_shape)  # [bs, I1, I2, ..., Im]
    if len(batched_data.shape) == 2:
        return batched_data[indices_of_batch, batched_indices.to(torch.long)]
    else:
        return batched_data[indices_of_batch, batched_indices.to(torch.long), :]


def batch_indexing_channel_first(batched_data: torch.Tensor, batched_indices: torch.Tensor):
    """
    :param batched_data: [batch_size, C, N]
    :param batched_indices: [batch_size, I1, I2, ..., Im]
    :return: indexed data: [batch_size, C, I1, I2, ..., Im]
    """
    def product(arr):
        p = 1
        for i in arr:
            p *= i
        return p
    assert batched_data.shape[0] == batched_indices.shape[0]
    batch_size, n_channels = batched_data.shape[:2]
    indices_shape = list(batched_indices.shape[1:])
    batched_indices = batched_indices.reshape([batch_size, 1, -1])
    batched_indices = batched_indices.expand([batch_size, n_channels, product(indices_shape)])
    result = torch.gather(batched_data, dim=2, index=batched_indices.to(torch.int64))
    result = result.view([batch_size, n_channels] + indices_shape)
    return result


def knn_interpolation(input_xyz, input_features, query_xyz, k=3):
    """
    :param input_xyz: 3D locations of input points, [batch_size, 3, n_inputs]
    :param input_features: features of input points, [batch_size, n_features, n_inputs]
    :param query_xyz: 3D locations of query points, [batch_size, 3, n_queries]
    :param k: k-nearest neighbor, int
    :return interpolated features: [batch_size, n_features, n_queries]
    """
    knn_indices = k_nearest_neighbor(input_xyz, query_xyz, k)  # [batch_size, n_queries, 3]
    knn_xyz = batch_indexing_channel_first(input_xyz, knn_indices)  # [batch_size, 3, n_queries, k]
    knn_dists = torch.linalg.norm(knn_xyz - query_xyz[..., None], dim=1).clamp(1e-8)  # [bs, n_queries, k]
    knn_weights = 1.0 / knn_dists  # [bs, n_queries, k]
    knn_weights = knn_weights / torch.sum(knn_weights, dim=-1, keepdim=True)  # [bs, n_queries, k]
    knn_features = batch_indexing_channel_first(input_features, knn_indices)  # [bs, n_features, n_queries, k]
    interpolated = torch.sum(knn_features * knn_weights[:, None, :, :], dim=-1)  # [bs, n_features, n_queries]

    return interpolated


def backwarp_3d(xyz1, xyz2, flow12, k=3):
    """
    :param xyz1: 3D locations of points1, [batch_size, 3, n_points]
    :param xyz2: 3D locations of points2, [batch_size, 3, n_points]
    :param flow12: scene flow, [batch_size, 3, n_points]
    :param k: k-nearest neighbor, int
    """
    xyz1_warp = xyz1 + flow12
    flow21 = knn_interpolation(xyz1_warp, -flow12, query_xyz=xyz2, k=k)
    xyz2_warp = xyz2 + flow21
    return xyz2_warp


mesh_grid_cache = {}
def mesh_grid(n, h, w, device, channel_first=True):
    global mesh_grid_cache
    str_id = '%d,%d,%d,%s,%s' % (n, h, w, device, channel_first)
    if str_id not in mesh_grid_cache:
        x_base = torch.arange(0, w, dtype=torch.float32, device=device)[None, None, :].expand(n, h, w)
        y_base = torch.arange(0, h, dtype=torch.float32, device=device)[None, None, :].expand(n, w, h)  # NWH
        grid = torch.stack([x_base, y_base.transpose(1, 2)], 1)  # B2HW
        if not channel_first:
            grid = grid.permute(0, 2, 3, 1)  # BHW2
        mesh_grid_cache[str_id] = grid
    return mesh_grid_cache[str_id]


def backwarp_2d(x, flow12, padding_mode):
    def norm_grid(g):
        grid_norm = torch.zeros_like(g)
        grid_norm[:, 0, :, :] = 2.0 * g[:, 0, :, :] / (g.shape[3] - 1) - 1.0
        grid_norm[:, 1, :, :] = 2.0 * g[:, 1, :, :] / (g.shape[2] - 1) - 1.0
        return grid_norm.permute(0, 2, 3, 1)

    assert x.size()[-2:] == flow12.size()[-2:]
    batch_size, _, image_h, image_w = x.size()
    grid = mesh_grid(batch_size, image_h, image_w, device=x.device)
    grid = norm_grid(grid + flow12)

    return grid_sample(x, grid, padding_mode=padding_mode, align_corners=True)


def convex_upsample(flow, mask, scale_factor=8):
    """
    Upsample flow field [image_h / 4, image_w / 4, 2] -> [image_h, image_w, 2] using convex combination.
    """
    batch_size, _, image_h, image_w = flow.shape
    mask = mask.view(batch_size, 1, 9, scale_factor, scale_factor, image_h, image_w)
    mask = torch.softmax(mask, dim=2)

    up_flow = torch.nn.functional.unfold(flow * scale_factor, [3, 3], padding=1)
    up_flow = up_flow.view(batch_size, 2, 9, 1, 1, image_h, image_w)
    up_flow = torch.sum(mask * up_flow, dim=2)
    up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)

    return up_flow.reshape(batch_size, 2, image_h * scale_factor, image_w * scale_factor)


def resize_flow2d(flow, target_h, target_w):
    origin_h, origin_w = flow.shape[2:]
    if target_h == origin_h and target_w == origin_w:
        return flow
    flow = interpolate(flow, size=(target_h, target_w), mode='bilinear', align_corners=True)
    flow[:, 0] *= target_w / origin_w
    flow[:, 1] *= target_h / origin_h
    return flow


def resize_to_64x(inputs, target):
    n, c, h, w = inputs.shape

    if h % 64 == 0 and w % 64 == 0:
        return inputs, target

    resized_h, resized_w = ((h + 63) // 64) * 64, ((w + 63) // 64) * 64
    inputs = interpolate(inputs, size=(resized_h, resized_w), mode='bilinear', align_corners=True)

    if target is not None:
        target = interpolate(target, size=(resized_h, resized_w), mode='bilinear', align_corners=True)
        target[:, 0] *= resized_w / w
        target[:, 1] *= resized_h / h

    return inputs, target


def pad_to_64x(inputs, target):
    n, c, h, w = inputs.shape

    pad_h = 0 if h % 64 == 0 else 64 - (h % 64)
    pad_w = 0 if w % 64 == 0 else 64 - (w % 64)

    if pad_h == 0 and pad_w == 0:
        return inputs, target

    inputs = pad(inputs, [0, pad_w, 0, pad_h], value=0)
    if target is not None:
        target = pad(target, [0, pad_w, 0, pad_h], value=0)

    return inputs, target


def project_pc2image(pc, camera_info):
    assert pc.shape[1] == 3  # channel first
    batch_size, n_points = pc.shape[0], pc.shape[-1]

    if isinstance(camera_info['cx'], torch.Tensor):
        cx = camera_info['cx'][:, None].expand([batch_size, n_points])
        cy = camera_info['cy'][:, None].expand([batch_size, n_points])
    else:
        cx = camera_info['cx']
        cy = camera_info['cy']

    if camera_info['projection_mode'] == 'perspective':
        f = camera_info['f'][:, None].expand([batch_size, n_points])
        pc_x, pc_y, pc_z = pc[:, 0, :], pc[:, 1, :], pc[:, 2, :]
        image_x = cx + (f / pc_z) * pc_x
        image_y = cy + (f / pc_z) * pc_y
    elif camera_info['projection_mode'] == 'parallel':
        image_x = pc[:, 0, :] + cx
        image_y = pc[:, 1, :] + cy
    else:
        raise NotImplementedError

    return torch.cat([
        image_x[:, None, :],
        image_y[:, None, :],
    ], dim=1)


def grid_sample_wrapper(feat_2d, xy):
    image_h, image_w = feat_2d.shape[2:]
    new_x = 2.0 * xy[:, 0] / (image_w - 1) - 1.0  # [bs, n_points]
    new_y = 2.0 * xy[:, 1] / (image_h - 1) - 1.0  # [bs, n_points]
    new_xy = torch.cat([new_x[:, :, None, None], new_y[:, :, None, None]], dim=-1)  # [bs, n_points, 1, 2]
    result = grid_sample(feat_2d, new_xy, 'bilinear', align_corners=True)  # [bs, n_channels, n_points, 1]
    return result[..., 0]


@torch.no_grad()
def project_feat_with_nn_corr(xy, feat_2d, feat_3d, nn_indices=None):
    batch_size, _, image_h, image_w = feat_2d.shape

    grid = mesh_grid(batch_size, image_h, image_w, xy.device)  # [B, 2, H, W]
    grid = grid.reshape([batch_size, 2, -1])  # [B, 2, HW]

    if nn_indices is None:
        nn_indices = k_nearest_neighbor(xy, grid, k=1)[..., 0]  # [B, HW]
    else:
        assert nn_indices.shape == (batch_size, image_h * image_w)

    nn_feat2d = batch_indexing_channel_first(grid_sample_wrapper(feat_2d, xy), nn_indices)  # [B, n_channels_2d, HW]
    nn_feat3d = batch_indexing_channel_first(feat_3d, nn_indices)  # [B, n_channels_3d, HW]
    nn_offset = batch_indexing_channel_first(xy, nn_indices) - grid  # [B, 2, HW]
    nn_corr = torch.mean(nn_feat2d * feat_2d.reshape(batch_size, -1, image_h * image_w), dim=1, keepdim=True)  # [B, 1, HW]

    final = torch.cat([nn_offset, nn_corr, nn_feat3d], dim=1)  # [B, n_channels_3d + 3, HW]
    final = final.reshape([batch_size, -1, image_h, image_w])

    return final


def perspect2parallel(xyz, perspect_camera_info, parallel_camera_info):
    src_x, src_y, src_z = xyz[:, 0, :], xyz[:, 1, :], xyz[:, 2, :]  # [batch_size, n_points]

    # transformation
    batch_size, n_points = src_x.shape
    f = perspect_camera_info['f'][:, None].expand([batch_size, n_points])
    cx = perspect_camera_info['cx'][:, None].expand([batch_size, n_points])
    cy = perspect_camera_info['cy'][:, None].expand([batch_size, n_points])

    dst_x = cx + (f / src_z) * src_x
    dst_y = cy + (f / src_z) * src_y
    dst_z = f * torch.log(src_z) + 1

    # scaling
    perspect_h, perspect_w = perspect_camera_info['sensor_h'], perspect_camera_info['sensor_w']
    parallel_h, parallel_w = parallel_camera_info['sensor_h'], parallel_camera_info['sensor_w']

    scale_ratio_w = (parallel_w - 1) / (perspect_w - 1)
    scale_ratio_h = (parallel_h - 1) / (perspect_h - 1)

    dst_xyz = torch.cat([
        dst_x[:, None, :] * scale_ratio_w - (parallel_w - 1) / 2,
        dst_y[:, None, :] * scale_ratio_h - (parallel_h - 1) / 2,
        dst_z[:, None, :] * min(scale_ratio_w, scale_ratio_h),
    ], dim=1)

    return dst_xyz


def parallel2perspect(xyz, perspect_camera_info, parallel_camera_info):
    src_x, src_y, src_z = xyz[:, 0, :], xyz[:, 1, :], xyz[:, 2, :]  # [batch_size, n_points]

    # scaling
    perspect_h, perspect_w = perspect_camera_info['sensor_h'], perspect_camera_info['sensor_w']
    parallel_h, parallel_w = parallel_camera_info['sensor_h'], parallel_camera_info['sensor_w']

    scale_ratio_w = (parallel_w - 1) / (perspect_w - 1)
    scale_ratio_h = (parallel_h - 1) / (perspect_h - 1)

    src_x = (src_x + (parallel_w - 1) / 2) / scale_ratio_w
    src_y = (src_y + (parallel_h - 1) / 2) / scale_ratio_h
    src_z = src_z / min(scale_ratio_w, scale_ratio_h)

    # transformation
    batch_size, n_points = src_x.shape
    f = perspect_camera_info['f'][:, None].expand([batch_size, n_points])
    cx = perspect_camera_info['cx'][:, None].expand([batch_size, n_points])
    cy = perspect_camera_info['cy'][:, None].expand([batch_size, n_points])

    dst_z = torch.exp((src_z - 1) / f)
    dst_x = (src_x - cx) * dst_z / f
    dst_y = (src_y - cy) * dst_z / f

    return torch.cat([
        dst_x[:, None, :],
        dst_y[:, None, :],
        dst_z[:, None, :],
    ], dim=1)


def timer_func(func):
    # This function shows the execution time of the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        torch.cuda.synchronize()
        t2 = time.time()

        print(f'Function {func.__qualname__!r} executed in {(t2 - t1) * 1000:.4f}ms')
        return result
    return wrap_func
