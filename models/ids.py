import torch


def persp2paral(xyz, perspect_camera_info, parallel_camera_info):
    """
    Perspective projection -> Parallel projection
    """
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


def paral2persp(xyz, perspect_camera_info, parallel_camera_info):
    """
    Parallel projection -> Perspective projection
    """
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
