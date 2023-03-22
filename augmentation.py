import cv2
import torch
import torchvision
import numpy as np


def color_jitter(image1, image2, brightness, contrast, saturation, hue):
    assert image1.shape == image2.shape
    cj_module = torchvision.transforms.ColorJitter(brightness, contrast, saturation, hue)

    images = np.concatenate([image1, image2], axis=0)
    images_t = torch.from_numpy(images.transpose([2, 0, 1]).copy())
    images_t = cj_module.forward(images_t / 255.0) * 255.0
    images = images_t.numpy().astype(np.uint8).transpose(1, 2, 0)
    image1, image2 = images[:image1.shape[0]], images[image1.shape[0]:]

    return image1, image2


def flip_point_cloud(pc, image_h, image_w, f, cx, cy, flip_mode):
    assert flip_mode in ['lr', 'ud']
    pc_x, pc_y, depth = pc[..., 0], pc[..., 1], pc[..., 2]

    image_x = cx + (f / depth) * pc_x
    image_y = cy + (f / depth) * pc_y

    if flip_mode == 'lr':
        image_x = image_w - 1 - image_x
    else:
        image_y = image_h - 1 - image_y

    pc_x = (image_x - cx) * depth / f
    pc_y = (image_y - cy) * depth / f
    pc = np.concatenate([pc_x[:, None], pc_y[:, None], depth[:, None]], axis=-1)

    return pc


def flip_scene_flow(pc1, flow_3d, image_h, image_w, f, cx, cy, flip_mode):
    new_pc1 = flip_point_cloud(pc1, image_h, image_w, f, cx, cy, flip_mode)
    new_pc1_warp = flip_point_cloud(pc1 + flow_3d[:, :3], image_h, image_w, f, cx, cy, flip_mode)
    return np.concatenate([new_pc1_warp - new_pc1, flow_3d[:, 3:]], axis=-1)


def flip_image(image, flip_mode):
    if flip_mode == 'lr':
        return np.fliplr(image).copy()
    else:
        return np.flipud(image).copy()


def flip_optical_flow(flow, flip_mode):
    assert flip_mode in ['lr', 'ud']
    if flip_mode == 'lr':
        flow = np.fliplr(flow).copy()
        flow[:, :, 0] *= -1
    else:
        flow = np.flipud(flow).copy()
        flow[:, :, 1] *= -1
    return flow


def random_flip(image1, image2, pc1, pc2, flow_2d, flow_3d, f, cx, cy, flip_mode):
    assert flow_3d.shape[1] <= 4
    assert flip_mode in ['lr', 'ud']
    image_h, image_w = image1.shape[:2]

    if np.random.rand() < 0.5:  # do nothing
        return image1, image2, pc1, pc2, flow_2d, flow_3d

    # flip images
    new_image1 = flip_image(image1, flip_mode)
    new_image2 = flip_image(image2, flip_mode)

    # flip point clouds
    new_pc1 = flip_point_cloud(pc1, image_h, image_w, f, cx, cy, flip_mode)
    new_pc2 = flip_point_cloud(pc2, image_h, image_w, f, cx, cy, flip_mode)

    # flip optical flow and scene flow
    new_flow_2d = flip_optical_flow(flow_2d, flip_mode)
    new_flow_3d = flip_scene_flow(pc1, flow_3d, image_h, image_w, f, cx, cy, flip_mode)

    return new_image1, new_image2, new_pc1, new_pc2, new_flow_2d, new_flow_3d


def crop_image_with_pc(image1, image2, pc1, pc2, flow_2d, flow_3d, f, cx, cy, crop_window, drop_pc=False):
    assert len(crop_window) == 4  # [x1, y1, x2, y2]

    x1, y1, x2, y2 = crop_window
    image_h, image_w = image1.shape[:2]

    # project points to image plane
    cx = (image_w - 1) / 2 if cx is None else cx
    cy = (image_h - 1) / 2 if cy is None else cy

    xyz1_x, xyz1_y, xyz1_z = pc1[..., 0], pc1[..., 1], pc1[..., 2]
    xyz2_x, xyz2_y, xyz2_z = pc2[..., 0], pc2[..., 1], pc2[..., 2]

    xy1_x = cx + (f / xyz1_z) * xyz1_x
    xy1_y = cy + (f / xyz1_z) * xyz1_y
    xy2_x = cx + (f / xyz2_z) * xyz2_x
    xy2_y = cy + (f / xyz2_z) * xyz2_y

    # crop images
    image1 = image1[y1:y2, x1:x2].copy()
    image2 = image2[y1:y2, x1:x2].copy()
    flow_2d = flow_2d[y1:y2, x1:x2].copy()

    # crop points
    if drop_pc:
        crop_mask1 = np.where(np.logical_and(
            np.logical_and(xy1_x > x1, xy1_x < x2),
            np.logical_and(xy1_y > y1, xy1_y < y2)
        ))[0]
        crop_mask2 = np.where(np.logical_and(
            np.logical_and(xy2_x > x1, xy2_x < x2),
            np.logical_and(xy2_y > y1, xy2_y < y2)
        ))[0]
        pc1, pc2, flow_3d = pc1[crop_mask1], pc2[crop_mask2], flow_3d[crop_mask1]

        if pc1.shape[0] == 0 or pc2.shape[0] == 0:
            raise AssertionError

    # adjust camera params
    cx = cx - x1
    cy = cy - y1

    return image1, image2, pc1, pc2, flow_2d, flow_3d, f, cx, cy


def random_crop(image1, image2, pc1, pc2, flow_2d, flow_3d, f, cx, cy, crop_size, drop_pc):
    assert flow_3d.shape[1] <= 4
    assert len(crop_size) == 2
    crop_w, crop_h = crop_size

    image_h, image_w = image1.shape[:2]
    assert crop_w <= image_w and crop_h <= image_h

    # top left of the cropping window
    x1 = np.random.randint(low=0, high=image_w - crop_w + 1)
    y1 = np.random.randint(low=0, high=image_h - crop_h + 1)
    crop_window = [x1, y1, x1 + crop_w, y1 + crop_h]

    return crop_image_with_pc(image1, image2, pc1, pc2, flow_2d, flow_3d, f, cx, cy, crop_window, drop_pc)


def resize_sparse_flow_map(flow, target_w, target_h):
    curr_h, curr_w = flow.shape[:2]

    coords = np.meshgrid(np.arange(curr_w), np.arange(curr_h))
    coords = np.stack(coords, axis=-1).astype(np.float32)

    mask = flow[..., -1] > 0
    coords0, flow0 = coords[mask], flow[mask][:, :2]

    scale_ratio_w = (target_w - 1) / (curr_w - 1)
    scale_ratio_h = (target_h - 1) / (curr_h - 1)

    coords1 = coords0 * [scale_ratio_w, scale_ratio_h]
    flow1 = flow0 * [scale_ratio_w, scale_ratio_h]

    xx = np.round(coords1[:, 0]).astype(np.int32)
    yy = np.round(coords1[:, 1]).astype(np.int32)
    valid = (xx >= 0) & (xx < target_w) & (yy >= 0) & (yy < target_h)
    xx, yy, flow1 = xx[valid], yy[valid], flow1[valid]

    flow_resized = np.zeros([target_h, target_w, 3], dtype=np.float32)
    flow_resized[yy, xx, :2] = flow1
    flow_resized[yy, xx, 2:] = 1.0

    return flow_resized


def random_scale(image1, image2, pc1, pc2, flow_2d, flow_3d, f, cx, cy, scale_range):
    assert len(scale_range) == 2
    assert 1 <= scale_range[0] < scale_range[1]

    if np.random.rand() < 0.5:
        return image1, image2, pc1, pc2, flow_2d, flow_3d, f, cx, cy

    scale_ratio = np.random.uniform(scale_range[0], scale_range[1])
    image_h, image_w = image1.shape[:2]
    crop_h, crop_w = int(image_h / scale_ratio), int(image_w / scale_ratio)

    # top left of the cropping window
    x1 = np.random.randint(low=0, high=image_w - crop_w + 1)
    y1 = np.random.randint(low=0, high=image_h - crop_h + 1)
    crop_window = [x1, y1, x1 + crop_w, y1 + crop_h]

    image1, image2, pc1, pc2, flow_2d, flow_3d, f, cx, cy = crop_image_with_pc(
        image1, image2, pc1, pc2, flow_2d, flow_3d, f, cx, cy, crop_window
    )

    # resize images and optical flow
    image1 = cv2.resize(image1, (image_w, image_h), interpolation=cv2.INTER_LINEAR)
    image2 = cv2.resize(image2, (image_w, image_h), interpolation=cv2.INTER_LINEAR)
    flow_2d = resize_sparse_flow_map(flow_2d, image_w, image_h)

    # resize points and scene flow
    scale_ratio_w = (image_w - 1) / (crop_w - 1)
    scale_ratio_h = (image_h - 1) / (crop_h - 1)
    pc1[:, 0] *= scale_ratio_w
    pc1[:, 1] *= scale_ratio_h
    pc2[:, 0] *= scale_ratio_w
    pc2[:, 1] *= scale_ratio_h
    flow_3d[:, 0] *= scale_ratio_w
    flow_3d[:, 1] *= scale_ratio_h

    # adjust camera params
    cx *= scale_ratio_w
    cy *= scale_ratio_h

    return image1, image2, pc1, pc2, flow_2d, flow_3d, f, cx, cy


def joint_augmentation(image1, image2, pc1, pc2, flow_2d, flow_3d, f, cx, cy, cfgs):
    if not cfgs.enabled:
        return image1, image2, pc1, pc2, flow_2d, flow_3d, f, cx, cy

    if cfgs.color_jitter.enabled:
        image1, image2 = color_jitter(
            image1, image2,
            brightness=cfgs.color_jitter.brightness,
            contrast=cfgs.color_jitter.contrast,
            saturation=cfgs.color_jitter.saturation,
            hue=cfgs.color_jitter.hue,
        )

    if cfgs.random_horizontal_flip.enabled:
        image1, image2, pc1, pc2, flow_2d, flow_3d = random_flip(
            image1, image2, pc1, pc2, flow_2d, flow_3d, f, cx, cy, flip_mode='lr'
        )

    if cfgs.random_vertical_flip.enabled:
        image1, image2, pc1, pc2, flow_2d, flow_3d = random_flip(
            image1, image2, pc1, pc2, flow_2d, flow_3d, f, cx, cy, flip_mode='ud'
        )

    if cfgs.random_crop.enabled:
        image1, image2, pc1, pc2, flow_2d, flow_3d, f, cx, cy = random_crop(
            image1, image2, pc1, pc2, flow_2d, flow_3d, f, cx, cy,
            crop_size=cfgs.random_crop.crop_size,
            drop_pc=cfgs.random_crop.drop_pc
        )

    if cfgs.random_scale.enabled:
        image1, image2, pc1, pc2, flow_2d, flow_3d, f, cx, cy = random_scale(
            image1, image2, pc1, pc2, flow_2d, flow_3d, f, cx, cy,
            scale_range=cfgs.random_scale.scale_range
        )

    return image1, image2, pc1, pc2, flow_2d, flow_3d, f, cx, cy
