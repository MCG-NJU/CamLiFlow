import os
import cv2
import numpy as np
import torch.utils.data
from utils import disp2pc, project_pc2image, load_flow_png, load_disp_png, load_calib, zero_padding
from augmentation import joint_augmentation


class KITTI(torch.utils.data.Dataset):
    def __init__(self, cfgs):
        assert os.path.isdir(cfgs.root_dir)
        assert cfgs.split in ['training200', 'training160', 'training40', 'testing200']

        if 'training' in cfgs.split:
            self.root_dir = os.path.join(cfgs.root_dir, 'training')
        else:
            self.root_dir = os.path.join(cfgs.root_dir, 'testing')

        self.split = cfgs.split
        self.cfgs = cfgs

        if self.split == 'training200' or self.split == 'testing200':
            self.indices = np.arange(200)
        elif self.split == 'training160':
            self.indices = [i for i in range(200) if i % 5 != 0]
        elif self.split == 'training40':
            self.indices = [i for i in range(200) if i % 5 == 0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        if not self.cfgs.augmentation.enabled:
            np.random.seed(23333)

        index = self.indices[i]
        data_dict = {'index': index}

        proj_mat = load_calib(os.path.join(self.root_dir, 'calib_cam_to_cam', '%06d.txt' % index))
        f, cx, cy = proj_mat[0, 0], proj_mat[0, 2], proj_mat[1, 2]

        image1 = cv2.imread(os.path.join(self.root_dir, 'image_2', '%06d_10.png' % index))[..., ::-1]
        image2 = cv2.imread(os.path.join(self.root_dir, 'image_2', '%06d_11.png' % index))[..., ::-1]
        data_dict['input_h'] = image1.shape[0]
        data_dict['input_w'] = image1.shape[1]

        flow_2d, flow_2d_mask = load_flow_png(os.path.join(self.root_dir, 'flow_occ', '%06d_10.png' % index))

        disp1, mask1 = load_disp_png(os.path.join(self.root_dir, 'disp_occ_0', '%06d_10.png' % index))
        disp2, mask2 = load_disp_png(os.path.join(self.root_dir, 'disp_occ_1', '%06d_10.png' % index))
        mask = np.logical_and(np.logical_and(mask1, mask2), flow_2d_mask)

        pc1 = disp2pc(disp1, baseline=0.54, f=f, cx=cx, cy=cy)[mask]
        pc2 = disp2pc(disp2, baseline=0.54, f=f, cx=cx, cy=cy, flow=flow_2d)[mask]
        flow_3d = pc2 - pc1
        flow_3d_mask = np.ones(flow_3d.shape[0], dtype=np.float32)

        # remove out-of-boundary regions of pc2 to create occlusion
        image_h, image_w = disp2.shape[:2]
        xy2 = project_pc2image(pc2, image_h, image_w, f, cx, cy, clip=False)
        boundary_mask = np.logical_and(
            np.logical_and(xy2[..., 0] >= 0, xy2[..., 0] < image_w),
            np.logical_and(xy2[..., 1] >= 0, xy2[..., 1] < image_h)
        )
        pc2 = pc2[boundary_mask]

        flow_2d = np.concatenate([flow_2d, flow_2d_mask[..., None].astype(np.float32)], axis=-1)
        flow_3d = np.concatenate([flow_3d, flow_3d_mask[..., None].astype(np.float32)], axis=-1)

        # images from KITTI have various sizes, padding them to a unified size of 1242x376
        padding_h, padding_w = 376, 1242
        image1 = zero_padding(image1, padding_h, padding_w)
        image2 = zero_padding(image2, padding_h, padding_w)
        flow_2d = zero_padding(flow_2d, padding_h, padding_w)

        # data augmentation
        image1, image2, pc1, pc2, flow_2d, flow_3d, f, cx, cy = joint_augmentation(
            image1, image2, pc1, pc2, flow_2d, flow_3d, f, cx, cy, self.cfgs.augmentation,
        )

        # random sampling
        indices1 = np.random.choice(pc1.shape[0], size=self.cfgs.n_points, replace=pc1.shape[0] < self.cfgs.n_points)
        indices2 = np.random.choice(pc2.shape[0], size=self.cfgs.n_points, replace=pc2.shape[0] < self.cfgs.n_points)
        pc1, pc2, flow_3d = pc1[indices1], pc2[indices2], flow_3d[indices1]

        pcs = np.concatenate([pc1, pc2], axis=1)
        images = np.concatenate([image1, image2], axis=-1)

        data_dict['images'] = images.transpose([2, 0, 1])
        data_dict['flow_2d'] = flow_2d.transpose([2, 0, 1])
        data_dict['pcs'] = pcs.transpose()
        data_dict['flow_3d'] = flow_3d.transpose()
        data_dict['intrinsics'] = np.float32([f, cx, cy])

        return data_dict


class KITTITest(torch.utils.data.Dataset):
    def __init__(self, cfgs):
        assert os.path.isdir(cfgs.root_dir)
        assert cfgs.split in ['testing200']

        self.root_dir = os.path.join(cfgs.root_dir, 'testing')
        self.split = cfgs.split
        self.cfgs = cfgs

    def __len__(self):
        return 200

    def __getitem__(self, index):
        np.random.seed(23333)
        data_dict = {'index': index}

        proj_mat = load_calib(os.path.join(self.root_dir, 'calib_cam_to_cam', '%06d.txt' % index))
        f, cx, cy = proj_mat[0, 0], proj_mat[0, 2], proj_mat[1, 2]

        image1 = cv2.imread(os.path.join(self.root_dir, 'image_2', '%06d_10.png' % index))[..., ::-1]
        image2 = cv2.imread(os.path.join(self.root_dir, 'image_2', '%06d_11.png' % index))[..., ::-1]
        data_dict['input_h'] = image1.shape[0]
        data_dict['input_w'] = image1.shape[1]

        disp1, mask1 = load_disp_png(os.path.join(self.root_dir, 'disp_%s' % self.cfgs.disp_provider, '%06d_10.png' % index))
        disp2, mask2 = load_disp_png(os.path.join(self.root_dir, 'disp_%s' % self.cfgs.disp_provider, '%06d_11.png' % index))

        # ignore top 110 rows without evaluation
        mask1[:110] = 0
        mask2[:110] = 0

        pc1 = disp2pc(disp1, baseline=0.54, f=f, cx=cx, cy=cy)[mask1]
        pc2 = disp2pc(disp2, baseline=0.54, f=f, cx=cx, cy=cy)[mask2]

        # limit max height (2.0m)
        pc1 = pc1[pc1[..., 1] > -2.0]
        pc2 = pc2[pc2[..., 1] > -2.0]

        # limit max depth
        pc1 = pc1[pc1[..., -1] < self.cfgs.max_depth]
        pc2 = pc2[pc2[..., -1] < self.cfgs.max_depth]

        # images from KITTI have various sizes, padding them to a unified size of 1242x376
        padding_h, padding_w = 376, 1242
        image1 = zero_padding(image1, padding_h, padding_w)
        image2 = zero_padding(image2, padding_h, padding_w)

        # random sampling
        indices1 = np.random.choice(pc1.shape[0], size=self.cfgs.n_points, replace=pc1.shape[0] < self.cfgs.n_points)
        indices2 = np.random.choice(pc2.shape[0], size=self.cfgs.n_points, replace=pc2.shape[0] < self.cfgs.n_points)
        pc1, pc2 = pc1[indices1], pc2[indices2]

        pcs = np.concatenate([pc1, pc2], axis=1)
        images = np.concatenate([image1, image2], axis=-1)

        data_dict['images'] = images.transpose([2, 0, 1])
        data_dict['pcs'] = pcs.transpose()
        data_dict['intrinsics'] = np.float32([f, cx, cy])

        return data_dict
