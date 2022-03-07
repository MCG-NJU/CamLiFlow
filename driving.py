import os
import cv2
import random
import numpy as np
import torch.utils.data
from utils import load_flow_png
from augmentation import joint_augmentation


class Driving(torch.utils.data.Dataset):
    def __init__(self, cfgs):
        assert os.path.isdir(cfgs.root_dir)

        self.root_dir = str(cfgs.root_dir)
        self.cfgs = cfgs

        self.indices = list(range(1, 300))  # 1, 2, ... 299

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        if not self.cfgs.augmentation.enabled:
            np.random.seed(0)

        idx = self.indices[i]

        focal_type = random.choice(self.cfgs.focal_type)
        moving_dir = random.choice(self.cfgs.moving_dir)
        speed = random.choice(self.cfgs.speed)
        flow_dir = random.choice(self.cfgs.flow_dir)

        if flow_dir == 'into_future':
            idx1 = idx
            idx2 = idx + 1
        elif flow_dir == 'into_past':
            idx1 = idx + 1
            idx2 = idx
        else:
            raise ValueError

        data_dict = {'index': idx1}

        # camera intrinsics
        f = 450.0 if focal_type == '15mm_focallength' else 1050.0
        baseline, cx, cy = 1.0, 479.5, 269.5

        # load data
        pcs = np.load(os.path.join(
            self.root_dir, 'pc', focal_type, moving_dir, speed, flow_dir, '%04d.npz' % idx1
        ))
        flow_2d, flow_mask_2d = load_flow_png(os.path.join(
            self.root_dir, 'flow_2d', focal_type, moving_dir, speed, flow_dir, '%04d.png' % idx1
        ), scale=32.0)
        flow_3d = np.load(os.path.join(
            self.root_dir, 'flow_3d', focal_type, moving_dir, speed, flow_dir, '%04d.npy' % idx1
        ))
        pc1, pc2 = pcs['pc1'], pcs['pc2']

        # depth mask
        mask1 = pc1[..., -1] < self.cfgs.max_depth
        mask2 = pc2[..., -1] < self.cfgs.max_depth
        pc1, pc2, flow_3d = pc1[mask1], pc2[mask2], flow_3d[mask1]

        image1 = cv2.imread(os.path.join(
            self.root_dir, 'image', focal_type, moving_dir, speed, 'left', '%04d.png' % idx1
        ))[..., ::-1]
        image2 = cv2.imread(os.path.join(
            self.root_dir, 'image', focal_type, moving_dir, speed, 'left', '%04d.png' % idx2
        ))[..., ::-1]
        flow_2d = np.concatenate([
            flow_2d, flow_mask_2d[..., None].astype(np.float32)
        ], axis=2)

        image1, image2, pc1, pc2, flow_2d, flow_3d, f, cx, cy = joint_augmentation(
            image1, image2, pc1, pc2, flow_2d, flow_3d, f, cx, cy, self.cfgs.augmentation
        )

        # random sampling
        indices1 = np.random.choice(pc1.shape[0], size=self.cfgs.n_points, replace=pc1.shape[0] < self.cfgs.n_points)
        indices2 = np.random.choice(pc2.shape[0], size=self.cfgs.n_points, replace=pc2.shape[0] < self.cfgs.n_points)
        pc1, pc2, flow_3d = pc1[indices1], pc2[indices2], flow_3d[indices1]

        images = np.concatenate([image1, image2], axis=-1)
        pcs = np.concatenate([pc1, pc2], axis=1)

        data_dict['images'] = images.transpose([2, 0, 1])
        data_dict['flow_2d'] = flow_2d.transpose([2, 0, 1])
        data_dict['pcs'] = pcs.transpose()
        data_dict['flow_3d'] = flow_3d.transpose()
        data_dict['intrinsics'] = np.float32([f, cx, cy])

        return data_dict
