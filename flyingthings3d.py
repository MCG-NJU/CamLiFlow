import os
import cv2
import numpy as np
import torch.utils.data
from utils import load_flow_png
from augmentation import joint_augmentation


class FlyingThings3D(torch.utils.data.Dataset):
    def __init__(self, cfgs):
        assert os.path.isdir(cfgs.root_dir)

        self.root_dir = str(cfgs.root_dir)
        self.split = str(cfgs.split)
        self.split_dir = os.path.join(self.root_dir, self.split)
        self.cfgs = cfgs

        self.indices = []
        for filename in os.listdir(os.path.join(self.root_dir, self.split, 'flow_2d')):
            self.indices.append(int(filename.split('.')[0]))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        if not self.cfgs.augmentation.enabled:
            np.random.seed(0)

        idx1 = self.indices[i]
        idx2 = idx1 + 1
        data_dict = {'index': idx1}

        # camera intrinsics
        f, cx, cy = 1050, 479.5, 269.5

        # load data
        pcs = np.load(os.path.join(self.split_dir, 'pc', '%07d.npz' % idx1))
        pc1, pc2 = pcs['pc1'], pcs['pc2']

        flow_2d, flow_mask_2d = load_flow_png(os.path.join(self.split_dir, 'flow_2d', '%07d.png' % idx1))
        flow_3d = np.load(os.path.join(self.split_dir, 'flow_3d', '%07d.npy' % idx1))

        occ_mask_3d = np.load(os.path.join(self.split_dir, 'occ_mask_3d', '%07d.npy' % idx1))
        occ_mask_3d = np.unpackbits(occ_mask_3d, count=len(pc1))

        image1 = cv2.imread(os.path.join(self.split_dir, 'image', '%07d.png' % idx1))[..., ::-1]
        image2 = cv2.imread(os.path.join(self.split_dir, 'image', '%07d.png' % idx2))[..., ::-1]

        # ignore fast moving objects
        flow_mask_2d = np.logical_and(flow_mask_2d, np.linalg.norm(flow_2d, axis=-1) < 250.0)
        flow_2d = np.concatenate([flow_2d, flow_mask_2d[..., None].astype(np.float32)], axis=2)

        image1, image2, pc1, pc2, flow_2d, flow_3d, f, cx, cy = joint_augmentation(
            image1, image2, pc1, pc2, flow_2d, flow_3d, f, cx, cy, self.cfgs.augmentation
        )

        # random sampling during training
        if self.split == 'train':
            indices1 = np.random.choice(pc1.shape[0], size=self.cfgs.n_points, replace=pc1.shape[0] < self.cfgs.n_points)
            indices2 = np.random.choice(pc2.shape[0], size=self.cfgs.n_points, replace=pc2.shape[0] < self.cfgs.n_points)
            pc1, pc2, flow_3d, occ_mask_3d = pc1[indices1], pc2[indices2], flow_3d[indices1], occ_mask_3d[indices1]

        images = np.concatenate([image1, image2], axis=-1)
        pcs = np.concatenate([pc1, pc2], axis=1)

        data_dict['images'] = images.transpose([2, 0, 1])
        data_dict['flow_2d'] = flow_2d.transpose([2, 0, 1])
        data_dict['pcs'] = pcs.transpose()
        data_dict['flow_3d'] = flow_3d.transpose()
        data_dict['occ_mask_3d'] = occ_mask_3d
        data_dict['intrinsics'] = np.float32([f, cx, cy])

        return data_dict
