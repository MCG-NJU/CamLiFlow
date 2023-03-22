import os
import cv2
import glob
import numpy as np
import torch.utils.data
from utils import load_flow


# Unofficial train-val split from https://github.com/lliuz/ARFlow/blob/master/datasets/flow_datasets.py#L94
TRAIN_SCENES = ['alley_1', 'ambush_4', 'ambush_6', 'ambush_7', 'bamboo_2', 'bandage_2', 'cave_2',
                'market_2', 'market_5', 'shaman_2', 'sleeping_2', 'temple_3']
VALIDATE_SCENES = ['alley_2', 'ambush_2', 'ambush_5', 'bamboo_1', 'bandage_1', 'cave_4', 'market_6',
                   'mountain_1', 'shaman_3', 'sleeping_1', 'temple_2']


def depth_read(filename):
    """ Read depth data from file, return as numpy array.
    from datasets/sintel/stereo/sdk/python/sintel_io.py
    """
    f = open(filename,'rb')

    TAG_FLOAT = 202021.25
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    
    width = np.fromfile(f,dtype=np.int32,count=1)[0]
    height = np.fromfile(f,dtype=np.int32,count=1)[0]
    size = width*height
    assert width > 0 and height > 0 and size > 1 and size < 100000000, ' depth_read:: Wrong input size (width = {0}, height = {1}).'.format(width,height)
    
    depth = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width))
    return depth


def depth2pc(depth, f, cx, cy):
    h, w = depth.shape

    xx = np.tile(np.arange(w, dtype=np.float32)[None, :], (h, 1))
    yy = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))

    x = (xx - cx) * depth / f
    y = (yy - cy) * depth / f

    pc = np.concatenate([
        x[:, :, None], y[:, :, None], depth[:, :, None]
    ], axis=-1)

    return pc


class Sintel(torch.utils.data.Dataset):
    def __init__(self, cfgs):
        assert os.path.isdir(cfgs.root_dir)
        assert cfgs.pass_name in ['clean', 'final', 'clean_final']

        self.dataset_dir = cfgs.root_dir
        self.split = cfgs.split
        self.cfgs = cfgs

        if self.split == 'train':
            scene_names = TRAIN_SCENES
        elif self.split == 'val':
            scene_names = VALIDATE_SCENES
        elif self.split == 'trainval':
            scene_names = TRAIN_SCENES + VALIDATE_SCENES
        else:
            raise ValueError

        self.samples = []
        for pass_name in ['clean', 'final']:
            if pass_name not in self.cfgs.pass_name:
                continue
            for scene_name in scene_names:
                image_dir = os.path.join(self.dataset_dir, 'flow', 'training', pass_name, scene_name)
                depth_dir = os.path.join(self.dataset_dir, 'depth', 'training', 'depth', scene_name)
                flow_dir = os.path.join(self.dataset_dir, 'flow', 'training', 'flow', scene_name)

                image_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))
                depth_paths = sorted(glob.glob(os.path.join(depth_dir, '*.dpt')))
                flow_paths = sorted(glob.glob(os.path.join(flow_dir, '*.flo')))
                assert len(image_paths) == len(depth_paths) == len(flow_paths) + 1

                for i, flow_path in enumerate(flow_paths):
                    image1_path, image2_path = image_paths[i], image_paths[i + 1]
                    depth1_path, depth2_path = depth_paths[i], depth_paths[i + 1]
                    
                    self.samples.append({
                        'image1_path': image1_path,
                        'image2_path': image2_path,
                        'depth1_path': depth1_path,
                        'depth2_path': depth2_path,
                        'flow_path': flow_path,
                        'scene_name': scene_name,
                        'sample_name': os.path.basename(image1_path).split('.')[0],
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        np.random.seed(0)
        sample = self.samples[index]

        data_dict = {
            'index': index,
            'scene_name': sample['scene_name'],
            'sample_name': sample['sample_name']
        }

        image1_path = sample['image1_path']
        image2_path = sample['image2_path']
        depth1_path = sample['depth1_path']
        depth2_path = sample['depth2_path']
        flow_path = sample['flow_path']

        image1, image2 = cv2.imread(image1_path), cv2.imread(image2_path)
        depth1, depth2 = depth_read(depth1_path), depth_read(depth2_path)
        flow_2d = load_flow(flow_path)

        # make sure there are enough input points
        while min(np.count_nonzero(depth1 < self.cfgs.max_depth),
                  np.count_nonzero(depth2 < self.cfgs.max_depth)) < self.cfgs.n_points:
            depth1 *= 0.1
            depth2 *= 0.1

        min_depth = min(np.min(depth1), np.min(depth2))
        depth1 += 5 - min_depth
        depth2 += 5 - min_depth

        # generate point clouds
        f, cx, cy = 1500.0, 511.5, 217.5
        pc1 = depth2pc(depth1, f, cx, cy)
        pc2 = depth2pc(depth2, f, cx, cy)
        flow_3d = np.zeros_like(pc1)

        # limit max depth
        mask1 = (pc1[..., -1] < max(np.min(pc1[..., -1]) + 1, self.cfgs.max_depth))
        mask2 = (pc2[..., -1] < max(np.min(pc2[..., -1]) + 1, self.cfgs.max_depth))
        pc1, pc2, flow_3d = pc1[mask1], pc2[mask2], flow_3d[mask1]

        # random sampling
        indices1 = np.random.choice(pc1.shape[0], size=self.cfgs.n_points, replace=pc1.shape[0] < self.cfgs.n_points)
        indices2 = np.random.choice(pc2.shape[0], size=self.cfgs.n_points, replace=pc2.shape[0] < self.cfgs.n_points)
        pc1, pc2, flow_3d = pc1[indices1], pc2[indices2], flow_3d[indices1]

        pc_pair = np.concatenate([pc1, pc2], axis=1)
        data_dict['pcs'] = pc_pair.transpose()
        data_dict['flow_3d'] = flow_3d.transpose()
        data_dict['intrinsics'] = np.float32([f, cx, cy])

        image1, image2 = image1[..., ::-1], image2[..., ::-1]
        image_pair = np.concatenate([image1, image2], axis=-1)
        data_dict['images'] = image_pair.transpose([2, 0, 1])
        data_dict['flow_2d'] = flow_2d.transpose([2, 0, 1])

        return data_dict
