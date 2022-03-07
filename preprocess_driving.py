import os
import shutil
import logging
import argparse
import torch.utils.data
import numpy as np
from tqdm import tqdm
from utils import init_logging, load_fpm, disp2pc, save_flow_png


'''
Download the "Driving" dataset from:
https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html

You need to download the following parts:

* RGB images (finalpass)
* Disparity
* Disparity change
* Optical flow

Uncompress them and organize the structure of directory as follows:

/mnt/data/driving
├── disparity
├── disparity_change
├── frames_finalpass
└── optical_flow

Then preprocess the data:

python preprocess_driving.py --input_dir /mnt/data/driving
'''


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', required=True, help='Path to the Driving dataset')
parser.add_argument('--output_dir', required=False, default='datasets/driving')
parser.add_argument('--n_points', required=False, default=65536)
parser.add_argument('--max_depth', required=False, default=100.0)
args = parser.parse_args()


class Preprocessor(torch.utils.data.Dataset):
    def __init__(self, input_dir, output_dir, n_points, max_depth):
        super(Preprocessor, self).__init__()

        self.input_dir = input_dir
        self.output_dir = output_dir
        self.n_points = n_points
        self.max_depth = max_depth

        self.samples = []
        for focal_type in ['15mm_focallength', '35mm_focallength']:
            for moving_dir in ['scene_forwards', 'scene_backwards']:
                for speed in ['slow', 'fast']:
                    for flow_dir in ['into_future', 'into_past']:
                        for idx in range(1, 300):
                            self.samples.append({
                                'focal_type': focal_type,
                                'moving_dir': moving_dir,
                                'speed': speed,
                                'flow_dir': flow_dir,
                                'idx': idx,
                            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        sample = self.samples[i]

        focal_type = sample['focal_type']
        moving_dir = sample['moving_dir']
        speed = sample['speed']
        flow_dir = sample['flow_dir']
        idx = sample['idx']

        if flow_dir == 'into_future':
            idx1 = idx
            idx2 = idx + 1
        elif flow_dir == 'into_past':
            idx1 = idx + 1
            idx2 = idx
        else:
            raise ValueError

        # camera intrinsics
        baseline, cx, cy = 1.0, 479.5, 269.5
        f = 450.0 if sample['focal_type'] == '15mm_focallength' else 1050.0

        disp1 = load_fpm(os.path.join(
            self.input_dir, 'disparity', focal_type, moving_dir, speed, 'left', '%04d.pfm' % idx1
        ))
        disp2 = load_fpm(os.path.join(
            self.input_dir, 'disparity', focal_type, moving_dir, speed, 'left', '%04d.pfm' % idx2
        ))
        disp1_c = load_fpm(os.path.join(
            self.input_dir, 'disparity_change', focal_type, moving_dir, speed, flow_dir, 'left', '%04d.pfm' % idx1
        ))
        flow_2d = load_fpm(os.path.join(
            self.input_dir, 'optical_flow', focal_type, moving_dir, speed, flow_dir, 'left',
            'OpticalFlowInto%s_%04d_L.pfm' % ('Future' if flow_dir == 'into_future' else 'Past', idx1)
        ))[..., :2]

        pc1 = disp2pc(disp1, baseline, f, cx, cy)
        pc2 = disp2pc(disp2, baseline, f, cx, cy)
        flow_3d = disp2pc(disp1 + disp1_c, baseline, f, cx, cy, flow_2d) - pc1

        # apply depth mask
        mask1 = (pc1[..., -1] < self.max_depth)
        mask2 = (pc2[..., -1] < self.max_depth)
        pc1, pc2, flow_3d = pc1[mask1], pc2[mask2], flow_3d[mask1]

        # random sampling
        indices1 = np.random.choice(pc1.shape[0], size=min(self.n_points, pc1.shape[0]), replace=False)
        indices2 = np.random.choice(pc2.shape[0], size=min(self.n_points, pc2.shape[0]), replace=False)
        pc1, pc2, flow_3d = pc1[indices1], pc2[indices2], flow_3d[indices1]

        # NaN check
        mask1 = np.logical_not(np.isnan(np.sum(pc1, axis=-1) + np.sum(flow_3d, axis=-1)))
        mask2 = np.logical_not(np.isnan(np.sum(pc2, axis=-1)))
        pc1, pc2, flow_3d = pc1[mask1], pc2[mask2], flow_3d[mask1]

        os.makedirs(os.path.join(self.output_dir, 'pc', focal_type, moving_dir, speed, flow_dir), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'flow_2d', focal_type, moving_dir, speed, flow_dir), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'flow_3d', focal_type, moving_dir, speed, flow_dir), exist_ok=True)

        # save point clouds and occ mask
        np.savez(
            os.path.join(self.output_dir, 'pc', focal_type, moving_dir, speed, flow_dir, '%04d.npz' % idx1),
            pc1=pc1, pc2=pc2
        )

        # mask regions moving extremely fast
        flow_mask = np.logical_and(np.abs(flow_2d[..., 0]) < 1000, np.abs(flow_2d[..., 1]) < 1000)
        flow_2d[np.logical_not(flow_mask)] = 0.0

        # save ground-truth flow
        save_flow_png(
            os.path.join(self.output_dir, 'flow_2d', focal_type, moving_dir, speed, flow_dir, '%04d.png' % idx1),
            flow_2d, flow_mask, scale=32.0
        )
        np.save(
            os.path.join(self.output_dir, 'flow_3d', focal_type, moving_dir, speed, flow_dir, '%04d.npy' % idx1),
            flow_3d
        )

        return 0


def main():
    os.makedirs(os.path.join(args.output_dir, 'pc'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'flow_2d'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'flow_3d'), exist_ok=True)

    logging.info('Copying images...')
    if not os.path.exists(os.path.join(args.output_dir, 'image')):
        shutil.copytree(
            src=os.path.join(args.input_dir, 'frames_finalpass'),
            dst=os.path.join(args.output_dir, 'image')
        )

    logging.info('Generating point clouds...')
    preprocessor = Preprocessor(
        args.input_dir,
        args.output_dir,
        args.n_points,
        args.max_depth,
    )
    preprocessor = torch.utils.data.DataLoader(dataset=preprocessor, num_workers=4)

    for _ in tqdm(preprocessor):
        pass


if __name__ == '__main__':
    init_logging()
    main()
    logging.info('All done.')
