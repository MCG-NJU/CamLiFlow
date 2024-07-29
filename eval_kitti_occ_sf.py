import os
import glob
import utils
import hydra
import shutil
import logging
import torch
import torch.optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig
from factory import model_factory
from utils import copy_to_device, size_of_batch, load_calib


class KITTIFlowNet3d(data.Dataset):
    """ Occluded evaluation following FlowNet3D """
    def __init__(self, root='datasets/kitti_scene_flow/training/kitti_rm_ground', npoints=8192):
        self.npoints = npoints
        self.root = root
        self.datapath = glob.glob(os.path.join(self.root, '*.npz'))
        self.cache = {}
        self.cache_size = 30000

    def __getitem__(self, index):
        np.random.seed(1)
        data_dict = {'index': index}

        if index in self.cache:
            pos1, pos2, flow = self.cache[index]
        else:
            fn = self.datapath[index]

            with open(fn, 'rb') as fp:
                data = np.load(fp)
                pos1 = data['pos1']
                pos2 = data['pos2']
                flow = data['gt']

            if len(self.cache) < self.cache_size:
                self.cache[index] = (pos1, pos2, flow)

            n1 = pos1.shape[0]
            n2 = pos2.shape[0]

            if n1 >= self.npoints:
                sample_idx1 = np.random.choice(n1, self.npoints, replace=False)
            else:
                sample_idx1 = np.concatenate((np.arange(n1), np.random.choice(n1, self.npoints - n1, replace=True)), axis=-1)

            if n2 >= self.npoints:
                sample_idx2 = np.random.choice(n2, self.npoints, replace=False)
            else:
                sample_idx2 = np.concatenate((np.arange(n2), np.random.choice(n2, self.npoints - n2, replace=True)), axis=-1)

            pos1_ = np.copy(pos1)[sample_idx1, :]
            pos2_ = np.copy(pos2)[sample_idx2, :]
            flow_ = np.copy(flow)[sample_idx1, :]

        # pack
        xyz_order = [1, 2, 0]
        pos1_ = pos1_[:, xyz_order]
        pos2_ = pos2_[:, xyz_order]
        flow_ = flow_[:, xyz_order]
        pc_pair = np.concatenate([pos1_, pos2_], axis=1)
        data_dict['pcs'] = pc_pair.transpose().astype(np.float32)
        data_dict['flow_3d'] = flow_.transpose().astype(np.float32)

        # pass camera params for IDS
        proj_mat = load_calib(os.path.join('datasets/kitti_scene_flow/training/calib_cam_to_cam', '%06d.txt' % index))
        f, cx, cy = proj_mat[0, 0], proj_mat[0, 2], proj_mat[1, 2]
        data_dict['intrinsics'] = np.float32([f, cx, cy])  # f, cx, cy

        # adjust domain range according to mean and std
        data_dict['src_mean'] = np.array([3.8450, -3.6596, 86.1627], dtype=np.float32)  # kitti
        data_dict['src_std'] = np.array([10.1774,  1.2327, 13.5970], dtype=np.float32)
        data_dict['dst_mean'] = np.array([0.079332, 1.8988, 91.909], dtype=np.float32)  # things
        data_dict['dst_std'] = np.array([8.0472,  4.1851, 13.6923], dtype=np.float32)

        return data_dict

    def __len__(self):
        return len(self.datapath)


class Evaluator:
    def __init__(self, device: torch.device, cfgs: DictConfig):
        self.cfgs = cfgs
        self.device = device

        logging.info('Loading test set from %s' % self.cfgs.testset.root_dir)
        self.test_dataset = KITTIFlowNet3d()
        self.test_loader = utils.FastDataLoader(
            dataset=self.test_dataset,
            batch_size=8,
            num_workers=self.cfgs.testset.n_workers
        )

        logging.info('Creating model: %s' % self.cfgs.model.name)
        self.model = model_factory(self.cfgs.model).to(device=self.device)
        self.model.eval()

        logging.info('Loading checkpoint from %s' % self.cfgs.ckpt.path)
        checkpoint = torch.load(self.cfgs.ckpt.path, self.device)
        self.model.load_state_dict(checkpoint['state_dict'], strict=self.cfgs.ckpt.strict)

    @torch.no_grad()
    def run(self):
        logging.info('Running evaluation...')
        metrics_3d = {'counts': 0, 'EPE3d': 0.0, 'AccS': 0.0, 'AccR': 0.0, 'Outlier': 0.0}

        for inputs in tqdm(self.test_loader):
            inputs = copy_to_device(inputs, self.device)
            
            with torch.cuda.amp.autocast(enabled=False):
                outputs = self.model.forward(inputs)

            for batch_id in range(size_of_batch(inputs)):
                flow_3d_pred = outputs['flow_3d'][batch_id]
                flow_3d_target = inputs['flow_3d'][batch_id]

                epe3d_map = torch.sqrt(torch.sum((flow_3d_pred - flow_3d_target) ** 2, dim=0))
                gt_norm = torch.linalg.norm(flow_3d_target, axis=0)
                relative_err = epe3d_map / (gt_norm + 1e-4)

                acc3d_strict = torch.logical_or(epe3d_map < 0.05, relative_err < 0.05)
                acc3d_relax = torch.logical_or(epe3d_map < 0.1, relative_err < 0.1)
                outlier = torch.logical_or(epe3d_map > 0.3, relative_err > 0.1)

                metrics_3d['counts'] += 1  # averaged over batch (following FlowNet3D)
                metrics_3d['EPE3d'] += epe3d_map.sum().item() / epe3d_map.shape[0]
                metrics_3d['AccS'] += torch.count_nonzero(acc3d_strict).item() / epe3d_map.shape[0]
                metrics_3d['AccR'] += torch.count_nonzero(acc3d_relax).item() / epe3d_map.shape[0]
                metrics_3d['Outlier'] += torch.count_nonzero(outlier).item() / epe3d_map.shape[0]

        logging.info('#### 3D Metrics ####')
        logging.info('EPE: %.3f' % (metrics_3d['EPE3d'] / metrics_3d['counts']))
        logging.info('AccS: %.2f%%' % (metrics_3d['AccS'] / metrics_3d['counts'] * 100.0))
        logging.info('AccR: %.2f%%' % (metrics_3d['AccR'] / metrics_3d['counts'] * 100.0))
        logging.info('Outlier: %.2f%%' % (metrics_3d['Outlier'] / metrics_3d['counts'] * 100.0))


@hydra.main(config_path='conf', config_name='evaluator')
def main(cfgs: DictConfig):
    utils.init_logging()

    # change working directory
    shutil.rmtree(os.getcwd(), ignore_errors=True)
    os.chdir(hydra.utils.get_original_cwd())

    if torch.cuda.device_count() == 0:
        device = torch.device('cpu')
        logging.info('No CUDA device detected, using CPU for evaluation')
    elif torch.cuda.device_count() == 1:
        device = torch.device('cuda:0')
        logging.info('Using GPU: %s' % torch.cuda.get_device_name(device))
        cudnn.benchmark = True
    else:
        raise RuntimeError('Evaluation script does not support multi-GPU systems.')

    evaluator = Evaluator(device, cfgs)
    evaluator.run()


if __name__ == '__main__':
    main()
