import os
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


class KITTIPointPWC(data.Dataset):
    """ Non-occluded evaluation following PointPWC """
    def __init__(self, remove_ground=True):
        self.root = 'datasets/kitti_scene_flow/training/pointcloud'
        self.remove_ground = remove_ground

        self.DEPTH_THRESHOLD = 35.0
        self.no_corr = True
        self.num_points = 8192
        self.allow_less_points = False

        self.samples = self.make_dataset()
        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        data_dict = {'index': index}

        pc1_loaded, pc2_loaded = self.pc_loader(self.samples[index])
        pc1_transformed, pc2_transformed, sf_transformed = self.process_pc(pc1_loaded, pc2_loaded)
        
        # pack
        pc_pair = np.concatenate([pc1_transformed, pc2_transformed], axis=1)
        data_dict['pcs'] = pc_pair.transpose()
        data_dict['flow_3d'] = sf_transformed.transpose()

        # pass camera params for IDS
        proj_mat = load_calib(os.path.join('datasets/kitti_scene_flow/training/calib_cam_to_cam', '%06d.txt' % index))
        f, cx, cy = proj_mat[0, 0], proj_mat[0, 2], proj_mat[1, 2]
        data_dict['intrinsics'] = np.float32([f, cx, cy])  # f, cx, cy

        # adjust domain range according to mean and std
        data_dict['src_mean'] = np.array([1.9823, -4.0814, 87.4855], dtype=np.float32)  # kitti
        data_dict['src_std'] = np.array([11.1490,  1.3005, 10.9335], dtype=np.float32)
        data_dict['dst_mean'] = np.array([0.079332, 1.8988, 91.909], dtype=np.float32)  # things
        data_dict['dst_std'] = np.array([8.0472,  4.1851, 13.6923], dtype=np.float32)

        return data_dict

    def make_dataset(self):
        do_mapping = True
        root = os.path.realpath(os.path.expanduser(self.root))

        all_paths = sorted(os.walk(root))
        useful_paths = [item[0] for item in all_paths if len(item[1]) == 0]

        try:
            assert (len(useful_paths) == 200)
        except AssertionError:
            print('assert (len(useful_paths) == 200) failed!', len(useful_paths))

        if do_mapping:
            mapping_path = os.path.join(self.root, 'KITTI_mapping.txt')
            with open(mapping_path) as fd:
                lines = fd.readlines()
                lines = [line.strip() for line in lines]
            useful_paths = [path for path in useful_paths if lines[int(os.path.split(path)[-1])] != '']

        res_paths = useful_paths
        return res_paths

    def pc_loader(self, path):
        pc1 = np.load(os.path.join(path, 'pc1.npy'))  #.astype(np.float32)
        pc2 = np.load(os.path.join(path, 'pc2.npy'))  #.astype(np.float32)

        if self.remove_ground:
            is_ground = np.logical_and(pc1[:,1] < -1.4, pc2[:,1] < -1.4)
            not_ground = np.logical_not(is_ground)

            pc1 = pc1[not_ground]
            pc2 = pc2[not_ground]

        return pc1, pc2

    def process_pc(self, pc1, pc2):
        np.random.seed(1)

        if pc1 is None:
            return None, None, None,

        sf = pc2[:, :3] - pc1[:, :3]

        if self.DEPTH_THRESHOLD > 0:
            near_mask = np.logical_and(pc1[:, 2] < self.DEPTH_THRESHOLD, pc2[:, 2] < self.DEPTH_THRESHOLD)
        else:
            near_mask = np.ones(pc1.shape[0], dtype=np.bool)

        indices = np.where(near_mask)[0]
        assert len(indices) > 0

        if self.num_points > 0:
            try:
                sampled_indices1 = np.random.choice(indices, size=self.num_points, replace=False, p=None)
                if self.no_corr:
                    sampled_indices2 = np.random.choice(indices, size=self.num_points, replace=False, p=None)
                else:
                    sampled_indices2 = sampled_indices1
            except ValueError:
                if not self.allow_less_points:
                    #replicate some points
                    sampled_indices1 = np.random.choice(indices, size=self.num_points, replace=True, p=None)
                    if self.no_corr:
                        sampled_indices2 = np.random.choice(indices, size=self.num_points, replace=True, p=None)
                    else:
                        sampled_indices2 = sampled_indices1
                else:
                    sampled_indices1 = indices
                    sampled_indices2 = indices
        else:
            sampled_indices1 = indices
            sampled_indices2 = indices

        pc1 = pc1[sampled_indices1]
        sf = sf[sampled_indices1]
        pc2 = pc2[sampled_indices2]

        return pc1, pc2, sf


class Evaluator:
    def __init__(self, device: torch.device, cfgs: DictConfig):
        self.cfgs = cfgs
        self.device = device

        logging.info('Loading test set from %s' % self.cfgs.testset.root_dir)
        self.test_dataset = KITTIPointPWC()
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

                metrics_3d['counts'] += 1  # averaged over batch (following PointPWC)
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
