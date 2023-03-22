import os
import cv2
import utils
import hydra
import shutil
import logging
import torch
import torch.optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig
from factory import model_factory
from kitti import KITTITest
from models.utils import mesh_grid, knn_interpolation
from utils import copy_to_device, size_of_batch, save_flow_png, save_disp_png, load_disp_png, disp2pc


class Evaluator:
    def __init__(self, device: torch.device, cfgs: DictConfig):
        self.cfgs = cfgs
        self.device = device

        logging.info('Loading test set from %s' % self.cfgs.testset.root_dir)
        logging.info('Dataset split: %s' % self.cfgs.testset.split)
        self.test_dataset = KITTITest(self.cfgs.testset)

        self.test_loader = utils.FastDataLoader(
            dataset=self.test_dataset,
            batch_size=self.cfgs.model.batch_size,
            num_workers=self.cfgs.testset.n_workers,
            pin_memory=True
        )

        logging.info('Creating model: %s' % self.cfgs.model.name)
        self.model = model_factory(self.cfgs.model).to(device=self.device)

        logging.info('Loading checkpoint from %s' % self.cfgs.ckpt.path)
        checkpoint = torch.load(self.cfgs.ckpt.path, self.device)
        self.model.load_state_dict(checkpoint['state_dict'], strict=self.cfgs.ckpt.strict)

    @torch.no_grad()
    def run(self):
        logging.info('Generating outputs for KITTI submission...')
        self.model.eval()

        for inputs in tqdm(self.test_loader):
            inputs = copy_to_device(inputs, self.device)

            with torch.cuda.amp.autocast(enabled=False):
                outputs = self.model.forward(inputs)

            for batch_id in range(size_of_batch(inputs)):
                flow_2d_pred = outputs['flow_2d'][batch_id]
                flow_3d_pred = outputs['flow_3d'][batch_id]

                test_id = inputs['index'][batch_id].item()
                input_h = inputs['input_h'][batch_id].item()
                input_w = inputs['input_w'][batch_id].item()

                if 'training' in self.cfgs.testset.split:
                    out_dir = 'submission/training'
                else:
                    out_dir = 'submission/testing'

                f = inputs['intrinsics'][batch_id][0].item()
                cx = inputs['intrinsics'][batch_id][1].item()
                cy = inputs['intrinsics'][batch_id][2].item()

                if self.test_dataset.cfgs.disp_provider == 'kitti':
                    disp1 = load_disp_png(os.path.join(
                        self.test_dataset.root_dir, 'disp_occ_0', '%06d_10.png' % test_id
                    ))[0]
                else:
                    disp1 = load_disp_png(os.path.join(
                        self.test_dataset.root_dir,
                        'disp_%s' % self.test_dataset.cfgs.disp_provider,
                        '%06d_10.png' % test_id
                    ))[0]
                os.makedirs('%s/disp_0' % out_dir, exist_ok=True)
                save_disp_png('%s/disp_0/%06d_10.png' % (out_dir, test_id), disp1)

                flow_2d_pred = flow_2d_pred.permute(1, 2, 0).clamp(-500, 500).cpu().numpy()
                flow_2d_pred = flow_2d_pred[:input_h, :input_w]
                os.makedirs('%s/flow_initial' % out_dir, exist_ok=True)
                save_flow_png('%s/flow_initial/%06d_10.png' % (out_dir, test_id), flow_2d_pred)

                # densification
                pc1_dense = disp2pc(disp1, baseline=0.54, f=f, cx=cx, cy=cy)
                pc1_dense = torch.from_numpy(pc1_dense.reshape([-1, 3]).transpose()).to(self.device)
                pc1 = inputs['pcs'][batch_id, :3]
                flow_3d_dense = knn_interpolation(pc1[None, ...], flow_3d_pred[None, ...], pc1_dense[None, ...])[0]
                pc1_dense_warp = pc1_dense + flow_3d_dense
                disp_c = 0.54 * f / pc1_dense_warp[2].cpu().numpy().reshape(input_h, input_w)
                disp_c[disp1 < 0] = -1.0
                os.makedirs('%s/disp_1_initial' % out_dir, exist_ok=True)
                save_disp_png('%s/disp_1_initial/%06d_10.png' % (out_dir, test_id), disp_c)

                # generate a coarse occlusion mask for rigid background refinement
                grid = mesh_grid(1, input_h, input_w, device='cpu', channel_first=False)[0].numpy()
                grid_warp = grid + flow_2d_pred
                x_out = np.logical_or(grid_warp[..., 0] < 0, grid_warp[..., 0] > input_w)
                y_out = np.logical_or(grid_warp[..., 1] < 0, grid_warp[..., 1] > input_h)
                occ_mask1_2d = np.logical_or(x_out, y_out).astype(np.uint8) * 255
                os.makedirs('%s/occ' % out_dir, exist_ok=True)
                cv2.imwrite('%s/occ/%06d_10.png' % (out_dir, test_id), occ_mask1_2d)


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
        raise RuntimeError('Submission script does not support multi-GPU systems.')

    evaluator = Evaluator(device, cfgs)
    evaluator.run()


if __name__ == '__main__':
    main()
