import os
import cv2
import yaml
import utils
import logging
import argparse
import torch
import torch.optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig
from factory import model_factory
from kitti import KITTITest
from refine_utils import mod_flow
from models.utils import mesh_grid, knn_interpolation
from utils import copy_to_device, save_flow_png, save_disp_png, load_disp_png, load_calib, disp2pc


def rigid_refinement(test_id, disp, disp_c, flow, noc_mask):
    np.random.seed(0)

    bg_mask = cv2.imread(os.path.join(
        cfgs.testset.root_dir, 'testing', 'semantic_ddr', '%06d_10.png' % test_id), 0) <= 22
    K0 = load_calib(os.path.join(
        cfgs.testset.root_dir, 'testing', 'calib_cam_to_cam', '%06d.txt' % test_id))[0:3, 0:3]

    flow_refine, disp_c_refine = mod_flow(
        bg_mask.copy(), disp.copy(), disp_c.copy(), flow.copy(), K0,
        K1=K0, bl=0.54, noc_mask=noc_mask
    )
    flow_refine = np.clip(flow_refine, -500, 500)

    return flow_refine, disp_c_refine


class Evaluator:
    def __init__(self, device: torch.device, cfgs: DictConfig):
        self.cfgs = cfgs
        self.device = device

        logging.info('Loading test set from %s' % self.cfgs.testset.root_dir)
        logging.info('Dataset split: %s' % self.cfgs.testset.split)
        self.test_dataset = KITTITest(self.cfgs.testset)

        self.test_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.cfgs.model.batch_size,
            num_workers=self.cfgs.testset.n_workers,
            pin_memory=True
        )

        logging.info('Creating model: CamLiFlow')
        self.model = model_factory(self.cfgs.model).to(device=self.device)

        logging.info('Loading checkpoint from %s' % self.cfgs.ckpt.path)
        checkpoint = torch.load(self.cfgs.ckpt.path, self.device)
        self.model.load_state_dict(checkpoint['state_dict'], strict=self.cfgs.ckpt.strict)

    @torch.no_grad()
    def run(self):
        logging.info('Generating results for KITTI submission...')
        self.model.eval()

        out_dir = self.cfgs.out_dir
        os.makedirs('%s/disp_0' % out_dir, exist_ok=True)
        os.makedirs('%s/disp_1' % out_dir, exist_ok=True)
        os.makedirs('%s/flow' % out_dir, exist_ok=True)

        for inputs in tqdm(self.test_loader):
            inputs = copy_to_device(inputs, self.device)

            with torch.cuda.amp.autocast(enabled=False):
                outputs = self.model.forward(inputs)

            for batch_id in range(self.cfgs.model.batch_size):
                flow_2d_pred = outputs['flow_2d'][batch_id]
                flow_3d_pred = outputs['flow_3d'][batch_id]

                test_id = inputs['index'][batch_id].item()
                input_h = inputs['input_h'][batch_id].item()
                input_w = inputs['input_w'][batch_id].item()

                f = inputs['intrinsics'][batch_id][0].item()
                cx = inputs['intrinsics'][batch_id][1].item()
                cy = inputs['intrinsics'][batch_id][2].item()

                # disp_0: from GA-Net
                disp_0 = load_disp_png(os.path.join(
                    self.test_dataset.root_dir, 'disp_ganet', '%06d_10.png' % test_id)
                )[0]

                # predicted flow
                flow_2d_pred = flow_2d_pred.permute(1, 2, 0).clamp(-500, 500).cpu().numpy()
                flow_2d_pred = flow_2d_pred[:input_h, :input_w]

                # densification
                pc1_dense = disp2pc(disp_0, baseline=0.54, f=f, cx=cx, cy=cy)
                pc1_dense = torch.from_numpy(pc1_dense.reshape([-1, 3]).transpose()).to(self.device)
                pc1 = inputs['pcs'][batch_id, :3]
                flow_3d_dense = knn_interpolation(
                    input_xyz=pc1[None, ...],
                    input_features=flow_3d_pred[None, ...],
                    query_xyz=pc1_dense[None, ...],
                )[0]

                # compute disp_change (dense)
                pc1_dense_warp = pc1_dense + flow_3d_dense
                disp_c = 0.54 * f / pc1_dense_warp[2].cpu().numpy().reshape(input_h, input_w)
                disp_c[disp_0 < 0] = -1.0

                if self.cfgs.refine:
                    # generate a coarse occlusion mask for rigid background refinement
                    grid = mesh_grid(1, input_h, input_w, device='cpu', channel_first=False)[0].numpy()
                    grid_warp = grid + flow_2d_pred
                    x_out = np.logical_or(grid_warp[..., 0] < 0, grid_warp[..., 0] > input_w)
                    y_out = np.logical_or(grid_warp[..., 1] < 0, grid_warp[..., 1] > input_h)
                    occ_mask1_2d = np.logical_or(x_out, y_out).astype(np.uint8) * 255

                    flow_2d_pred, disp_c = rigid_refinement(
                        test_id, disp_0, disp_c, flow_2d_pred, occ_mask1_2d == 0
                    )

                save_disp_png('%s/disp_0/%06d_10.png' % (out_dir, test_id), disp_0)
                save_disp_png('%s/disp_1/%06d_10.png' % (out_dir, test_id), disp_c)
                save_flow_png('%s/flow/%06d_10.png' % (out_dir, test_id), flow_2d_pred)
        
        logging.info('Results have been saved to %s' % out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True,
                        help='Path to weights')
    parser.add_argument('--out_dir', required=False, default='submission',
                        help='Directory to store results')
    parser.add_argument('--refine', action='store_true',
                        help='Rigid refinement for background areas')
    args = parser.parse_args()

    # load config
    with open('conf/test/kitti.yaml', encoding='utf-8') as f:
        cfgs = DictConfig(yaml.load(f, Loader=yaml.FullLoader))
        cfgs.ckpt.path = args.weights
        cfgs.out_dir = args.out_dir
        cfgs.refine = args.refine

    utils.init_logging()

    if torch.cuda.device_count() == 0:
        device = torch.device('cpu')
        logging.info('No CUDA device detected, using CPU for evaluation')
    else:
        device = torch.device('cuda:0')
        logging.info('Using GPU: %s' % torch.cuda.get_device_name(device))
        cudnn.benchmark = True

    evaluator = Evaluator(device, cfgs)
    evaluator.run()
