import cv2
import yaml
import utils
import open3d
import logging
import argparse
import numpy as np
import torch
import torch.optim
import torch.utils.data
from matplotlib.colors import hsv_to_rgb
from omegaconf import DictConfig
from factory import model_factory
from utils import copy_to_device, load_fpm, disp2pc


def viz_optical_flow(flow, max_flow=512):
    n = 8
    u, v = flow[:, :, 0], flow[:, :, 1]
    mag = np.sqrt(np.square(u) + np.square(v))
    angle = np.arctan2(v, u)

    image_h = np.mod(angle / (2 * np.pi) + 1, 1)
    image_s = np.clip(mag * n / max_flow, a_min=0, a_max=1)
    image_v = np.ones_like(image_s)

    image_hsv = np.stack([image_h, image_s, image_v], axis=2)
    image_rgb = hsv_to_rgb(image_hsv)
    image_rgb = np.uint8(image_rgb * 255)

    return image_rgb


class Demo:
    def __init__(self, device: torch.device, cfgs: DictConfig, ckpt_path: str):
        self.cfgs = cfgs
        self.device = device

        logging.info('Creating model: %s' % cfgs.model.name)
        self.model = model_factory(self.cfgs.model).to(device=self.device)

        logging.info('Loading checkpoint from %s' % ckpt_path)
        checkpoint = torch.load(ckpt_path, self.device)
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)

    def prepare_images_and_depths(self):
        # load images
        image1 = cv2.imread(args.image1)[..., ::-1]
        image2 = cv2.imread(args.image2)[..., ::-1]

        # load disparity maps
        disp1 = -load_fpm(args.disp1)
        disp2 = -load_fpm(args.disp2)

        # lift disparity maps into point clouds
        pc1 = disp2pc(disp1, args.baseline, args.f, args.cx, args.cy)
        pc2 = disp2pc(disp2, args.baseline, args.f, args.cx, args.cy)

        # apply depth mask
        mask1 = (pc1[..., -1] < args.max_depth)
        mask2 = (pc2[..., -1] < args.max_depth)
        pc1, pc2 = pc1[mask1], pc2[mask2]

        # NaN check
        mask1 = np.logical_not(np.isnan(np.sum(pc1, axis=-1)))
        mask2 = np.logical_not(np.isnan(np.sum(pc2, axis=-1)))
        pc1, pc2 = pc1[mask1], pc2[mask2]

        # random sampling
        indices1 = np.random.choice(pc1.shape[0], size=min(args.n_points, pc1.shape[0]), replace=False)
        indices2 = np.random.choice(pc2.shape[0], size=min(args.n_points, pc2.shape[0]), replace=False)
        pc1, pc2 = pc1[indices1], pc2[indices2]

        return image1, image2, pc1, pc2

    @torch.no_grad()
    def run(self):
        logging.info('Running demo...')
        self.model.eval()

        image1, image2, pc1, pc2 = self.prepare_images_and_depths()

        # numpy -> torch
        images = np.concatenate([image1, image2], axis=-1).transpose([2, 0, 1])
        images = torch.from_numpy(images).float().unsqueeze(0)
        pcs = np.concatenate([pc1, pc2], axis=1).transpose()
        pcs = torch.from_numpy(pcs).float().unsqueeze(0)
        intrinsics = torch.as_tensor([args.f, args.cx, args.cy]).unsqueeze(0)
        
        # inference
        inputs = {'images': images, 'pcs': pcs, 'intrinsics': intrinsics}
        inputs = copy_to_device(inputs, self.device)
        outputs = self.model(inputs)

        # NCHW -> NHWC
        flow_2d = outputs['flow_2d'][0].cpu().numpy().transpose(1, 2, 0)
        flow_3d = outputs['flow_3d'][0].cpu().numpy().transpose()

        self.display(image1, image2, pc1, pc2, flow_2d, flow_3d)

    def display(self, image1, image2, pc1, pc2, flow_2d, flow_3d):
        # visualize optical flow
        flow_2d_img = viz_optical_flow(flow_2d)
        images = np.concatenate([image1, image2, flow_2d_img], axis=0)
        images = cv2.resize(images, dsize=None, fx=0.5, fy=0.5)
        cv2.imshow('', images[..., ::-1])
        cv2.waitKey(0)

        # visualize scene flow
        point_cloud1 = open3d.geometry.PointCloud()
        point_cloud2 = open3d.geometry.PointCloud()
        point_cloud3 = open3d.geometry.PointCloud()  # pc1 + flow3d
        point_cloud1.points = open3d.utility.Vector3dVector(pc1)
        point_cloud2.points = open3d.utility.Vector3dVector(pc2)
        point_cloud3.points = open3d.utility.Vector3dVector(pc1 + flow_3d)
        point_cloud1.colors = open3d.utility.Vector3dVector(np.zeros_like(pc1) + [1, 0, 0])
        point_cloud2.colors = open3d.utility.Vector3dVector(np.zeros_like(pc2) + [0, 1, 0])
        point_cloud3.colors = open3d.utility.Vector3dVector(np.zeros_like(pc1) + [0, 0, 1])
        open3d.visualization.draw_geometries([point_cloud1, point_cloud2, point_cloud3])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model weights
    parser.add_argument('--model', required=True, help='Model name, e.g. camliraft')
    parser.add_argument('--weights', required=True, help='Path to pretrained weights')

    # RGB input
    parser.add_argument('--image1', required=False, default='asserts/demo_image1.png')
    parser.add_argument('--image2', required=False, default='asserts/demo_image2.png')
    
    # disparity input
    parser.add_argument('--disp1', required=False, default='asserts/demo_disp1.pfm')
    parser.add_argument('--disp2', required=False, default='asserts/demo_disp2.pfm')
    
    # disparity -> point clouds
    parser.add_argument('--n_points', required=False, default=8192)
    parser.add_argument('--max_depth', required=False, default=35.0)
    
    # camera intrinsics
    parser.add_argument('--baseline', required=False, default=1.0)
    parser.add_argument('--f', required=False, default=1050.0)
    parser.add_argument('--cx', required=False, default=479.5)
    parser.add_argument('--cy', required=False, default=269.5)
    
    args = parser.parse_args()

    assert args.model in ['camlipwc', 'camliraft']

    with open('conf/model/%s.yaml' % args.model, encoding='utf-8') as f:
        cfgs = DictConfig(yaml.load(f, Loader=yaml.FullLoader))

    utils.init_logging()

    if torch.cuda.device_count() == 0:
        device = torch.device('cpu')
        logging.info('No CUDA device detected, using CPU for evaluation')
    else:
        device = torch.device('cuda:0')
        logging.info('Using GPU: %s' % torch.cuda.get_device_name(device))

    demo = Demo(device, cfgs, args.weights)
    demo.run()
