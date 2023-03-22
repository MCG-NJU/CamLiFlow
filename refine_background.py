import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from refine_utils import mod_flow
from utils import load_disp_png, load_flow_png, save_flow_png, init_logging, save_disp_png, load_calib


def main(args):
    for i in tqdm(range(200)):
        np.random.seed(0)

        bg_mask = cv2.imread(os.path.join(args.dataset_dir, args.dataset_split, args.semantic_dir, '%06d_10.png' % i), 0) <= 22
        K0 = load_calib(os.path.join(args.dataset_dir, args.dataset_split, 'calib_cam_to_cam', '%06d.txt' % i))[0:3, 0:3]

        disp = load_disp_png('submission/%s/disp_0/%06d_10.png' % (args.dataset_split, i))[0]
        disp_c = load_disp_png('submission/%s/disp_1_initial/%06d_10.png' % (args.dataset_split, i))[0]
        flow = load_flow_png('submission/%s/flow_initial/%06d_10.png' % (args.dataset_split, i))[0]
        occ_mask = cv2.imread('submission/%s/occ/%06d_10.png' % (args.dataset_split, i), 0) == 0

        flow_refine, disp_c_refine = mod_flow(bg_mask, disp, disp_c.copy(), flow, K0, K1=K0, bl=0.54, occ_mask=occ_mask)
        flow_refine = np.clip(flow_refine, -500, 500)

        os.makedirs('submission/%s/flow' % args.dataset_split, exist_ok=True)
        save_flow_png('submission/%s/flow/%06d_10.png' % (args.dataset_split, i), flow_refine)

        os.makedirs('submission/%s/disp_1' % args.dataset_split, exist_ok=True)
        save_disp_png('submission/%s/disp_1/%06d_10.png' % (args.dataset_split, i), disp_c_refine)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', required=False, default='datasets/kitti_scene_flow')
    parser.add_argument('--dataset_split', required=False, default='testing')
    parser.add_argument('--semantic_dir', required=False, default='semantic_ddr')
    args = parser.parse_args()

    init_logging()
    main(args)
