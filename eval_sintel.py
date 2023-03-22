import os
import utils
import hydra
import shutil
import logging
import torch
import torch.optim
import torch.backends.cudnn as cudnn
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig
from factory import model_factory, Sintel
from sintel import TRAIN_SCENES, VALIDATE_SCENES
from utils import copy_to_device, size_of_batch, save_flow_png


class Evaluator:
    def __init__(self, device: torch.device, cfgs: DictConfig):
        self.cfgs = cfgs
        self.device = device

        logging.info('Loading test set from %s' % self.cfgs.testset.root_dir)
        self.test_dataset = Sintel(self.cfgs.testset)
        self.test_loader = utils.FastDataLoader(
            dataset=self.test_dataset,
            batch_size=8,
            num_workers=self.cfgs.testset.n_workers
        )

        logging.info('Creating model: %s' % self.cfgs.model.name)
        self.model = model_factory(self.cfgs.model).to(device=self.device)

        logging.info('Loading checkpoint from %s' % self.cfgs.ckpt.path)
        checkpoint = torch.load(self.cfgs.ckpt.path, self.device)
        self.model.load_state_dict(checkpoint['state_dict'], strict=self.cfgs.ckpt.strict)

    @torch.no_grad()
    def run(self):
        logging.info('Running evaluation...')
        self.model.eval()

        metrics_2d, metrics_3d = {}, {}
        for scene_name in TRAIN_SCENES + VALIDATE_SCENES:
            metrics_2d[scene_name] = {'counts': 0, 'EPE2d': 0.0, '1px': 0.0, 'Fl': 0.0}
            metrics_3d[scene_name] = {'counts': 0, 'EPE3d': 0.0, '5cm': 0.0, '10cm': 0.0}

        for inputs in tqdm(self.test_loader):
            inputs = copy_to_device(inputs, self.device)

            with torch.cuda.amp.autocast(enabled=False):
                outputs = self.model.forward(inputs)

            for batch_id in range(size_of_batch(inputs)):
                flow_2d_pred = outputs['flow_2d'][batch_id]
                flow_2d_target = inputs['flow_2d'][batch_id]

                if flow_2d_target.shape[0] > 2:
                    flow_2d_mask = flow_2d_target[2] > 0
                    flow_2d_target = flow_2d_target[:2]
                else:
                    flow_2d_mask = torch.ones(flow_2d_target.shape[1:], dtype=torch.int64, device=self.device)

                epe2d_map = torch.sqrt(torch.sum((flow_2d_pred - flow_2d_target) ** 2, dim=0))
                flow_2d_mask = torch.logical_and(flow_2d_mask, torch.logical_not(torch.isnan(epe2d_map)))
                flow_2d_target_mag = torch.linalg.norm(flow_2d_target, dim=0)
                fl_err_map = torch.logical_and(epe2d_map > 3.0, epe2d_map / flow_2d_target_mag > 0.05)

                scene_name = inputs['scene_name'][batch_id]
                metrics_2d[scene_name]['counts'] += epe2d_map[flow_2d_mask].shape[0]
                metrics_2d[scene_name]['EPE2d'] += epe2d_map[flow_2d_mask].sum().item()
                metrics_2d[scene_name]['1px'] += torch.count_nonzero(epe2d_map[flow_2d_mask] < 1.0).item()
                metrics_2d[scene_name]['Fl'] += fl_err_map[flow_2d_mask].float().sum().item()

                if self.cfgs.save_results:
                    sample_name = inputs['sample_name'][batch_id]
                    os.makedirs('prediction/sintel/%s' % scene_name, exist_ok=True)
                    flow_2d_pred = flow_2d_pred.clamp(-500, 500).permute(1, 2, 0).cpu().numpy()
                    save_flow_png('prediction/sintel/%s/%s.png' % (scene_name, sample_name), flow_2d_pred)

        logging.info('#### 2D EPE ####')
        total_epe2d, total_count = 0, 0
        for scene_name, metrics in metrics_2d.items():
            if metrics['counts'] == 0:
                continue
            total_epe2d += metrics['EPE2d']
            total_count += metrics['counts']
            logging.info('%s:\t%.3f' % (scene_name, metrics['EPE2d'] / metrics['counts']))
        logging.info('Total:\t%.3f' % (total_epe2d / total_count))


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
