import os
import glob
import time
import utils
import hydra
import random
import shutil
import logging
import torch
import torch.optim
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler
from factory import dataset_factory, model_factory, optimizer_factory
from utils import copy_to_device, override_cfgs, FastDataLoader, get_max_memory, get_grad_norm
from models.utils import timer


class Trainer:
    def __init__(self, device: torch.device, cfgs: DictConfig):
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['NCCL_IB_DISABLE'] = '1'
        os.environ['NCCL_P2P_DISABLE'] = '1'

        # MMCV, please shut up
        from mmcv.utils.logging import get_logger
        get_logger('root').setLevel(logging.ERROR)
        get_logger('mmcv').setLevel(logging.ERROR)

        self.cfgs = cfgs
        self.curr_epoch = 1
        self.device = device
        self.n_gpus = torch.cuda.device_count()
        self.is_main = device.index is None or device.index == 0
        utils.init_logging(os.path.join(self.cfgs.log.dir, 'train.log'), self.cfgs.debug)

        if device.index is None:
            logging.info('No CUDA device detected, using CPU for training')
        else:
            logging.info('Using GPU %d: %s' % (device.index, torch.cuda.get_device_name(device)))
            if self.n_gpus > 1:
                init_process_group('nccl', 'tcp://localhost:%d' % self.cfgs.port,
                                   world_size=self.n_gpus, rank=self.device.index)
                self.cfgs.model.batch_size = int(self.cfgs.model.batch_size / self.n_gpus)
                self.cfgs.trainset.n_workers = int(self.cfgs.trainset.n_workers / self.n_gpus)
                self.cfgs.valset.n_workers = int(self.cfgs.valset.n_workers / self.n_gpus)
            if not cfgs.debug:
                cudnn.benchmark = True
            torch.cuda.set_device(self.device)

        if self.is_main:
            logging.info('Logs will be saved to %s' % self.cfgs.log.dir)
            self.summary_writer = SummaryWriter(self.cfgs.log.dir)
            logging.info('Configurations:\n' + OmegaConf.to_yaml(self.cfgs))
        else:
            logging.root.disabled = True

        logging.info('Loading training set from %s' % self.cfgs.trainset.root_dir)
        self.train_dataset = dataset_factory(self.cfgs.trainset)
        self.train_sampler = DistributedSampler(self.train_dataset) if self.n_gpus > 1 else None
        self.train_loader = FastDataLoader(
            dataset=self.train_dataset,
            batch_size=self.cfgs.model.batch_size,
            shuffle=(self.train_sampler is None),
            num_workers=self.cfgs.trainset.n_workers,
            pin_memory=True,
            sampler=self.train_sampler,
            drop_last=self.cfgs.trainset.drop_last,
        )

        logging.info('Loading validation set from %s' % self.cfgs.valset.root_dir)
        self.val_dataset = dataset_factory(self.cfgs.valset)
        self.val_sampler = DistributedSampler(self.val_dataset) if self.n_gpus > 1 else None
        self.val_loader = FastDataLoader(
            dataset=self.val_dataset,
            batch_size=self.cfgs.model.batch_size,
            shuffle=False,
            num_workers=self.cfgs.valset.n_workers,
            pin_memory=True,
            sampler=self.val_sampler,
        )

        logging.info('Creating model: %s' % self.cfgs.model.name)
        self.model = model_factory(self.cfgs.model)
        self.model.to(device=self.device)

        n_params = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        logging.info('Trainable parameters: %d (%.1fM)' % (n_params, n_params / 1e6))

        if self.n_gpus > 1:
            if self.cfgs.sync_bn:
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.ddp = DistributedDataParallel(self.model, [self.device.index])
        else:
            self.ddp = self.model

        self.best_metrics = None
        if self.cfgs.ckpt.path is not None:
            self.load_ckpt(self.cfgs.ckpt.path, resume=self.cfgs.ckpt.resume)

        logging.info('Creating optimizer: %s' % self.cfgs.training.opt)
        self.optimizer, self.scheduler = optimizer_factory(self.cfgs.training, self.model)
        self.scheduler.step(self.curr_epoch - 1)

        self.amp_scaler = GradScaler(enabled=self.cfgs.amp)

    def run(self):
        while self.curr_epoch <= self.cfgs.training.epochs:
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(self.curr_epoch)
            if self.val_sampler is not None:
                self.val_sampler.set_epoch(self.curr_epoch)

            self.train_one_epoch()

            if self.curr_epoch % self.cfgs.val_interval == 0:
                self.validate()

            self.save_ckpt()
            self.scheduler.step(self.curr_epoch)

            self.curr_epoch += 1

    def train_one_epoch(self):
        logging.info('Start training...')

        self.ddp.train()
        self.model.clear_metrics()
        self.optimizer.zero_grad()

        lr = self.optimizer.param_groups[0]['lr']
        self.save_scalar_summary({'learning_rate': lr}, prefix='train')

        start_time = time.time()
        for i, inputs in enumerate(self.train_loader):
            inputs = copy_to_device(inputs, self.device)

            # forward
            with torch.cuda.amp.autocast(enabled=self.cfgs.amp):
                self.ddp.forward(inputs)
                loss = self.model.get_loss()

            # backward
            self.amp_scaler.scale(loss).backward()

            # get grad norm statistics
            grad_norm_2d = get_grad_norm(self.model, prefix='core.branch_2d')
            grad_norm_3d = get_grad_norm(self.model, prefix='core.branch_3d')
            self.model.update_metrics('grad_norm_2d', grad_norm_2d)
            self.model.update_metrics('grad_norm_3d', grad_norm_3d)

            # grad clip
            if 'grad_max_norm' in self.cfgs.training.keys():
                self.amp_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    parameters=self.model.parameters(),
                    max_norm=self.cfgs.training.grad_max_norm
                )

            # update
            self.amp_scaler.step(self.optimizer)
            self.amp_scaler.update()
            self.optimizer.zero_grad()

            timing = time.time() - start_time
            start_time = time.time()
            mem = get_max_memory(self.device, self.n_gpus)

            logging.info('Epoch: [%d/%d]' % (self.curr_epoch, self.cfgs.training.epochs) +
                            '[%d/%d] ' % (i + 1, len(self.train_loader)) +
                            'loss: %.1f, time: %.2fs, mem: %dM' % (loss, timing, mem))

            for k, v in timer.get_timing_stat().items():
                logging.info('Function "%s" takes %.1fms' % (k, v))

            timer.clear_timing_stat()

        metrics = self.model.get_metrics()
        self.save_scalar_summary(metrics, prefix='train')

    @torch.no_grad()
    def validate(self):
        logging.info('Start validating...')

        self.ddp.eval()
        self.model.clear_metrics()

        for inputs in tqdm(self.val_loader):
            inputs = copy_to_device(inputs, self.device)
            self.ddp.forward(inputs)

        metrics = self.model.get_metrics()
        self.save_scalar_summary(metrics, prefix='val')

        for k, v in metrics.items():
            logging.info('%s: %.4f' % (k, v))

        if self.model.is_better(metrics, self.best_metrics):
            self.best_metrics = metrics
            self.save_ckpt('best.pt')

    def save_scalar_summary(self, scalar_summary: dict, prefix):
        if self.is_main and self.cfgs.log.save_scalar_summary:
            for name in scalar_summary.keys():
                self.summary_writer.add_scalar(
                    prefix + '/' + name,
                    scalar_summary[name],
                    self.curr_epoch
                )

    def save_image_summary(self, image_summary: dict, prefix):
        if self.is_main and self.cfgs.log.save_image_summary:
            for name in image_summary.keys():
                self.summary_writer.add_image(
                    prefix + '/' + name,
                    image_summary[name],
                    self.curr_epoch
                )

    def save_ckpt(self, filename=None):
        if self.is_main and self.cfgs.log.save_ckpt:
            ckpt_dir = os.path.join(self.cfgs.log.dir, 'ckpts')
            os.makedirs(ckpt_dir, exist_ok=True)
            filepath = os.path.join(ckpt_dir, filename or 'epoch-%03d.pt' % self.curr_epoch)
            logging.info('Saving checkpoint to %s' % filepath)
            torch.save({
                'last_epoch': self.curr_epoch,
                'state_dict': self.model.state_dict(),
                'best_metrics': self.best_metrics
            }, filepath)

    def load_ckpt(self, filepath, resume=True):
        logging.info('Loading checkpoint from %s' % filepath)
        checkpoint = torch.load(filepath, self.device)
        if resume:
            self.curr_epoch = checkpoint['last_epoch'] + 1
            self.best_metrics = checkpoint['best_metrics']
            logging.info('Current best metrics: %s' % str(self.best_metrics))
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)


def create_trainer(device_id, cfgs):
    device = torch.device('cpu' if device_id is None else 'cuda:%d' % device_id)
    trainer = Trainer(device, cfgs)
    trainer.run()


@hydra.main(config_path='conf', config_name='trainer')
def main(cfgs: DictConfig):
    # set num_workers of data loader
    if not cfgs.debug:
        n_devices = max(torch.cuda.device_count(), 1)
        cfgs.trainset.n_workers = min(os.cpu_count(), cfgs.trainset.n_workers * n_devices)
        cfgs.valset.n_workers = min(os.cpu_count(), cfgs.valset.n_workers * n_devices)
    else:
        cfgs.trainset.n_workers = 0
        cfgs.valset.n_workers = 0

    # resolve configurations
    if cfgs.ckpt.path is not None and cfgs.ckpt.resume:
        assert os.path.isfile(os.path.join(hydra.utils.get_original_cwd(), cfgs.ckpt.path))
        assert os.path.dirname(os.path.join(hydra.utils.get_original_cwd(), cfgs.ckpt.path))[-5:] == 'ckpts'
        shutil.rmtree(os.getcwd(), ignore_errors=True)
        cfgs.log.dir = os.path.dirname(os.path.dirname(cfgs.ckpt.path))

    if cfgs.log.dir is None:
        shutil.rmtree(os.getcwd(), ignore_errors=True)

        run_name = ''
        if cfgs.log.ask_name:
            run_name = input('Name your run (leave blank for default): ')
        if run_name == '':
            run_name = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")

        cfgs.log.dir = os.path.join('outputs', cfgs.model.name, run_name)
        if os.path.exists(os.path.join(hydra.utils.get_original_cwd(), cfgs.log.dir)):
            if input('Run "%s" already exists, overwrite it? [Y/n] ' % run_name) == 'n':
                exit(0)
            shutil.rmtree(os.path.join(hydra.utils.get_original_cwd(), cfgs.log.dir), ignore_errors=True)

        os.makedirs(os.path.join(hydra.utils.get_original_cwd(), cfgs.log.dir), exist_ok=False)

    if cfgs.port == 'random':
        cfgs.port = random.randint(10000, 20000)

    if 'override' in cfgs:
        cfgs = override_cfgs(cfgs, cfgs.override)

    if cfgs.training.accum_iter > 1:
        cfgs.model.batch_size //= int(cfgs.training.accum_iter)

    # create trainers
    os.chdir(hydra.utils.get_original_cwd())
    if torch.cuda.device_count() == 0:  # CPU
        create_trainer(None, cfgs)
    elif torch.cuda.device_count() == 1:  # Single GPU
        create_trainer(0, cfgs)
    elif torch.cuda.device_count() > 1:  # Multiple GPUs
        mp.spawn(create_trainer, (cfgs,), torch.cuda.device_count())


if __name__ == '__main__':
    main()
