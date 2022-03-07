import os
import time
import yaml
import utils
import shutil
import logging
import argparse
import torch
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig, OmegaConf
from factory import dataset_factory, model_factory, optimizer_factory
from utils import copy_to_device, size_of_batch, dist_reduce_sum


class Trainer:
    def __init__(self, device, cfgs):
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'

        self.cfgs = cfgs
        self.curr_epoch, self.curr_step = 1, 1
        self.device = device
        self.n_gpus = torch.cuda.device_count()
        self.is_main = device.index is None or device.index == 0
        utils.init_logging(os.path.join(self.cfgs.log.dir, 'train.log'))

        if device.index is None:
            logging.info('No CUDA device detected, using CPU for training')
        else:
            logging.info('Using GPU %d: %s' % (device.index, torch.cuda.get_device_name(device)))
            if self.n_gpus > 1:
                init_process_group('nccl', 'tcp://localhost:6789', world_size=self.n_gpus, rank=self.device.index)
                self.cfgs.model.batch_size = int(self.cfgs.model.batch_size / self.n_gpus)
                self.cfgs.trainset.n_workers = int(self.cfgs.trainset.n_workers / self.n_gpus)
                self.cfgs.valset.n_workers = int(self.cfgs.valset.n_workers / self.n_gpus)
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
        self.train_loader = torch.utils.data.DataLoader(
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
        self.val_loader = torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.cfgs.model.batch_size,
            shuffle=False,
            num_workers=self.cfgs.valset.n_workers,
            pin_memory=True,
            sampler=self.val_sampler,
        )

        logging.info('Creating model: CamLiFlow')
        self.model = model_factory(self.cfgs.model).to(device=self.device)

        logging.info('Trainable parameters: %d' %
                     sum([p.numel() for p in self.model.parameters() if p.requires_grad]))

        if self.n_gpus > 1:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.ddp = DistributedDataParallel(self.model, [self.device.index])
        else:
            self.ddp = self.model

        self.best_metrics = None
        if self.cfgs.ckpt.path is not None:
            self.load_ckpt(self.cfgs.ckpt.path, resume=self.cfgs.ckpt.resume)

        logging.info('Creating optimizer: %s' % self.cfgs.training.optimizer)
        self.optimizer, self.lr_scheduler = optimizer_factory(self.cfgs.training, self.model.named_parameters(), self.curr_epoch - 1)
        self.amp_scaler = torch.cuda.amp.GradScaler()

    def run(self):
        while self.curr_epoch <= self.cfgs.training.max_epochs:
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(self.curr_epoch)
            if self.val_sampler is not None:
                self.val_sampler.set_epoch(self.curr_epoch)

            self.save_scalar_summary({'learning_rate': self.optimizer.param_groups[0]['lr']}, self.curr_epoch)

            self.train_one_epoch()
            self.validate()
            self.lr_scheduler.step()

            if self.curr_epoch % self.cfgs.log.save_ckpt_every_n_epochs == 0:
                self.save_ckpt()

            self.curr_epoch += 1

    def train_one_epoch(self):
        self.ddp.train()

        start_time = time.time()
        for i, inputs in enumerate(self.train_loader):
            inputs = copy_to_device(inputs, self.device)

            with torch.cuda.amp.autocast(enabled=self.cfgs.amp):
                self.ddp.forward(inputs)
                loss = self.model.get_loss()

            self.optimizer.zero_grad()
            self.amp_scaler.scale(loss).backward()
            self.amp_scaler.step(self.optimizer)
            self.amp_scaler.update()

            timing = time.time() - start_time
            start_time = time.time()

            logging.info('E: [%d/%d] ' % (self.curr_epoch, self.cfgs.training.max_epochs) +
                         'S: [%d/%d] ' % (i + 1, len(self.train_loader)) +
                         '| %s, timing: %.2fs' % (self.model.get_log_string(), timing))

            self.save_scalar_summary(self.model.get_scalar_summary(), self.curr_step, prefix='train/')
            self.curr_step += 1

    @torch.no_grad()
    def validate(self):
        self.ddp.eval()
        epoch_summary = None

        start_time = time.time()
        for i, inputs in enumerate(self.val_loader):
            inputs = copy_to_device(inputs, self.device)

            with torch.cuda.amp.autocast(enabled=False):
                self.ddp.forward(inputs)

            timing = time.time() - start_time
            start_time = time.time()

            logging.info('S: [%d/%d] ' % (i + 1, len(self.val_loader)) +
                         '| %s, timing: %.2fs' % (self.model.get_log_string(), timing))

            batch_summary = self.model.get_scalar_summary()

            if epoch_summary is None:
                epoch_summary = {k: batch_summary[k] * size_of_batch(inputs) for k in batch_summary}
            else:
                epoch_summary = {k: batch_summary[k] * size_of_batch(inputs) + epoch_summary[k] for k in batch_summary}

        epoch_summary = {k: dist_reduce_sum(epoch_summary[k], self.n_gpus) / len(self.val_dataset) for k in epoch_summary}
        logging.info('Statistics on validation set: %s' % str(epoch_summary))
        self.save_scalar_summary(epoch_summary, self.curr_epoch, prefix='val/')

        if self.model.is_better(epoch_summary, self.best_metrics):
            self.best_metrics = epoch_summary
            self.save_ckpt('best.pt')

    def save_scalar_summary(self, scalar_summary: dict, curr_step: int, prefix=''):
        if self.is_main:
            for name in scalar_summary.keys():
                self.summary_writer.add_scalar(prefix + name, scalar_summary[name], curr_step)

    def save_ckpt(self, filename=None):
        if self.is_main and self.cfgs.log.save_ckpt:
            ckpt_dir = os.path.join(self.cfgs.log.dir, 'ckpts')
            os.makedirs(ckpt_dir, exist_ok=True)
            filepath = os.path.join(ckpt_dir, filename or 'epoch-%03d.pt' % self.curr_epoch)
            logging.info('Saving checkpoint to %s' % filepath)
            torch.save({
                'last_epoch': self.curr_epoch,
                'last_step': self.curr_step,
                'state_dict': self.model.state_dict(),
                'best_metrics': self.best_metrics
            }, filepath)

    def load_ckpt(self, filepath, resume=True):
        logging.info('Loading checkpoint from %s' % filepath)
        checkpoint = torch.load(filepath, self.device)
        if resume:
            self.curr_epoch = checkpoint['last_epoch'] + 1
            self.curr_step = checkpoint['last_step'] + 1
            self.best_metrics = checkpoint['best_metrics']
            logging.info('Current best metrics: %s' % str(self.best_metrics))
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)


def create_trainer(device_id, cfgs):
    device = torch.device('cpu' if device_id is None else 'cuda:%d' % device_id)
    trainer = Trainer(device, cfgs)
    trainer.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True,
                        help='Path to the configuration (YAML format)')
    parser.add_argument('--weights', required=False, default=None,
                        help='Path to pretrained weights')
    args = parser.parse_args()

    # load config
    with open(args.config, encoding='utf-8') as f:
        cfgs = DictConfig(yaml.load(f, Loader=yaml.FullLoader))
        cfgs.ckpt.path = args.weights

    # set num_workers of data loader
    n_devices = max(torch.cuda.device_count(), 1)
    cfgs.trainset.n_workers = min(os.cpu_count(), cfgs.trainset.n_workers * n_devices)
    cfgs.valset.n_workers = min(os.cpu_count(), cfgs.valset.n_workers * n_devices)

    # create log dir
    if os.path.exists(cfgs.log.dir):
        if input('Run "%s" already exists, overwrite it? [Y/n] ' % cfgs.log.run_name) == 'n':
            exit(0)
        shutil.rmtree(cfgs.log.dir, ignore_errors=True)
    os.makedirs(cfgs.log.dir, exist_ok=False)

    # create trainers
    if torch.cuda.device_count() == 0:  # CPU
        create_trainer(None, cfgs)
    elif torch.cuda.device_count() == 1:  # Single GPU
        create_trainer(0, cfgs)
    elif torch.cuda.device_count() > 1:  # Multiple GPUs
        mp.spawn(create_trainer, (cfgs,), torch.cuda.device_count())
