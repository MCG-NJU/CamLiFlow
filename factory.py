import torch
from omegaconf import DictConfig
from models import PWCFusionProSupervised
from driving import Driving
from flyingthings3d import FlyingThings3D
from kitti import KITTI


def dataset_factory(cfgs: DictConfig):
    if cfgs.name == 'driving':
        return Driving(cfgs)
    elif cfgs.name == 'flyingthings3d':
        return FlyingThings3D(cfgs)
    elif cfgs.name == 'kitti':
        return KITTI(cfgs)
    else:
        raise NotImplementedError('Unknown dataset: %s' % cfgs.name)


def model_factory(cfgs: DictConfig):
    return PWCFusionProSupervised(cfgs)


def optimizer_factory(cfgs, named_params, last_epoch):
    param_groups = [
        {'params': [p for name, p in named_params if 'weight' in name],
         'weight_decay': cfgs.weight_decay},
        {'params': [p for name, p in named_params if 'bias' in name],
         'weight_decay': cfgs.bias_decay}
    ]

    if cfgs.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            params=param_groups,
            lr=cfgs.lr.init_value,
            eps=1e-7
        )
    elif cfgs.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            params=param_groups,
            lr=cfgs.lr.init_value,
            momentum=cfgs.lr.momentum
        )
    else:
        raise NotImplementedError('Unknown optimizer: %s' % cfgs.optimizer)

    if isinstance(cfgs.lr.decay_milestones, int):
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=cfgs.lr.decay_milestones,
            gamma=cfgs.lr.decay_rate
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=cfgs.lr.decay_milestones,
            gamma=cfgs.lr.decay_rate
        )

    for _ in range(last_epoch):
        optimizer.step()
        lr_scheduler.step()

    return optimizer, lr_scheduler
