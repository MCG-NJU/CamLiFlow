from omegaconf import DictConfig
from models import BaseModel, CamLiPWC, CamLiRAFT, PWC, CamLiPWC_L, RAFT, CamLiRAFT_L
from flyingthings3d import FlyingThings3D
from kitti import KITTI
from sintel import Sintel
from torch.optim import Adam, AdamW
from timm.scheduler import create_scheduler


def dataset_factory(cfgs: DictConfig):
    if cfgs.name == 'flyingthings3d':
        return FlyingThings3D(cfgs)
    elif cfgs.name == 'kitti':
        return KITTI(cfgs)
    elif cfgs.name == 'sintel':
        return Sintel(cfgs)
    else:
        raise NotImplementedError('Unknown dataset: %s' % cfgs.name)


def model_factory(cfgs: DictConfig) -> BaseModel:
    if cfgs.name == 'camlipwc':
        return CamLiPWC(cfgs)
    elif cfgs.name == 'camliraft':
        return CamLiRAFT(cfgs)
    elif cfgs.name == 'pwc':
        return PWC(cfgs)
    elif cfgs.name == 'camlipwc_l':
        return CamLiPWC_L(cfgs)
    elif cfgs.name == 'raft':
        return RAFT(cfgs)
    elif cfgs.name == 'camliraft_l':
        return CamLiRAFT_L(cfgs)
    else:
        raise NotImplementedError('Unknown model: %s' % cfgs.name)


def optimizer_factory(cfgs, model):
    params_2d_decay = []
    params_3d_decay = []

    params_2d_no_decay = []
    params_3d_no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            if name.startswith('core.branch_3d'):
                params_3d_no_decay.append(param)
            else:
                params_2d_no_decay.append(param)
        else:
            if name.startswith('core.branch_3d'):
                params_3d_decay.append(param)
            else:
                params_2d_decay.append(param)

    lr = getattr(cfgs, 'lr', None)
    lr_2d = getattr(cfgs, 'lr_2d', lr)
    lr_3d = getattr(cfgs, 'lr_3d', lr)

    params = [
        {'params': params_2d_decay, 'weight_decay': cfgs.weight_decay, 'lr': lr_2d},
        {'params': params_3d_decay, 'weight_decay': cfgs.weight_decay, 'lr': lr_3d},
        {'params': params_2d_no_decay, 'weight_decay': 0, 'lr': lr_2d},
        {'params': params_3d_no_decay, 'weight_decay': 0, 'lr': lr_3d},
    ]

    if cfgs.opt == 'adam':
        optimizer = Adam(params)
    elif cfgs.opt == 'adamw':
        optimizer = AdamW(params)
    else:
        raise NotImplementedError
    
    scheduler = create_scheduler(cfgs, optimizer)[0]

    return optimizer, scheduler
