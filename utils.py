import re
import os
import cv2
import sys
import smtplib
import logging
import numpy as np
import torch
import torch.utils.data
import torch.distributed as dist
from tqdm import tqdm
from email.mime.text import MIMEText
from omegaconf import DictConfig


class _RepeatSampler(object):
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class FastDataLoader(torch.utils.data.dataloader.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class BadLossChecker:
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.bad_steps = 0

    def check(self, loss):
        if not self.cfgs.enabled:
            return False

        if loss.isnan() or loss.isinf() or loss.item() > self.cfgs.threshold:
            self.bad_steps += 1
            if self.bad_steps == self.cfgs.max_bad_steps:
                self.bad_steps = 0
                return True
        else:
            self.bad_steps = 0

        return False


def check_gpu_availability():
    try:
        import pynvml  # type: ignore[import]
    except ModuleNotFoundError:
        return("pynvml module not found, please install pynvml")

    from pynvml import NVMLError_DriverNotLoaded

    try:
        pynvml.nvmlInit()
    except NVMLError_DriverNotLoaded:
        return ("cuda driver can't be loaded, is cuda enabled?")

    n_gpus = pynvml.nvmlDeviceGetCount()

    for i in range(n_gpus):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        pids = [p.pid for p in procs]
        if os.getpid() in pids and len(pids) > 1:
            return False

    return True


def get_grad_norm(model, prefix, norm_type: float = 2.0):
    parameters = [p for n, p in model.named_parameters() if n.startswith(prefix)]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    param_norm = [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
    total_norm = torch.norm(torch.stack(param_norm), norm_type)
    return total_norm


def get_max_memory(device, n_gpus):
    mem = torch.cuda.max_memory_allocated(device=device)
    mem_mb = torch.tensor([mem / (1024 * 1024)], dtype=torch.int, device=device)
    if n_gpus > 1:
        dist.reduce(mem_mb, 0, op=dist.ReduceOp.MAX)
    return mem_mb.item()


def send_mail(sender, receivers, subject, content):
    message = MIMEText(content, 'plain', 'utf-8')
    message['From'] = sender
    message['To'] = ','.join(receivers)
    message['Subject'] = subject

    try:
        smtp_obj = smtplib.SMTP('localhost', 25)
        smtp_obj.sendmail('', receivers, message.as_string())
        logging.info("Mail sent successfully!")
    except smtplib.SMTPException as e:
        logging.error("An exception occurs! Failed to send mail!", e)


def init_logging(filename=None, debug=False):
    logging.root = logging.RootLogger('DEBUG' if debug else 'INFO')
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] - %(message)s')

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logging.root.addHandler(stream_handler)

    if filename is not None:
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)


def dist_reduce_sum(value, n_gpus):
    if n_gpus <= 1:
        return value
    tensor = torch.Tensor([value]).cuda()
    dist.all_reduce(tensor)
    return tensor


def copy_to_device(inputs, device, non_blocking=True):
    if isinstance(inputs, list):
        inputs = [copy_to_device(item, device, non_blocking) for item in inputs]
    elif isinstance(inputs, dict):
        inputs = {k: copy_to_device(v, device, non_blocking) for k, v in inputs.items()}
    elif isinstance(inputs, torch.Tensor):
        inputs = inputs.to(device=device, non_blocking=non_blocking)
    return inputs


def size_of_batch(inputs):
    if isinstance(inputs, list):
        return size_of_batch(inputs[0])
    elif isinstance(inputs, dict):
        return size_of_batch(list(inputs.values())[0])
    elif isinstance(inputs, torch.Tensor):
        return inputs.shape[0]
    else:
        raise TypeError('Unknown type: %s' % str(type(inputs)))


def override_cfgs(dst: DictConfig, src: DictConfig):
    for key in src:
        if isinstance(src[key], DictConfig):
            dst[key] = override_cfgs(dst[key], src[key])
        else:
            dst[key] = src[key]
    return dst


def eat_all_ram(device, reserved=1024):
    num = int(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024) - reserved
    cache = torch.rand([num, 1024, 256], dtype=torch.float32)
    while True:
        try:
            cache[:num, ...].to(device)
            del cache
            break
        except RuntimeError:
            num -= reserved


def load_fpm(filename):
    with open(filename, 'rb') as f:
        header = f.readline().rstrip()
        if header.decode("ascii") == 'PF':
            color = True
        elif header.decode("ascii") == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', f.readline().decode("ascii"))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise Exception('Malformed PFM header.')

        scale = float(f.readline().decode("ascii").rstrip())
        if scale < 0:  # little-endian
            endian = '<'
        else:
            endian = '>'  # big-endian

        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)

    return data


def load_flow(filepath):
    with open(filepath, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert (202021.25 == magic), 'Invalid .flo file: incorrect magic number'
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        flow = np.fromfile(f, np.float32, count=2 * w * h).reshape([h, w, 2])

    return flow


def load_flow_png(filepath, scale=64.0):
    # for KITTI which uses 16bit PNG images
    # see 'https://github.com/ClementPinard/FlowNetPytorch/blob/master/datasets/KITTI.py'
    # The -1 is here to specify not to change the image depth (16bit), and is compatible
    # with both OpenCV2 and OpenCV3
    flow_img = cv2.imread(filepath, -1)
    flow = flow_img[:, :, 2:0:-1].astype(np.float32)
    mask = flow_img[:, :, 0] > 0
    flow = flow - 32768.0
    flow = flow / scale
    return flow, mask


def save_flow(filepath, flow):
    assert flow.shape[2] == 2
    magic = np.array(202021.25, dtype=np.float32)
    h = np.array(flow.shape[0], dtype=np.int32)
    w = np.array(flow.shape[1], dtype=np.int32)
    with open(filepath, 'wb') as f:
        f.write(magic.tobytes())
        f.write(w.tobytes())
        f.write(h.tobytes())
        f.write(flow.tobytes())


def save_flow_png(filepath, flow, mask=None, scale=64.0):
    assert flow.shape[2] == 2
    assert np.abs(flow).max() < 32767.0 / scale
    flow = flow * scale
    flow = flow + 32768.0

    if mask is None:
        mask = np.ones_like(flow)[..., 0]
    else:
        mask = np.float32(mask > 0)

    flow_img = np.concatenate([
        mask[..., None],
        flow[..., 1:2],
        flow[..., 0:1]
    ], axis=-1).astype(np.uint16)

    cv2.imwrite(filepath, flow_img)


def load_disp_png(filepath):
    array = cv2.imread(filepath, -1)
    valid_mask = array > 0
    disp = array.astype(np.float32) / 256.0
    disp[np.logical_not(valid_mask)] = -1.0
    return disp, valid_mask


def save_disp_png(filepath, disp, mask=None):
    if mask is None:
        mask = disp > 0
    disp = np.uint16(disp * 256.0)
    disp[np.logical_not(mask)] = 0
    cv2.imwrite(filepath, disp)


def load_calib(filepath):
    with open(filepath) as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('P_rect_02'):
                proj_mat = line.split()[1:]
                proj_mat = [float(param) for param in proj_mat]
                proj_mat = np.array(proj_mat, dtype=np.float32).reshape(3, 4)
                assert proj_mat[0, 1] == proj_mat[1, 0] == 0
                assert proj_mat[2, 0] == proj_mat[2, 1] == 0
                assert proj_mat[0, 0] == proj_mat[1, 1]
                assert proj_mat[2, 2] == 1

    return proj_mat


def zero_padding(inputs, pad_h, pad_w):
    input_dim = len(inputs.shape)
    assert input_dim in [2, 3]

    if input_dim == 2:
        inputs = inputs[..., None]

    h, w, c = inputs.shape
    assert h <= pad_h and w <= pad_w

    result = np.zeros([pad_h, pad_w, c], dtype=inputs.dtype)
    result[:h, :w] = inputs

    if input_dim == 2:
        result = result[..., 0]

    return result


def disp2pc(disp, baseline, f, cx, cy, flow=None):
    h, w = disp.shape
    depth = baseline * f / (disp + 1e-5)

    xx = np.tile(np.arange(w, dtype=np.float32)[None, :], (h, 1))
    yy = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))

    if flow is None:
        x = (xx - cx) * depth / f
        y = (yy - cy) * depth / f
    else:
        x = (xx - cx + flow[..., 0]) * depth / f
        y = (yy - cy + flow[..., 1]) * depth / f

    pc = np.concatenate([
        x[:, :, None],
        y[:, :, None],
        depth[:, :, None],
    ], axis=-1)

    return pc


def project_pc2image(pc, image_h, image_w, f, cx=None, cy=None, clip=True):
    pc_x, pc_y, depth = pc[..., 0], pc[..., 1], pc[..., 2]

    cx = (image_w - 1) / 2 if cx is None else cx
    cy = (image_h - 1) / 2 if cy is None else cy

    image_x = cx + (f / depth) * pc_x
    image_y = cy + (f / depth) * pc_y

    if clip:
        return np.concatenate([
            np.clip(image_x[:, None], a_min=0, a_max=image_w - 1),
            np.clip(image_y[:, None], a_min=0, a_max=image_h - 1),
        ], axis=-1)
    else:
        return np.concatenate([
            image_x[:, None],
            image_y[:, None]
        ], axis=-1)
