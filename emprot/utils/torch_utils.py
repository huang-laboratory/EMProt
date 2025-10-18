import os
import glob
import stat
import shutil
import random
import warnings
import importlib.util

from collections import OrderedDict
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence

from emprot.utils.misc_utils import flatten_dict, unflatten_dict

def get_batch_slices(num_total: int, batch_size: int,) -> List[List[int]]:
    if num_total <= batch_size:
        return [list(range(num_total))]

    num_batches = num_total // batch_size
    batches = [
        list(range(i * batch_size, (i + 1) * batch_size)) for i in range(num_batches)
    ]
    if num_total % batch_size > 0:
        batches += [list(range(num_batches * batch_size, num_total))]
    return batches

def padded_sequence_softmax(
    padded_sequence_values: torch.Tensor,
    padded_mask: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-6,
) -> torch.Tensor:
    padded_softmax = torch.softmax(padded_sequence_values, dim=dim)
    padded_softmax = padded_softmax * padded_mask  # Mask out padded values
    padded_softmax = (
        padded_softmax / (padded_softmax.sum(dim=dim, keepdim=True) + eps).detach()
    )  # Renormalize
    return padded_softmax


def expand_as(x, y):
    n = len(x.shape)
    m = len(y.shape)

    assert m > n

    ones = (1,) * (m - n)

    return x.reshape(*x.shape, *ones)

def get_activation_function(string):
    acts = {
        "sigmoid": torch.sigmoid,
        "relu": torch.nn.ReLU(),
        "gelu": torch.nn.GELU(),
        "tanh": torch.tanh,
        "swish": lambda x: x * torch.sigmoid(x),
        "sin": torch.sin,
    }
    return acts[string]


def get_activation_class(string):
    acts = {
        "sigmoid": torch.nn.Sigmoid,
        "relu": torch.nn.ReLU,
        "gelu": torch.nn.GELU,
        "tanh": torch.nn.Tanh,
        "elu": torch.nn.ELU,
        "leaky_relu": torch.nn.LeakyReLU,
    }
    return acts[string]


def get_normalization_class(string):
    norms = {
        "batch": torch.nn.BatchNorm3d,
        "instance": torch.nn.InstanceNorm3d,
        "none": torch.nn.Identity,
    }
    return norms[string]


def get_pooling_cls(string):
    pooling = {
        "avg": torch.nn.AvgPool3d,
        "max": torch.nn.MaxPool3d,
    }
    return pooling[string]

# No get_spatial_cls

def count_parameters(model):
    return sum([p.numel() for p in model.parameters()])

def get_batches_to_idx(idx_to_batches: torch.Tensor) -> List[torch.Tensor]:
    assert len(idx_to_batches.shape) == 1
    max_batch_num = idx_to_batches.max().item() + 1
    idxs = torch.arange(0, len(idx_to_batches), dtype=int, device=idx_to_batches.device)
    return [idxs[idx_to_batches == i] for i in range(max_batch_num)]


def get_module_device(module: nn.Module) -> torch.device:
    return next(module.parameters()).device

def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def shared_cat(args, dim=0, is_torch=True) -> Union[torch.Tensor, np.ndarray]:
    if is_torch:
        return torch.cat(args, dim=dim)
    else:
        return np.concatenate(args, axis=dim)


def one_hot(index: int, num_classes: int, device: str = "cpu") -> torch.Tensor:
    return F.one_hot(torch.LongTensor([index]).to(device=device), num_classes)[0]


def is_ndarray(x) -> bool:
    return isinstance(x, np.ndarray)

def get_device_name(device_name: str) -> str:
    if device_name is None:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device_name == "cpu":
        return "cpu"
    if device_name.startswith("cuda:"):
        return device_name
    if device_name.isnumeric():
        return f"cuda:{device_name}"
    else:
        raise RuntimeError(
            f"Device name: {device_name} not recognized. "
            f"Either do not set, set to cpu, or give a number"
        )

def get_device_names(device_name_str: str) -> List[str]:
    if device_name_str is None or "," not in device_name_str:
        return [get_device_name(device_name_str)]
    else:
        return [get_device_name(x.strip()) for x in device_name_str.split(",") if len(x.strip()) > 0]


def set_overall_seed(seed: int):
    import torch, random, numpy
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


class ShapeError(Exception):
    pass


def compile_if_possible(module: nn.Module) -> nn.Module:
    if hasattr(torch, "compile"):
        module = torch.compile(module)
    return module

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

