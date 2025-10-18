import os
import sys
import pickle
import numpy as np
from typing import List
import torch
import random
import tempfile
import contextlib

def batch_iterator(iterator, batch_size):
    if len(iterator) <= batch_size:
        return [iterator]

    output = []
    i = 0

    while (len(iterator) - i) > batch_size:
        output.append(iterator[i : i + batch_size])
        i += batch_size

    output.append(iterator[i:])
    return output


def make_empty_dirs(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "coordinates"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "summary"), exist_ok=True)


def accelerator_print(string, accelerator):
    if accelerator.is_main_process:
        print(string)


def flatten_dict(dictionary: dict, level: List = []) -> dict:
    tmp_dict = {}
    for key, val in dictionary.items():
        if type(val) == dict:
            tmp_dict.update(flatten_dict(val, level + [str(key)]))
        else:
            tmp_dict[".".join(level + [str(key)])] = val
    return tmp_dict


def unflatten_dict(dictionary: dict, to_int: bool = True) -> dict:
    result_dict = dict()
    for key, value in dictionary.items():
        parts = key.split(".")
        if to_int:
            parts = [p if not p.isnumeric() else int(p) for p in parts]
        d = result_dict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return result_dict


def pickle_dump(obj: object, file_path: str):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def pickle_load(file_path: str) -> object:
    with open(file_path, "rb") as f:
        obj = pickle.load(f)
    return obj


def assertion_check(if_statement: bool, failure_message: str = ""):
    assert if_statement, failure_message


class FileHandle:
    def __init__(self, print_fn):
        self.print_fn = print_fn

    def write(self, *args, **kwargs):
        self.print_fn(*args, **kwargs)

    def flush(self, *args, **kwargs):
        pass


def filter_useless_warnings():
    import warnings

    warnings.filterwarnings("ignore", ".*nn\.functional\.upsample is deprecated.*")
    warnings.filterwarnings("ignore", ".*none of the inputs have requires_grad.*")
    warnings.filterwarnings("ignore", ".*with given element none.*")
    warnings.filterwarnings("ignore", ".*invalid value encountered in true\_divide.*")


class Args(object):  # Generic container for arguments
    def __init__(self, kwarg_dict):
        for (k, v) in kwarg_dict.items():
            setattr(self, k, v)

    def __repr__(self):
        return str(self.__dict__)

def upper_and_lower_case_annotation(string: str):
    output = []
    for char in string:
        if char.isalpha():
            if char.isupper():
                output.append("u")
            else:
                output.append("l")
    return string + "_" + "".join(output)


# Newly added
def atom14_to_atom37(
    atom14_pos,
    atom14_mask,
    res_type,
):
    raise NotImplementedError


def list_select(l, idxs):
    if isinstance(l, list):
        return [l[idx] for idx in idxs]
    if isinstance(l, np.ndarray):
        return l[idxs]

def abspath(path):
    return os.path.abspath(os.path.expanduser(path))

def pjoin(*p):
    return os.path.join(*p)

def find_first_of(iterable, x):
    # Iterate over the iterable with both index and value
    for index, value in enumerate(iterable):
        if value == x:
            return index  # Return the index if the element is found
    return -1  # Return -1 if the element is not found after exhausting the iterable

def find_first_not_of(iterable, x):
    # Iterate over the iterable with both index and value
    for index, value in enumerate(iterable):
        if value != x:
            return index  # Return the index of the first element that is not x
    return -1  # Return -1 if no element is found that is not x

def find_last_of(iterable, x):
    # Iterate over the iterable in reverse with both index and value
    for index, value in reversed(list(enumerate(iterable))):
        if value == x:
            return index  # Return the index if the element is found
    return -1  # Return -1 if the element is not found

def find_last_not_of(iterable, x):
    # Iterate over the iterable in reverse with both index and value
    for index, value in reversed(list(enumerate(iterable))):
        if value != x:
            return index  # Return the index of the first element that is not x
    return -1  # Return -1 if no element is found that is not x


def split_array(array, end_idxs):
    segments = []
    start_idx = 0
    for end_idx in end_idxs:
        segment = array[start_idx:end_idx+1]
        segments.append(segment)
        start_idx = end_idx + 1
    if start_idx <= len(array) - 1:
        segments.append(array[start_idx:])
    return segments

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def to_tensor(x):
    if x is None:
        return x
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    if x.dtype in [np.float32, np.float64]:
        return torch.from_numpy(x).float()
    elif x.dtype in [np.int32, np.int64]:
        return torch.from_numpy(x).int()
    else:
        return torch.from_numpy(x)

@contextlib.contextmanager
def get_temp_dir(provided_dir=None):
    if provided_dir is not None:
        yield provided_dir
    else:
        with tempfile.TemporaryDirectory() as tmp:
            yield tmp
