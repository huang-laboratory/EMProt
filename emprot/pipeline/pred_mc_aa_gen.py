import os
import gc
import sys
import json
import time
import torch
import random
import argparse
import numpy as np
from torch import nn
from tqdm import tqdm
from math import ceil
from torch import FloatTensor as FT
import warnings
warnings.filterwarnings('ignore')

from emprot.scunet.scunet import SCUNet
from emprot.utils.torch_utils import get_device_names
from emprot.utils.cryo_utils import parse_map, write_map, pad_map, chunk_generator, get_batch_from_generator, map_batch_to_map

def pjoin(*args):
    return os.path.join(*args)

def abspath(path):
    return os.path.abspath(os.path.expanduser(path))

def softmax(x, axis=0):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def load_model_and_run_inference_on_map(model_file, map_file, **kwargs):
    # Load map data
    apix = kwargs['apix']
    stride = kwargs['stride']
    box_size = kwargs['box_size']
    n_classes = kwargs['n_classes']
    batch_size = kwargs['batch_size']

    device = kwargs['device'] # cpu or cuda:x

    print("# Load map data from {}".format(map_file))
    map, origin, nxyz, voxel_size = parse_map(map_file, ignorestart=False, apix=apix)
    # Clip stride lower-bound for too large maps
    if np.min(nxyz) > 400:
        stride = max(stride, 16)

    print("# Map dimensions = {}".format(nxyz))
    #print("# Using stride = {}".format(stride))

    # Load model weights instead of model
    model_state_dict = torch.load(model_file)
    model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}

    model = SCUNet(input_resolution=box_size, n_classes=n_classes)
    model.load_state_dict(model_state_dict)

    model = model.to(device)
    model.eval()

    # new version
    # use generator to save memory
    padded_map = pad_map(map, box_size, dtype=np.float32, padding=0.0)
    maximum = np.percentile(map[map > 0], 99.999)
    #del map

    map_pred = np.zeros( (n_classes, ) + padded_map.shape, dtype=np.float32 ) # (n_classes, h, w, d)
    denominator = np.zeros( (n_classes, ) + padded_map.shape, dtype=np.float32 ) # (n_classes, h, w, d)

    print("# Start processing")

    generator = chunk_generator(padded_map, maximum, box_size, stride)
    ncx, ncy, ncz = [ceil(nxyz[2-i] / stride) for i in range(3)]
    total_steps = float(ncx * ncy * ncz)
    acc_steps, acc_steps_x, l_bar = 0.0, 0, 0

    ts = time.time()
    with torch.inference_mode():
        while True:
            positions, chunks = get_batch_from_generator(generator, batch_size, dtype=np.float32)

            if len(positions) == 0:
                break

            acc_steps += len(chunks)
            acc_steps_x = int((acc_steps / total_steps) * 100.0) // 5
            if acc_steps_x > l_bar:
                l_bar = acc_steps_x
                te = time.time()
                bar = f"|{'#' * (2*l_bar)}{'-' * ((20-l_bar)*2)}| {int(l_bar*5)}% {te-ts:.4f} seconds elapsed"
                print(f"\r{bar}", flush=True)

            X = FT(chunks).view(-1, 1, box_size, box_size, box_size)

            X = X.to(device)

            y_pred = model(X) # (b, n_classes, h, w, d)
            y_pred = y_pred.cpu().detach().numpy()

            map_pred, denominator = map_batch_to_map(map_pred, denominator, positions, y_pred, box_size)

    map_pred = (map_pred/denominator.clip(min=1))[
        :, # add a channel dim
        box_size:box_size+nxyz[2], 
        box_size:box_size+nxyz[1], 
        box_size:box_size+nxyz[0]
    ]

    if acc_steps < total_steps:
        bar = f"|{'#' * 40}| 100%"
        print(f"\r{bar}", flush=True)
    
    return map_pred, map, origin, nxyz, voxel_size

# Data params
data_params = {
    "apix": 1.0,
    "box_size": 48,
    "stride": 16,
}

# Test params
test_params = {
    "batch_size": 160
}

# Only inference ncac
def inference_mc(dir_map, contour, dir_model, dir_out, data_params, test_params, device):
    print(f"# Select map contour at {contour:.6f}", flush=True)
    print(f"# Running on device {device}")

    batch_size = test_params['batch_size']

    apix = data_params['apix']
    box_size = data_params['box_size']
    assert box_size % 2 == 0
    stride = data_params['stride']

    map_pred, map, origin, _, voxel_size = load_model_and_run_inference_on_map(
        model_file=dir_model,
        map_file=dir_map,

        apix=apix,
        box_size=box_size,
        stride=stride,
        n_classes=3,
        batch_size=batch_size,

        device=device,
    )
    mask = np.where(map <= contour, 0, 1).astype(np.int8)
    types = ["n.mrc", "ca.mrc", "c.mrc"]
    for i in range(3):
        out = mask * map_pred[i]
        dir_map_out = os.path.join(dir_out, types[i])
        write_map(dir_map_out, out.astype(np.float32), voxel_size, origin=origin)
        print("# Write map to {}".format(dir_map_out), flush=True)

    # write combined N, Ca, C to MC.mrc
    summed_map = map_pred.sum(axis=0) # (3, L, M, N) -> (L, M, N)
    dir_map_out = os.path.join(dir_out, "mc.mrc")
    write_map(dir_map_out, out.astype(np.float32), voxel_size, origin=origin)
    print("# Write map to {}".format(dir_map_out), flush=True)


# Only inference aa
def inference_aa(dir_map, contour, dir_model, dir_out, data_params, test_params, device):
    print(f"# Select map contour at {contour:.6f}", flush=True)
    print(f"# Running on device {device}")

    batch_size = test_params['batch_size']

    apix = data_params['apix']
    box_size = data_params['box_size']
    assert box_size % 2 == 0
    stride = data_params['stride']

    map_pred, map, origin, _, voxel_size = load_model_and_run_inference_on_map(
        model_file=dir_model,
        map_file=dir_map,

        apix=apix,
        box_size=box_size,
        stride=stride,
        n_classes=20,
        batch_size=batch_size,

        device=device,
    )
    #mask = np.where(map <= contour, 0, 1).astype(np.int8)
   
    # Keep original logits
    dir_npz_out = os.path.join(dir_out, "aa_logits.npz")
    
    np.savez(
        dir_npz_out,
        map=map_pred, # [n_classes, H, W, D]
        origin=origin,
        voxel_size=voxel_size,
    )
    print("# Write aa logits npz file to {}".format(dir_npz_out), flush=True)


def main(args):
    seed_torch(42)

    """Set maximul threads used"""
    cpu_num = 4
    os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    start = time.time()

    # Get current directory
    dir_script = abspath(os.path.dirname(__file__))
    print("# Script dir is {}".format(dir_script))

    # Do deep learning prediction
    dir_map = abspath(args.input)
    dir_out = abspath(args.output)
    contour = args.contour
    if args.model is not None:
        dir_model = abspath(args.model)
    else:
        dir_model = pjoin(dir_script, "weights")
    contour = args.contour
    print("# Specify model path to {}".format(dir_model))

    if not os.path.exists(pjoin(dir_model, "model_mc")):
        raise Exception("Cannot find model weights for model_mc")
    if not os.path.exists(pjoin(dir_model, "model_aa")):
        raise Exception("Cannot find model weights for model_aa")

    print("# Making directory {}".format(dir_out), flush=True)
    os.makedirs(dir_out, exist_ok=True)

    if isinstance(args.stride, int):
        assert 12 <= args.stride <= 48, "Invalid stride = {} -> 12 <= stride <= 48".format(args.stride)
        data_params["stride"] = args.stride

    if args.batchsize is not None:
        test_params["batch_size"] = args.batchsize


    ## To save the time and memory space
    ## First zone a larger box according to the given contour
    #r_zone = 20
    #data, origin, _, vsize = parse_map(dir_map, False, None)
    #crds_above_contour = np.argwhere(data > contour)
    #crds_above_contour = np.flip(crds_above_contour, axis=-1)
    #crds_above_contour = crds_above_contour * vsize + origin
    #print("# Found {} voxels above contour level of {:.6f}".format(len(crds_above_contour), contour))
    #datax, originx = zone_box(coords=crds_above_contour, data=data, origin=origin, voxel_size=vsize, r_zone=r_zone)
    #print("# Zone the map of shape {} to box of shape {}".format(data.shape, datax.shape))

    #dir_zmap = os.path.join(dir_out, f"box_zoned_by_contour_r_{r_zone}.mrc")
    #write_map(dir_zmap, map=datax, origin=originx, voxel_size=vsize)
    #print("# Write zoned map to {}".format(dir_zmap))
    ## Make sure they are cleaned by python GC
    #del datax, data
    #gc.collect()



    # Parse devices
    devices = get_device_names(args.device)
    device = devices[0]


    # Run N CA C
    if args.predict_mc:
        print("# Start inference", flush=True)
        inference_mc(dir_map, contour, os.path.join(dir_model, "model_mc"), dir_out, data_params, test_params, device=device)
        print("# Done  inference", flush=True)
    else:
        print("# Skip mc prediction\n")

    # Run aa
    if args.predict_aa:
        print("# Start inference", flush=True)
        inference_aa(dir_map, contour, os.path.join(dir_model, "model_aa"), dir_out, data_params, test_params, device=device)
        print("# Done  inference", flush=True)
    else:
        print("# Skip aa prediction\n")


    if "cuda" in device:
        print("# Clean CUDA cache")
        torch.cuda.empty_cache()


    # End
    end = time.time()
    print("# Time consuming {:.4f}".format(end - start), flush=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True, help="Input EM density map file")
    parser.add_argument("--output", "-o", type=str, default='./', help="Output directory of predicted maps")
    parser.add_argument("--contour", "-c", type=float, default=1e-6, help="Input contour level")
    parser.add_argument("--batchsize", "-b", type=int, default=40, help="Batchsize for prediction")
    parser.add_argument("--device", "-g", type=str, help="Which GPU to use, '0' for #0", default='0')
    parser.add_argument("--model", "-m", type=str, help="Directory to deep learning models")
    parser.add_argument("--stride", "-s", type=int, help="Stride for splitting chunks", default=16)
    parser.add_argument("--usecpu", action='store_true', help="Run prediction on CPU")
    # For zone-box
    #parser.add_argument("--no-zone-box", action='store_true', help="Not zone a larger box, keep map shape")
    # Prediction option
    parser.add_argument("--predict-mc", "--mc", "-mc", action='store_true', help="Run mc prediction")
    parser.add_argument("--predict-aa", "--aa", "-aa", action='store_true', help="Run aa prediction")
    # If aa prediction, can also specify ca coordinates
    #parser.add_argument("--ca-coords", "--ca", "-ca", type=str, help="Input ca coordinates")
    args = parser.parse_args()
    main(args)

