import os
import sys
import tqdm
import torch
import shutil
import tempfile
import warnings
import argparse
import numpy as np
from collections import namedtuple

from emprot.models.model import Model

from emprot.utils.cryo_utils import parse_map, write_map, enlarge_grid, MRCObject
from emprot.utils.torch_utils import seed_everything
from emprot.utils import residue_constants as rc
from emprot.utils.protein import get_protein_from_file_path
from emprot.utils.misc_utils import pjoin, abspath

from emprot.utils.multi_gpu_wrapper import MultiGPUWrapper
from emprot.utils.torch_utils import get_device_names
from emprot.utils.gnn_inference_utils import (
    init_empty_collate_results,
    init_protein_from_translation,
    get_neighbour_idxs,
    argmin_random,
    get_inference_data,
    run_inference_on_data,
    collate_nn_results,
    get_final_nn_results, 
)


def infer(args):
    # Set output dir
    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = os.path.dirname(args.output_dir)

    # Device
    device_names = get_device_names(args.device)
    num_devices = len(device_names)

    # Read protein
    protein = None
    if args.protein.endswith(".cif") or args.protein.endswith(".pdb"):
        print("# Read structure from {}".format(args.protein))
        if not args.no_use_random_affine:
            print("# Use random affine")
            protein = init_protein_from_translation(args.protein, random_shift=False)
        else:
            print("# Use affine from file")
            protein = get_protein_from_file_path(args.protein)


    if protein is None:
        raise RuntimeError(f"File {args.protein} is not a supported file format.")

    if not (args.map.endswith(".mrc") or args.map.endswith(".map")):
        warnings.warn(f"The file {args.map} does not end with '.mrc' or '.map'\nPlease make sure it is an MRC file.")

    # Read map data
    grid, origin, _, voxel_size = parse_map(args.map, False, args.voxel_size)

    # Process grid (this does not change grid origin)
    maximum = np.percentile(grid[grid > 0.0], 99.999)
    grid = np.clip(grid, a_min=0.0, a_max=maximum)
    grid = grid / (maximum + 1e-6)
    grid = enlarge_grid(grid)

    grid_data = MRCObject(
        grid=grid,
        origin=origin, 
        voxel_size=voxel_size,
    )

    # Process data
    num_res = len(protein.rigidgroups_gt_frames)

    collated_results = init_empty_collate_results(num_res, device="cpu",)

    residues_left = num_res
    total_steps = num_res * args.repeat_per_residue
    steps_left_last = total_steps

    pbar = tqdm.tqdm(total=total_steps, file=sys.stdout, position=0, leave=True)

    # Get an initial set of pointers to neighbours for more efficient inference
    num_pred_residues = 50 if num_res > args.crop_length else num_res
    init_neighbours = get_neighbour_idxs(protein, k=num_pred_residues)

    model_class = Model
    model_args = {
        "n_block": 16, 
        "n_tors": rc.canonical_num_residues * 5 + 3, 
        "use_checkpoint": False,
    }
    state_dict_path = args.model_dir

    with MultiGPUWrapper(model_class, model_args, state_dict_path, device_names, args.fp16) as wrapper:
        while residues_left > 0:
            idxs = argmin_random(
                collated_results["counts"], init_neighbours, args.batch_size * num_devices
            )
            data = get_inference_data(
                protein, grid_data, idxs, crop_length=args.crop_length, num_devices=num_devices,
            )
            # run iter = 1/2
            results = run_inference_on_data(wrapper, data, fp16=args.fp16, run_iters=1)
            for device_id in range(num_devices):
                for i in range(args.batch_size):
                    collated_results, protein = collate_nn_results(
                        collated_results,
                        results[device_id],
                        data[device_id]["indices"],
                        protein,
                        offset=i * args.crop_length,
                        num_pred_residues=num_pred_residues,
                    )
            residues_left = (
                num_res
                - torch.sum(collated_results["counts"] > args.repeat_per_residue - 1).item()
            )
            steps_left = (
                total_steps
                - torch.sum(
                    collated_results["counts"].clip(0, args.repeat_per_residue)
                ).item()
            )
            pbar.update(n=int(steps_left_last - steps_left))
            steps_left_last = steps_left

    pbar.close()

    final_results = get_final_nn_results(collated_results)


    if "cuda" in device_names[0]:
        print("# Clean CUDA cache")
        #torch.cuda.empty_cache()


    if not args.refine:
        ##########################################
        ### Assign aa type for denovo modeling ###
        ##########################################
        print("# Assign restype for denovo modeling")
        from emprot.utils.denovo import final_results_align_to_sequence
        final_results = final_results_align_to_sequence(
            final_results, 
            args.aamap, 
            args.sequence, 
            args.output_dir,
        )
    else:
        #############################
        ### Get all-atom position ###
        #############################
        print("# Get all-atom position")
        from emprot.utils.refine import final_results_to_cif_refine
        final_results_to_cif_refine(
            final_results,
            protein, 
            args.output_dir, 
        )


def get_args():
    script_dir = abspath(os.path.dirname(__file__))
    num_res_per_run = 200
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", "--i", required=True, help="The path to the input map")
    parser.add_argument(
        "--protein", "--p", required=True, help="The path to the protein file"
    )
    parser.add_argument(
        "--model-dir", 
        help="Where the model at",
        default=pjoin(script_dir, "..", "weights", "model_all_atom"), 
    )
    parser.add_argument("--output-dir", default=".", help="Where to save the results")
    parser.add_argument("--device", default="cpu", help="Which device to run on")
    parser.add_argument(
        "--crop-length", type=int, default=num_res_per_run, help="How many points per batch"
    )
    parser.add_argument(
        "--repeat-per-residue",
        default=1,
        type=int,
        help="How many times to repeat per residue",
    )
    parser.add_argument(
        "--batch-size", default=1, type=int, help="How many batches to run in parallel"
    )
    parser.add_argument("--fp16", action="store_true", help="Use fp16 in inference")
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=1.0,
        help="The voxel size that the GNN should be interpolating to."
    )
    # For refine only
    parser.add_argument(
        "--refine", 
        action='store_true', 
        help="Refine of structure",
    )

    # For denovo
    parser.add_argument(
        "--aamap",
        type=str,
        help="Predicted aamap",
    )
    parser.add_argument(
        "--sequence",
        "--seq",
        "-s",
        type=str,
        help="Input sequence(s)"
    )
    parser.add_argument(
        "--no-use-random-affine",
        action='store_true',
    )
    parser.add_argument(
        "--recycle",
        type=int,
        default=3, 
        help="Recycling times"
    )
    args = parser.parse_args()
    return args


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # seed everything
    seed_everything(42)

    # Set final output dir
    output_dir = args.output_dir 

    # Read aa map
    print("# Load aa logits map")
    aa_map = np.load(args.aamap)
    print("# AA map shape = {}".format(aa_map['map'].shape))
    args.aamap = aa_map

    # Infer rounds
    n_round_refine = args.recycle
    if n_round_refine < 1:
        n_round_refine = 1

    last_output_dir = None
    for i in range(n_round_refine):
        if i > 0:
            args.no_use_random_affine = True
        else:
            args.no_use_random_affine = False

        print("# Infer {} / {}".format(i + 1, n_round_refine))

        # Set output dir
        args.output_dir = os.path.join(output_dir, f"recycle_{i}")

        infer(args)

        last_output_dir = os.path.join(output_dir, f"recycle_{i}", "output.cif")
        args.protein = last_output_dir

    print("# Done all rounds")

if __name__ == "__main__":
    args = get_args()
    main(args)
