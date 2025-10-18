import os
import sys
import tqdm
import shutil
import argparse
import warnings

import torch
import numpy as np

from emprot.utils.cryo_utils import parse_map, write_map, enlarge_grid, MRCObject
from emprot.utils.torch_utils import seed_everything
from emprot.utils import residue_constants as rc
from emprot.utils.protein import get_protein_from_file_path
from emprot.utils.misc_utils import pjoin, abspath
from emprot.utils.refine import final_results_to_cif_refine

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
from emprot.models.model_post_refine import Model

def add_args(parser):
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
        default=pjoin(script_dir, "..", "weights", "model_all_atom_post_refine"),
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
    # For recycle
    parser.add_argument(
        "--recycle",
        type=int,
        default=3,
        help="Recycling times"
    )
    return parser

def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Seed everything
    seed_everything(42)

    final_output_dir = args.output_dir
    last_recycle_structure = args.protein

    for i in range(args.recycle):
        print("# Round {} / {}".format(i + 1, args.recycle))
        
        args.pdb = last_recycle_structure
        args.output_dir = os.path.join(final_output_dir, f"recycle_{i}")
        
        infer(args)

        last_recycle_structure = os.path.join(args.output_dir, "output.cif")

    # Final output
    pass

def infer(args):
    # Set final output dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Device
    device_names = get_device_names(args.device)
    num_devices = len(device_names)

    # Read protein
    protein = None
    if args.protein.endswith(".cif") or args.protein.endswith(".pdb"):
        protein = get_protein_from_file_path(args.protein)

        # init random affines
        rand_prot = init_protein_from_translation(args.protein, random_shift=False)
        protein.rigidgroups_gt_frames = rand_prot.rigidgroups_gt_frames


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


    print("# Done")
    print("# See the refined model at {}".format(args.output_dir))
    final_results_to_cif_refine(
        final_results,
        protein,
        args.output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser).parse_args()
    main(args)

