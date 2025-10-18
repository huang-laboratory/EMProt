import os
import torch
import numpy as np
from typing import Dict, Callable, List

from emprot.utils.misc_utils import pjoin, abspath
from emprot.utils.grid import get_grid_value

from emprot.io.pdbio import read_pdb

from emprot.utils.aa_probs_to_hmm import dump_aa_logits_to_hmm_file

def search(
    pdb_dir, 
    aamap_dir, 
    seq_dir, 
    output_dir, 
):
    # Read structure
    atom_pos, atom_mask, res_type, res_idx, chain_idx = read_pdb(pdb_dir, keep_valid=True)

    # Setting tempdir
    out_dir = abspath(output_dir)
    hmm_temp_dir = pjoin(out_dir, "hmm_profiles")
    os.makedirs(hmm_temp_dir, exist_ok=True)
    print("# Setting output directory to {}".format(out_dir))
    print("# Setting HMM temp dir to {}".format(hmm_temp_dir))

    # Read aa logits
    print("# Reading aa logits/probabilities")
    faamap = aamap_dir
    aa_logits_npz = np.load(faamap)
    aa_logits_map = aa_logits_npz['map']
    aa_logits_origin = aa_logits_npz['origin']
    aa_logits_vsize = aa_logits_npz['voxel_size']
    print("# AA logits have shape of {}".format(aa_logits_map.shape))

    n_chain = chain_idx.max() + 1
    chains_ca_pos = []
    for i in range(n_chain):
        chains_ca_pos.append(atom_pos[..., 1, :])

    # Assign aa_logits for each coords
    print("# Assign aa logits")
    chains_aa_logits = []
    ca_pos = np.concatenate(chains_ca_pos).reshape(-1, 3)
    chains = []
    chains_prot_mask = []

    start = 0
    for i, acoords in enumerate(chains_ca_pos):
        # AA logits
        aa_logits = []
        for k in range(len(acoords)):
            logits = get_grid_value(aa_logits_map, (acoords[k]-aa_logits_origin)/aa_logits_vsize)
            aa_logits.append(logits)

        aa_logits = np.asarray(aa_logits, dtype=np.float32)

        # Dump aa logits to file
        output_file = pjoin(hmm_temp_dir, f"{i}_u.hmm")
        dump_aa_logits_to_hmm_file(
            aa_logits=aa_logits,
            output_file=output_file, 
            name=f"{i}", 
        )

        print(f"# Dump HMM file to {output_file}")


def get_args():
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", "-p", required=True)
    parser.add_argument("--aamap", required=True)
    parser.add_argument("--seq", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    return args

def main(args):
    search(
        args.pdb,
        args.aamap,
        args.seq,
        args.out,
    )

if __name__ == '__main__':
    args = get_args()
    main(args)
