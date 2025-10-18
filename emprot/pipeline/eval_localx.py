"""
Evaluate built model
"""
import os
import numpy as np
from scipy.spatial import KDTree

from emprot.io.pdbio import read_pdb, read_pdb_simple, convert_to_chains, chains_atom_pos_to_pdb
from emprot.utils.grid import label_map, grid_value_interp
from emprot.utils.cryo_utils import parse_map, write_map

def smooth_window(data, kernel=[1.0, 4.0, 9.0, 4.0, 1.0]):
    # Kernel size must be odd
    assert len(kernel) % 2 == 1, "# Error Kernel size must be odd number"
    kernel = np.asarray(kernel, dtype=np.float32)
    window = len(kernel) // 2
    # . . . . . . . c . . . . . . .
    # . . . . . w w w w w . . . . .
    ret = np.zeros(len(data), dtype=np.float32)
    L = len(data)
    for center in range(L):
        l = center - window
        r = center + window + 1
        s = 0
        w = 1e-3

        for k in range(l, r):
            if 0 <= k < len(data):
                s += data[k] * kernel[k - l]
                w += kernel[k - l]
        ret[center] = s / w
    return ret


def main(args):
    # read predicted CAs
    raw_ca_pos, raw_ca_prob = read_pdb_simple(args.ca, ["CA"], res_type=False, bfactor=True)
    print("# Read {} predicted CA".format(len(raw_ca_pos)))

    # read pdb
    fpdb = args.pdb
    atom_pos, atom_mask, res_type, res_idx, chain_idx = read_pdb(fpdb, ignore_hetatm=True, keep_valid=False)
    print("# Read {} input residues".format(len(atom_pos)))
    
    assert np.all(np.logical_not(np.isnan(atom_pos[..., 1, :]))), "# CA coords have NaN"


    # read original map
    apix = 1.0
    grid, origin, nxyz, vsize = parse_map(args.map, False, apix)
    del grid

    # label map by CA position and density
    grid = label_map(origin, nxyz, apix, raw_ca_pos, raw_ca_prob, args.resolution)
    dmax = np.percentile(grid, 99.999)
    grid = grid / (dmax + 1e-6)
    
    scores = grid_value_interp(atom_pos[..., 1, :], grid, origin, vsize)
    scores = np.sqrt(scores)

    # smooth
    if args.smooth:
        print("# Smooth score")
        scores = smooth_window(scores)

    # statistic
    mean = scores.mean()
    std = scores.std()
    #print("# residue 1   std - mean - 1   std = {:.4f} {:.4f} {:.4f}".format(mean-1.0*std, mean, mean+1.0*std))
    #print("# residue 1.5 std - mean - 1.5 std = {:.4f} {:.4f} {:.4f}".format(mean-1.5*std, mean, mean+1.5*std))
    #print("# residue 2   std - mean - 2   std = {:.4f} {:.4f} {:.4f}".format(mean-2.0*std, mean, mean+2.0*std))
    print("# Mean plddt = {:.4f}".format(mean))

    # Repeat for each atom
    scores = np.repeat(scores[..., None], 14, axis=-1)
    # format like x.xxxx
    scores = (scores * 10000).astype(np.int32) / 10000.0

    scores = np.clip(scores, a_min=0.0, a_max=0.9999)

    # Write score to bfactor
    chains_atom_pos, \
    chains_atom_mask, \
    chains_res_type, \
    chains_res_idx, \
    chains_scores = convert_to_chains(
        chain_idx,
        atom_pos, atom_mask, res_type, res_idx, scores, 
    )
    fpdbout = os.path.join(args.output, "score.cif")
    os.makedirs(args.output, exist_ok=True)

    chains_atom_pos_to_pdb(
        fpdbout, 
        chains_atom_pos,
        chains_atom_mask,
        chains_res_type,
        chains_res_idx,
        chains_bfactors=chains_scores,
    )
    print("# Write scored PDB to {}".format(fpdbout))

def add_args(parser):
    parser.add_argument("--output", "-o", help="Output PDB", default="./")
    parser.add_argument("--pdb", "-p", help="Input pdb", required=True)
    parser.add_argument("--map", "-m", help="Input map", required=True)
    parser.add_argument("--ca", "-ca", help="Predicted CA atoms", required=True)
    parser.add_argument("--resolution", help="Resolution", default=5.0, type=float)
    parser.add_argument("--smooth", "-s", help="Smooth score", action='store_true')
    return parser

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args = add_args(parser).parse_args()
    main(args)
