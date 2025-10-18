import os
import time
import subprocess
import numpy as np
from scipy.spatial import KDTree

from emprot.utils.misc_utils import pjoin, abspath
from emprot.io.pdbio import read_pdb_simple

def distance(x, y):
    # x, y - (*, d)
    return np.sqrt(np.sum(np.power((x-y), 2), axis=-1))

def combine_ncac_pos(
        n_pos: np.ndarray, 
        ca_pos: np.ndarray, 
        c_pos: np.ndarray,
    ):
    # Check type
    if not isinstance(n_pos, np.ndarray):
        n_pos = np.asarray(n_pos, dtype=np.float32)
    if not isinstance(ca_pos, np.ndarray):
        ca_pos = np.asarray(ca_pos, dtype=np.float32)
    if not isinstance(c_pos, np.ndarray):
        c_pos = np.asarray(c_pos, dtype=np.float32)

    # Combine n_pos, ca_pos, c pos to to ncac_pos based on ca_pos
    # ca_pos - (L, 3)
    res_pos = []
    res_flag = []

    n_tree = KDTree(n_pos)
    c_tree = KDTree(c_pos)

    # Classical bond length
    d_ca_n = 1.459
    d_ca_c = 1.525
    # Use a wide tolerance
    d_ca_n_tolerance = 0.50
    d_ca_c_tolerance = 0.50

    use_ideal_bond = True
    # Build atom from ca
    def build_atom_from_ca(ca, src_pos, src_tree, d_bond=1.5, d_tolerance=0.5, eps=1e-3):
        is_good = False
        # Query candidates
        idxs = src_tree.query_ball_point(ca, r=2.1)
        idxs = [int(idx) for idx in idxs]
 
        # Sort according to distance
        ds = [distance(ca, src_pos[idx]) for idx in idxs]
        sorted_idxs = np.argsort(ds, axis=0)
        idxs = [idxs[k] for k in sorted_idxs]
        ds = [ds[k] for k in sorted_idxs]

        if len(idxs) > 0:
            # Enumerate from the nearest till find a suitable one
            for k, idx in enumerate(idxs):
                if d_bond - d_tolerance < ds[k] < d_bond + d_tolerance:
                    # Use ideal bond
                    if not use_ideal_bond:
                        pos = src_pos[idx]
                    else:
                        v = src_pos[idx] - ca
                        v = v / (np.linalg.norm(v) + eps)
                        pos = ca + v * d_bond
                    is_good = True
                    break

        # If not found, then init with a nearest one and set it with a proper bond
        if not is_good:
            d, idx = src_tree.query(ca, k=1)
            idx = int(idx)
            v = src_pos[idx] - ca
            v = v / (np.linalg.norm(v) + eps)
            pos = ca + v * d_bond

        return pos, is_good

    # Build
    for i in range(len(ca_pos)):
        ca = ca_pos[i] # (3, )

        # Build N atom
        n, n_good = build_atom_from_ca(ca, n_pos, n_tree, d_ca_n, d_ca_n_tolerance)

        # Build C atom
        c, c_good = build_atom_from_ca(ca, c_pos, c_tree, d_ca_c, d_ca_c_tolerance)

        # Append
        res_pos.append(
            np.stack(
                [n, ca, c], axis=0
            ) # (3, 3)
        )
        
        # Mark flag
        res_flag.append(n_good and c_good)

    # To numpy array
    res_pos = np.asarray(res_pos, dtype=np.float32) # (L, 3, 3)
    res_flag = np.asarray(res_flag, dtype=bool) # (L, )

    return res_pos, res_flag


def prune_points(coords, confidence=None, lower=2.5, upper=4.5):
    keep = [True] * len(coords)
    tree = KDTree(coords)
    if confidence is None:
        confidence = np.ones(len(coords), dtype=np.float32)

    # If two CAs distance are < lower, only one with higher confidence is kept
    for i, coord in enumerate(coords):
        idxs = tree.query_ball_point(coord, r=lower)
        idxs = np.asarray(idxs, np.int32)
        idxs = idxs[np.argsort(confidence[idxs], kind='stable')]
        for idx in idxs[:-1]:
            keep[idx] = False

    # For each CA, if its nearest neighbor is > upper, its not kept
    tree = KDTree(coords)
    for i, coord in enumerate(coords):
        ds, idxs = tree.query(coord, k=2)
        if ds[1] > upper:
            keep[i] = False

    return keep



def split_chain(atom3_pos, d=5.0, bond="C-N"):
    def distance(a, b):
        return np.sqrt(np.sum(np.power(a-b, 2)))

    # atom3_pos (L, 3, 3) and ordered
    # Split chain when consecutive distance is larger than d
    # If use C-N bond
    if bond == 'C-N':
        idx_prv = 2
        idx_nxt = 0
    # If use CA-CA pbond
    elif bond == "CA-CA":
        idx_prv = 1
        idx_nxt = 1
    else:
        raise "Not implemented"

    chains = []
    last = -1
    for k in range(len(atom3_pos) - 1):
        d0 = distance(atom3_pos[k][idx_prv], atom3_pos[k+1][idx_nxt])
        print(d0)

        if d0 > d:
            chain = []
            for kk in range(last + 1, k + 1):
                chain.append(atom3_pos[last])
            if len(chain) > 0:
                chains.append(chain)
            last = k

    if len(chain) > 0:
        chains.append(chain)
    return chains


def run_getp(map_dir, out_dir, lib_dir, pdb=None, res=6.0, thresh=20, nt=4, filter=0.0, dmerge=1.0, verbose=False, **kwargs):
    cmd = "{}/bin/getp --in {} --out {} --thresh {} --res {} --nt {} --filter {} --dmerge {}".format(lib_dir, map_dir, out_dir, thresh, res, nt, filter, dmerge)

    if pdb is not None:
        cmd += " --pdb {}".format(pdb)

    if verbose:
        print(f"# Running command {cmd}")
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    success = result.returncode == 0
    return success


def main(args):
    ts = time.time()

    out_dir = abspath(args.output)
    os.makedirs(out_dir, exist_ok=True)

    lib_dir = abspath(args.lib)

    # Other params
    res = args.res
    thresh = args.thresh
    nt = args.nt
    filter = args.filter
    dmerge = args.dmerge

    assert 2.0 <= res <= 8.0, "2.0 <= res <= 8.0 but got {:.2f}".format(res)
    assert thresh >= 0.0, "thresh >= 0.0 but got {:.2f}".format(thresh)
    assert nt >= 1, "nt >= 1 but got {}".format(nt)

    getp_args = {
        'res': res,
        'thresh': thresh,
        'nt': nt,
        'filter': filter,
        'dmerge': dmerge,
    }
    print(
        "# Running getp with args res={} thresh={} nt={}".format(
        getp_args['res'],
        getp_args['thresh'],
        getp_args['nt'],
        getp_args['filter'],
        getp_args['dmerge'],
        )
    )

    fcamap = args.camap

    if args.ca is None:
        print("# No   initial Ca positions are specified, shift on grids")
    else:
        print("# Read initial Ca positions from provided data path {}".format(args.ca))
        

    # Convert predicted map to points
    print("# Convert atom probability map to coords")
    fca = pjoin(out_dir, "raw_ca.pdb")
    ca_success = run_getp(
        fcamap, 
        fca, 
        lib_dir=lib_dir, 
        pdb=args.ca,
        **getp_args, verbose=True,
    )

    if not ca_success:
        print("Cannot convert camap")
 
    te = time.time()
    #print("# Time consuming {:.4f}".format(te-ts))


if __name__ =='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--camap", "-camap", help="Input camap")
    # If input coords
    parser.add_argument("--ca", "-ca", help="Input Ca coords in pdb format")
    # Others
    parser.add_argument("--output", "-o", help="Output directory", default='./')
    parser.add_argument("--lib", "-l", help="Lib directory")
    # Mean-Shift controls
    parser.add_argument("--thresh", "-thresh", type=float, default=10.0, help="Map threshold")
    parser.add_argument("--res", "-res", type=float, default=6.0, help="Resolution")
    parser.add_argument("--nt", "-nt", type=int, default=4, help="Num of threads")
    parser.add_argument("--filter", "-filter", type=float, default=0.0, help="Filter thresh")
    parser.add_argument("--dmerge", "-dmerge", type=int, default=1.5, help="Merge distance")
    args = parser.parse_args()
    main(args)
