import os
import sys
import time
import glob
import random
import argparse
import tempfile
import numpy as np

from emprot.io.fileio import writelines
from emprot.io.pdbio import read_pdb, chains_atom_pos_to_pdb

from emprot.utils.misc_utils import abspath, pjoin
from emprot.utils.domain import (
    run_unidoc, 
    parse_unidoc_result, 
    annotate_pdb_with_domains,
    convert_domains_to_1d_repr,
    merge_domains_simple,
)

from emprot.utils.dock_and_refine import run_flex_refine_pipeline
from emprot.utils.secstr import assign, segment_secstr, update_partition, remap_partition
from emprot.utils.domain import convert_1d_repr_to_domains

from emprot.utils.geo import random_euler_safe, euler_to_rot_mat

def main(args):
    # set random seed
    random.seed(42)
    np.random.seed(42)

    ts = time.time()
    fpdbs = args.pdb
    fmap = args.map
    lib_dir = abspath(args.lib)
    out_dir = abspath(args.output)
    os.makedirs(out_dir, exist_ok=True)
    temp_dir = pjoin(out_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    verbose = args.verbose

    # split secondary structures and write into domains
    dock_map_dir = fmap
    dock_pdb_dir = []
    dock_pdb_domains = []
    dock_pdb_init_trans_dir = []

    print("Parsing secondary structure")
    for i, fpdb in enumerate(fpdbs):
        # for each chain
        templ_dir = os.path.dirname(fpdb)

        atom_pos, atom_mask, res_type, res_idx, chain_idx, bfactor = read_pdb(fpdb, return_bfactor=True)
        sec_idx = assign(atom_pos)
        #print(sec_idx)

        # 1. parse sec and merge loop
        frag_idx = segment_secstr(sec_idx)
        #print(frag_idx)
        frag_idx = update_partition(sec_idx, frag_idx)
        #print(frag_idx)
        frag_idx = remap_partition(frag_idx)
        frag_idx = np.asarray(frag_idx, dtype=np.int32)

        # 2. annotate input PDB with fragment index
        domains = convert_1d_repr_to_domains(frag_idx)




        # 3. append
        dock_pdb_dir.append(fpdb)
        dock_pdb_domains.append(domains)

    #print(len(dock_pdb_dir))

    # parse docking options
    resolution = args.resolution
    mode = args.mode
    nt = args.nt
    thresh = args.thresh
    angle_step = args.angle_step
    fgrid = args.fgrid
    sgrid = args.sgrid

    assert 2.0 <= resolution <= 8.0, "2.0 <= resolution <= 8.0, but got {:.2f}".format(resolution)
    assert nt >= 1, "nt >= 1 but got {}".format(nt)
    assert thresh >= 0.0, "thresh >= 0.0 but got {:.6f}".format(thresh)
    assert angle_step >= 12.0, "angle_step >= 12.0 but got {:.2f}".format(angle_step)
    assert fgrid >= 1.0, "fgrid >= 1.0 but got {:.2f}".format(fgrid)
    assert sgrid >= 0.5, "sgrid >= 0.5 but got {:.2f}".format(sgrid)

    dock_args = {
        'resolution': resolution,
        'mode': mode,
        'nt': nt,
        'thresh': thresh,
        'angle_step': angle_step,
        'fgrid': fgrid,
        'sgrid': sgrid,
    }

    print("Docking args")
    for k, v in dock_args.items():
        print("{} -> {}".format(k, v))

    # Making temp dirs
    dock_domain_temp_dir = pjoin(out_dir, "temp")
    os.makedirs(dock_domain_temp_dir, exist_ok=True)


    ###########################################################################
    ###########################################################################
    ################ Flexible refinement ######################################
    ###########################################################################
    ###########################################################################
    # run dock and refine
    print("Running flex refine")
    fpdbout = pjoin(out_dir, "flex_refine.cif")
    flogout = pjoin(out_dir, "flex_refine.log")
    success = run_flex_refine_pipeline(
        map_dir=dock_map_dir,
        chains_pdb_dir=dock_pdb_dir,
        chains_domains=dock_pdb_domains,
        out_dir=fpdbout,
        log_dir=flogout,
        lib_dir=lib_dir,
        verbose=verbose,
        temp_dir=dock_domain_temp_dir,
        chains_init_trans_dir=dock_pdb_init_trans_dir,
        **dock_args,
    )
    print("Write docked pdbs to {}".format(fpdbout))


    te = time.time()
    #print("Time consuming = {:.4f}".format(te-ts), flush=True)

if __name__ == '__main__':
    script_dir = abspath(os.path.dirname(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", "-p", nargs='+', help="Input templates of AF/ESMFold")
    parser.add_argument("--map", "-m", help="Input predicted main-chain map")
    parser.add_argument("--lib", "-l", help="Lib directory", default=pjoin(script_dir, ".."))
    parser.add_argument("--verbose", "-v", action='store_true', help="Whether to pring log to stdout")
    parser.add_argument("--output", "-o", help="Output directory", default='./')
    # options
    parser.add_argument("--domain", help="Docking in domain level", action='store_true')
    parser.add_argument("--chain",  help="Docking in chain level", action='store_true')
    # docking options
    parser.add_argument('--resolution', '--res', '-res', type=float, default=5.0, help="Map resolution")
    parser.add_argument('--mode', '-mode', type=str, default='fast', help="Docking mode, 'normal' or 'fast'")
    parser.add_argument('--nt', '-nt', type=int, default=4, help="Num of threads to accelerate")
    parser.add_argument('--thresh', '-thresh', type=float, default=10, help="Threshold of input map")
    parser.add_argument('--angle_step', '-angle_step', type=float, default=18.0, help="Angle step for FFT sampling")
    parser.add_argument('--fgrid', '-fgrid', type=float, default=2.0, help="FTMatch grid apix")
    parser.add_argument('--sgrid', '-sgrid', type=float, default=1.0, help="Simplex grid apix")
    args = parser.parse_args()
    main(args)
