import os
import time
import numpy as np

from misc_utils import abspath, pjoin
from pdbio import read_pdb, chains_atom_pos_to_pdb

from cpsolver import solve

def main(args):
    """
        This script is only to rearrange the chain names (and the relative order) of the input fragments.
        In such case, the Coverage and Seq. Match (order-independent) should be unchanged.
        BUT, the TM-score is changeable since it's order-dependent.
    """
    ts = time.time()
    
    # read args
    fpdb = args.pdb
    fseq = args.seq
    fout = args.output
    fout = abspath(output)
    lib_dir = args.lib
    lib_dir = abspath(lib_dir)

    # read pdb
    atom_pos, atom_mask, res_type, res_idx, chain_idx = read_pdb(fpdb, keep_valid=False)
    n_chain = chain_idx.max() + 1
    print("# Read {} frags".format(n_chain))

    # align frag seq to given seq
    pass

    te = time.time()
    print("# Time consuming {:.4f}".format(te-ts))

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", "-p", type=str, help="Input fragments")
    parser.add_argument("--seq", "-s", type=str, help="Input sequences")
    parser.add_argument("--output", "-o", type=str, help="Output directory")
    parser.add_argument("--lib", "-l", help="Lib directory", default=script_dir)
    args = parser.parse_args()
    main(args)
