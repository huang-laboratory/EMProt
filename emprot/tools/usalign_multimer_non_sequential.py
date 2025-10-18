import os
import sys
import time
import tempfile
import argparse
import subprocess

from emprot.utils.misc_utils import pjoin
from emprot.io.pdbio import read_pdb, chains_atom_pos_to_pdb


def run_USalign_ns(a, b, lib_dir='./', verbose=True, mode="semi"):
    # align a onto b using USalign non-sequenetial mode
    # set ns mode
    if mode == "full":
        i_mode = 5
    else:
        i_mode = 6

    cmd = lib_dir + "/bin/USalign {} {} -mol prot -mm {} -fast".format(
        a, b, i_mode, 
    )
    if verbose:
        print(f"# Running command {cmd}")
    # run
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    # parse result
    if result.returncode == 0:
        output = result.stdout.decode('utf-8')
        output = read_USalign_ns_stdout(output)
    else:
        output = None

    return output


def read_USalign_ns_stdout(output):
    # output is a string like
    """
    Name of Chain_1: AF.C.pdb (to be superimposed onto Chain_2)
    Name of Chain_2: denovo_chain_4.pdb
    Length of Chain_1: 55 residues
    Length of Chain_2: 44 residues

    Aligned length= 41, RMSD=   1.76, Seq_ID=n_identical/n_aligned= 0.902
    TM-score= 0.59349 (if normalized by length of Chain_1, i.e., LN=55, d0=2.44)
    TM-score= 0.69650 (if normalized by length of Chain_2, i.e., LN=44, d0=2.01)
    (You should use TM-score normalized by length of the reference structure)

    (":" denotes residue pairs of d <  5.0 Angstrom, "." denotes other aligned residues)
    -LCGGELVDTLQFVCGDRGFYFSR--PASRVSRRSRGIVEECCFRSCDLALLETYCAT
     :: :::::::::::::::::     ..        ::::::::::::::::::::
    LCG-GELVDTLQFVCGDRGFY---FSRP--------GIVEECCFRSCDLALLETYC--

    ############### ############### #########
    #Aligned atom 1 Aligned atom 2  Distance#
     CA  SER A 358   CA  SER A1534      1.227
     CA  ASN A 359   CA  ASN A1535      1.195
     CA  LEU A 360   CA  LEU A1536      1.144
     CA  TYR A 361   CA  TYR A1537      1.096
     CA  LEU A 362   CA  LEU A1538      1.061
     CA  ILE A 363   CA  ILE A1539      1.010
     CA  TYR A 364   CA  TYR A1540      0.962
     CA  LYS A 365   CA  LYS A1541      0.922
     CA  LYS A 366   CA  LYS A1542      0.890
     CA  GLU A 367   CA  GLU A1543      0.827
     CA  THR A 368   CA  THR A1544      0.784
     CA  GLY A 369   CA  GLY A1545      0.750
    ############### ############### #########

    Total CPU time is  0.00 seconds
    """
    lines = output.split("\n")
    return lines


def main(args):
    f_tgt = args.target
    f_qry = args.query

    # read structure
    tgt_atom_pos, tgt_atom_mask, tgt_res_type, _, _ = read_pdb(f_tgt, keep_valid=False)
    qry_atom_pos, qry_atom_mask, qry_res_type, _, _ = read_pdb(f_qry, keep_valid=False)

    print("# Read {} residues from target structure {}".format(len(tgt_atom_pos), f_tgt))
    print("# Read {} residues from query  structure {}".format(len(qry_atom_pos), f_qry))

    if args.usalign is None:
        usalign = "USalign"
    else:
        usalign = args.usalign

    # non sequential mode
    if args.non_sequential_mode == "full":
        ns_mode = 5
    elif args.non_sequential_mode == "semi":
        ns_mode = 6
    else:
        raise Exception("# Wrong mode, can only be 'full' or 'semi'")

    # write into one chain
    script_dir = os.path.abspath(os.path.dirname(__file__))
    lib_dir = pjoin(script_dir, "..")

    with tempfile.TemporaryDirectory() as temp_dir:
        f_tgt_out = pjoin(temp_dir, "target.cif")
        chains_atom_pos_to_pdb(
            f_tgt_out, 
            [tgt_atom_pos],
            [tgt_atom_mask],
            [tgt_res_type], 
        )
        
        f_qry_out = pjoin(temp_dir, "query.cif")
        chains_atom_pos_to_pdb(
            f_qry_out, 
            [qry_atom_pos],
            [qry_atom_mask],
            [qry_res_type], 
        )

        print("# Write target residues into one chain at {}".format(f_tgt_out))
        print("# Write query  residues into one chain at {}".format(f_qry_out))
        print("# Running USalign with non-sequential mode: {}".format(args.non_sequential_mode))

        result = run_USalign_ns(f_tgt_out, f_qry_out, lib_dir=lib_dir)

        for line in result:
            print(line)

        """
        other_options = args.usalign_options
        cmd = "{} -mol prot -mm {} {} {} {}".format(usalign, ns_mode, f_tgt_out, f_qry_out, other_options)
        print("# {}".format(cmd))

        result = subprocess.run(
            cmd, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
        )

        # parse result
        if result.returncode == 0:
            print("# Run USalign succesfully")
            output = result.stdout.decode('utf-8')
        else:
            print("# Run USalign failed")
            output = result.stderr.decode('utf-8')

        print(output)
        """
    
def add_args(parser):
    script_dir = os.path.abspath(os.path.dirname(__file__))
    usalign = pjoin(script_dir, "..", "bin", "USalign")
    parser.add_argument("--target", "-t", required=True, help="Target structure")
    parser.add_argument("--query", "-q", required=True, help="Query structure")
    parser.add_argument("--usalign", "-u", help="Path to USalign", default=usalign)
    parser.add_argument("--non_sequential_mode", default="full", help="Non-sequential mode for USalign, 'full' or 'semi'")
    # USalign
    parser.add_argument("--usalign_options", type=str, default="", help="USalign options")
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = add_args(parser).parse_args()
    main(args)

