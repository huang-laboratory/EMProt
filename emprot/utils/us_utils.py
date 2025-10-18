import os
import sys
import time
import tempfile
import argparse
import subprocess

from emprot.utils.misc_utils import pjoin

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


