import os
import sys
import tempfile
import subprocess
import contextlib
import numpy as np

from emprot.io.fileio import getlines

def read_TMalign_transform(filename):
    lines = getlines(filename)
    R = []
    t = []
    for line in lines:
        if line[:1] in ["0", "1", "2"]:
            x0, x1, x2, x3 = [float(x) for x in line.strip().split()[1:5]]
            t.append(x0)
            R.append([x1, x2, x3])
    R = np.asarray(R, dtype=np.float32) # (3, 3)
    t = np.asarray(t, dtype=np.float32) # (3,  )
    return R, t

def read_TMalign_stdout(output):
    # output is a string of TMalign's stdout
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
    
    Total CPU time is  0.00 seconds
    """
    lines = output.split("\n")
    return lines

def run_TMalign(a, b, lib_dir='./', temp_dir=None, d=3.0, verbose=True, description=""):
    # align a onto b using TMalign
    with tempfile.TemporaryDirectory() as __temp_dir:
        # make a temp dir
        if temp_dir is None:
            temp_dir = __temp_dir

        cmd = lib_dir + "/bin/TMalign {} {} -d {:.2f} -m {}".format(
            a, b, d, temp_dir + "/" + description + "_MTX.txt"
        )
        if verbose:
            print(f"# Running command {cmd}")
        # run
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        # parse result
        if result.returncode == 0:
            tm_output = result.stdout.decode('utf-8')
            tm_output = read_TMalign_stdout(tm_output)
            R, t = read_TMalign_transform(temp_dir + "/" + description + "_MTX.txt")
        else:
            tm_output, R, t = None, None, None

    return tm_output, R, t

