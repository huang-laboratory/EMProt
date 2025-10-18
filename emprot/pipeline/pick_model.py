import os
import time
import shutil

from emprot.pipeline.eval_cc import compute_cc
from emprot.io.pdbio import read_pdb
from emprot.io.seqio import read_fasta, nwalign_fast

def main(args):
    t0 = time.time()

    # read sequences
    fseq = args.seq
    seqs = read_fasta(fseq)
    n_seq = len(seqs)
    assert n_seq >= 1, "# At least input one sequence"
    print("# Read {} seqs from {}".format(n_seq, fseq))

    # sort seqs
    seqs.sort(key=lambda x: len(x), reverse=True)
    for seq in seqs:
        print("Found sequence of length {}".format(len(seq)))
        print(seq)


    # grouping sequences
    nwalign_temp_dir = os.path.join(args.output, "nwalign_pmodel")
    os.makedirs(nwalign_temp_dir, exist_ok=True)
    lib_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..")

    seq_groups = dict()
    homo_seq_cutoff_A = 0.80 # for longer sequence
    homo_seq_cutoff_B = 0.80 # for shorter sequence
    seq_is_used = [False] * len(seqs)
    for i in range(len(seqs)):
        if seq_is_used[i]:
            continue

        seq_groups[i] = [i]
        seq_is_used[i] = True

        for k in range(i + 1, len(seqs)):
            if seq_is_used[k]:
                continue

            seqA = seqs[i]
            seqB = seqs[k]

            # fix a bug in NWalign
            if seqA[0] == seqB[0]:
                seqAx = "".join(seqA[1:])
                seqBx = "".join(seqB[1:])
            else:
                seqAx = seqA
                seqBx = seqB

            sA, align, sB, idA, covA, idB, covB = nwalign_fast(
                seqAx,
                seqBx,
                temp_dir=nwalign_temp_dir,
                lib_dir=lib_dir,
                verbose=True,
                fmt=False,
            )

            if idA > homo_seq_cutoff_A and idB > homo_seq_cutoff_B:
                seq_groups[i].append(k)
                seq_is_used[k] = True

    for i, (k, v) in enumerate(seq_groups.items()):
        print("# Group {} has {} seqs ".format(i, len(v)))

    homo_n_mer = False
    if len(seqs) > 1 and len(seq_groups) == 1 and len(seq_groups[0]) == len(seqs):
        homo_n_mer = True
    print("# Target omo-n-mer is {}".format(homo_n_mer))

    # resolution
    print("# Input resolution is {:.2f}".format(args.resolution))

    #exit()

    # if homo-n-mer, low resolution (> 3.5), fitted models have more residues
    # and num of assemble chains != num of sequences


    if os.path.exists(args.dock):
        atom_pos, _, _, _, chain_idx = read_pdb(args.assemble, keep_valid=True)
        n_res_asmb = len(atom_pos)
        n_chain_asmb = chain_idx.max() + 1

        # compute cc for structures
        print("# Computing CC for structures")
        result_asmb = compute_cc(
            args.assemble,
            args.map,
            args.resolution,
            args.bfactor,
        )

        print("# ASMB result")
        print("# ASMB n_chain   : {}".format(n_chain_asmb))
        print("# ASMB n_res     : {}".format(n_res_asmb))
        print("# ASMB CC_mask   : {:.4f}".format(result_asmb.cc_mask))
        print("# ASMB CC_volume : {:.4f}".format(result_asmb.cc_volume))
        print("# ASMB CC_peaks  : {:.4f}".format(result_asmb.cc_peaks))
        print("# ASMB CC_box    : {:.4f}".format(result_asmb.cc_box))


        result_dock = compute_cc(
            args.dock,
            args.map,
            args.resolution,
            args.bfactor,
        )

        # get plddt for dock result
        atom_pos, atom_mask, res_type, res_idx, chain_idx, bfactor = read_pdb(args.dock, keep_valid=True, return_bfactor=True)
        n_res_dock = len(atom_pos)
        n_chain_dock = chain_idx.max() + 1
        mean_plddt = bfactor[..., 1].mean()

        print("# DOCK result")
        print("# DOCK n_chain   : {}".format(n_chain_dock))
        print("# DOCK n_res     : {}".format(n_res_dock))
        print("# DOCK CC_mask   : {:.4f}".format(result_dock.cc_mask))
        print("# DOCK CC_volume : {:.4f}".format(result_dock.cc_volume))
        print("# DOCK CC_peaks  : {:.4f}".format(result_dock.cc_peaks))
        print("# DOCK CC_box    : {:.4f}".format(result_dock.cc_box))
        print("# DOCK PLDDT     : {:.4f}".format(mean_plddt))

        # For homo-n-mer
        ra = abs(n_res_asmb - n_res_dock) / (n_res_asmb + 1e-6)
        rb = abs(n_res_asmb - n_res_dock) / (n_res_dock + 1e-6)
        print("# ra = {:.4f} rb = {:.4f}".format(ra, rb))

        # calculate potentially unmodeled ratio
        potential_unmodeled_ratio = n_res_asmb / (n_res_dock + 1e-6)
    else:
        print("# No DOCK model found, will not show CCs")
        result_dock = None
        ra = 0.0
        rb = 0.0
        n_chain_dock = 0

    # by default, we use the assembled model
    model = args.assemble

    if homo_n_mer:
        print("# Homo target")
        if args.resolution > 3.50 and (ra > 0.20 or rb > 0.20) and not (len(seqs) == n_chain_dock == n_chain_asmb):
            model = args.dock

    if os.path.exists(args.dock) and result_dock is not None and potential_unmodeled_ratio < 2.0:
        # pick model
        # calculate the relative ccbox
        a = abs( (result_asmb.cc_box - result_dock.cc_box) / result_asmb.cc_box )
        b = abs( (result_asmb.cc_box - result_dock.cc_box) / result_dock.cc_box )
        print("# a = {:.4f} b = {:.4f}".format(a, b))

        # if dock model has better cc box
        if result_dock.cc_box > result_asmb.cc_box:
            model = args.dock

        # if assemble model and dock model is similar in CC box
        if a < 0.10 or b < 0.10:
            model = args.dock

    # output final
    os.makedirs(args.output, exist_ok=True)
    fo = os.path.join(args.output, "pmodel.cif")
    print("# Copy {} to {}".format(model, fo))
    dest = shutil.copyfile(model, fo) 

    t1 = time.time()
    print("# Time consumption {:.4f}".format(t1 - t0))

def add_args(parser):
    parser.add_argument("--assemble", required=True, help="Assembled and reordered structure")
    parser.add_argument("--dock", required=True, help="Docked structure")
    parser.add_argument("--map", "-m", required=True, help="Input map")
    parser.add_argument("--seq", "-s", required=True, help="Input sequences")
    parser.add_argument("--resolution", "-r", required=True, type=float, help="Map resolution")
    parser.add_argument("--output", "-out", "-o", default='./', help="Output directory")
    parser.add_argument("--bfactor", default=0.0, help="Input b-factor")
    return parser

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args = add_args(parser).parse_args()
    main(args)
