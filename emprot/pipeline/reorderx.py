import os
import re
import time
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from typing import List
from dataclasses import dataclass
from itertools import combinations

from ortools.sat.python import cp_model

from emprot.io.seqio import (
    read_fasta, 
    nwalign_fast, 
    find_first_of, find_last_of, 
)
from emprot.io.pdbio import read_pdb, chains_atom_pos_to_pdb
from emprot.utils.misc_utils import pjoin, abspath
from emprot.utils.residue_constants import index_to_restype_1

@dataclass
class Match:
    idx: int 
    # Alignment result
    seq_idx: int
    seq_identity: float
    seq_start: int
    seq_end: int
    seq_align: str
    seq_A: str
    seq_B: str

def main(args):
    t0 = time.time()

    # read fragments
    fpdb = args.pdb
    atom_pos, atom_mask, res_type, res_idx, chain_idx, atom_bfactor = read_pdb(fpdb, ignore_hetatm=True, keep_valid=False, return_bfactor=True)

    # split chains
    n_chain = chain_idx.max() + 1
    chains_atom_pos = []
    chains_atom_mask = []
    chains_res_type = []
    chains_res_idx = []
    chains_chain_idx = []
    chains_atom_bfactor = []
    for i_chain in range(n_chain):
        mask = chain_idx == i_chain
        chains_atom_pos.append(atom_pos[mask])
        chains_atom_mask.append(atom_mask[mask])
        chains_res_type.append(res_type[mask])
        chains_res_idx.append(res_idx[mask])
        chains_chain_idx.append(chain_idx[mask])
        chains_atom_bfactor.append(atom_bfactor[mask])
    print("# Read {} frags from {}".format(n_chain, fpdb))

    # read sequences
    fseq = args.seq
    seqs = read_fasta(fseq)
    n_seq = len(seqs)
    assert n_seq >= 1, "# At least input one sequence"
    print("# Read {} seqs from {}".format(n_seq, fseq))

    # step 1: construct alignment
    matches = {}
    min_res = 10
    identity_cutoff = 0.80
    cov_cutoff = 0.80
    for i in range(n_chain):
        # ignore too short fragment
        if len(chains_atom_pos[i]) < min_res:
            continue

        # get sequence from structure
        seqA = "".join([index_to_restype_1[x] for x in chains_res_type[i]])

        # save at most n_seq aligns
        for k in range(n_seq):
            new_seqA, align, new_seqB, identityA, covA, _, _ = nwalign_fast(
                seqA, 
                seqs[k], 
                lib_dir=os.path.join(abspath(os.path.dirname(__file__)), ".."), 
                verbose=True,
                fmt=True, 
            )
            if identityA > identity_cutoff and covA > cov_cutoff:

                start = find_first_of(align, ":")
                if start == -1:
                    start = len(align)

                end = find_last_of(align, ":")
                if end == -1:
                    end = len(align)
                else:
                    end += 1

                #print(new_seqA)
                #print(align)
                #print(new_seqB)
                #print(start, end)

                matches[i, k] = Match(i, k, identityA, start, end, align, new_seqA, new_seqB)

    # step 2: construct conflict graph
    # principles:
    # 1. same positions in sequences can not be matched > 1 times
    # 2. one fragmenet can only aligned to
    print("# Total {} matches".format(len(matches)))

    # solving
    print("# Solving assignment")
    model = cp_model.CpModel()
    x = {}
    for i in range(n_chain):
        for k in range(n_seq):
            if (i, k) in matches:
                x[i, k] = model.NewBoolVar(f"x_{i}_{k}")

    # add restraint 1
    # one fragment must be assigned to `one` chain
    for i in range(n_chain):
        ks = [k for k in range(n_seq) if (i, k) in matches]
        model.Add(sum(x[i, k] for k in ks) <= 1)

    # add restraint 2
    def get_overlap(interval1, interval2):
        a1, a2 = sorted(interval1)
        b1, b2 = sorted(interval2)
        overlap_start = max(a1, b1)
        overlap_end = min(a2, b2)
        return max(0, overlap_end - overlap_start)

    def find_colon_idx(A):
        colon_indices = []
        for index, char in enumerate(A):
            if char == ':':
                colon_indices.append(index)
        return colon_indices


    def find_common_colon_idx(A, B):
        common_indices = []
        min_length = min(len(A), len(B))

        for i in range(min_length):
            if A[i] == ':' and B[i] == ':':
                common_indices.append(i)
        return common_indices


    # for hetero-mer groups overlap_ratio can be smaller
    # for homo-mer groups overlap ratio can be bigger

    overlap_cutoff = 0.40
    for match_i, match_k in combinations(matches.values(), 2):
        if match_i.seq_idx == match_k.seq_idx:
            # if match is inconsistent like the following
            #   ::::::::     ::::::::::::::::     :::::   #
            #   s                                     e   #
            #          :::::::              :::::::       #
            #          s                          e       #
            # we should use the actual matched idxs

            n_i = len(find_colon_idx(match_i.seq_align))
            n_k = len(find_colon_idx(match_k.seq_align))

            if not (n_i > 0):
                n_i = 1e-6

            if not (n_k > 0):
                n_k = 1e-6

            n_common = len(find_common_colon_idx(match_i.seq_align, match_k.seq_align))
          
            # 0427 max-to-min 
            overlap = min(
                n_common / n_i, 
                n_common / n_k, 
            )
          
            print("=" * 100) 
            print(match_i.seq_A)
            print(match_k.seq_A)
            print(overlap, n_i, n_k, n_common)
            print("=" * 100) 

            # if two fragment matches the "same" sequence, only one is allowed
            if overlap > overlap_cutoff:
                model.Add(
                    (x[match_i.idx, match_i.seq_idx] + x[match_k.idx, match_k.seq_idx]) <= 1
                ).OnlyEnforceIf(x[match_i.idx, match_i.seq_idx])

                model.Add(
                    (x[match_i.idx, match_i.seq_idx] + x[match_k.idx, match_k.seq_idx]) <= 1
                ).OnlyEnforceIf(x[match_k.idx, match_k.seq_idx])

    chains_centers = []
    for i in range(n_chain):
        chains_centers.append( chains_atom_pos[i][..., 1, :].mean(0) )
    chains_centers = np.asarray(chains_centers, dtype=np.float32)
    distance = np.linalg.norm(chains_centers[:, None, :] - chains_centers[None, :, :], axis=-1)


    # link distance
    chains_c_pos = []
    chains_n_pos = []
    for i in range(n_chain):
        chains_c_pos.append(chains_atom_pos[i][-1, 2, :])
        chains_n_pos.append(chains_atom_pos[i][ 0, 0, :])

    chains_c_pos = np.asarray(chains_c_pos)
    chains_n_pos = np.asarray(chains_n_pos)

    link_distance = np.linalg.norm(
        chains_c_pos[:, None, :] - chains_n_pos[None, :, :],
        axis=-1,
    )

    #print(distance.shape)
    #print(link_distance.shape)
    #exit()


    # add score
    y = dict()
    scores = []

    for match_i in matches.values():
        scores.append(10.0 * x[match_i.idx, match_i.seq_idx])

    for match_i, match_j in combinations(matches.values(), 2):
        for k in range(n_seq):
            if not (match_i.seq_idx == match_j.seq_idx == k):
                continue

            i, j = match_i.idx, match_j.idx
            y[i, j, k] = model.NewBoolVar(f"y_{i}_{j}_{k}")

            model.Add(y[i, j, k] <= x[i, k])
            model.Add(y[i, j, k] <= x[j, k])
            model.Add(y[i, j, k] >= x[i, k] + x[j, k] - 1)

            d_center = distance[i, j]
            d_center = np.clip(d_center, a_min=10.0, a_max=100.0)


            d_link = link_distance[i, j]
            d_link = np.clip(d_link, a_min=3.0, a_max=60)
            
            d_combined = 1.0 / (1.0/d_center + 1.0/d_link )

            d = d_center
            #d = d_combined

            #print(d_center, d_link, d_combined, d)

            scores.append( (100.0 / (d + 1e-3)) * y[i, j, k])

    model.Maximize(sum(scores))



    # solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = args.time_limit
    solver.parameters.num_search_workers = args.num_workers
    solver.parameters.log_search_progress = False
    solution_printer = cp_model.ObjectiveSolutionPrinter()
    status = solver.SolveWithSolutionCallback(model, solution_printer)

    # get result
    assigned_chains = {}
    for k in range(n_seq):
        assigned_chains[k] = []

    if (status == cp_model.OPTIMAL) or \
        (status == cp_model.FEASIBLE):
        for k in range(n_seq):
            for i in range(n_chain):
                if (i, k) in matches:
                    if solver.BooleanValue(x[i, k]):
                        assigned_chains[k].append(i)
    else:
        raise Exception("No solution found")



    # convert results to final assigned nodes
    # 1. gather for each sequence
    # 2. sort gathered fragments (by start) for each sequence
    new_chains_atom_pos = []
    new_chains_atom_mask = []
    new_chains_res_type = []
    new_chains_res_idx = []
    new_chains_atom_bfactor = []

    delta = 10
    assigned_idxs = set()
    for k, chains in assigned_chains.items():
        # when chains is empty
        if len(chains) == 0:
            continue

        # for each chain
        # split to continuous fragment for each chain
        frags_aligned_pos = []
        frags_atom_pos = []
        frags_atom_mask = []
        frags_res_type = []
        frags_res_idx = []
        frags_atom_bfactor = []

        for kk, chain in enumerate(chains):
            assigned_idxs.add(chain)

            print("#" * 80)
            print("# Split chain {}".format(kk))
            # seq_A is smoethiong like  "----X-XX------XXXXXXXXX------XXXXXXXXXXX----"
            frag_matches = list(re.finditer(r'[A-Z]+', matches[chain, k].seq_A))

            frag_start_idx = 0
            for m in frag_matches:
                fragment = m.group()
                start = m.start()
                end = m.end()

                # keep larger fragment
                if end - start >= 2:
                    frag_idx = list(range(frag_start_idx, frag_start_idx + len(fragment)))
                    print(f"Fragment: '{fragment}' at position [{start}, {end})")

                    # append to aligned positions
                    frags_aligned_pos.append(start)

                    # append to frags
                    frags_atom_pos.append( chains_atom_pos[chain][frag_idx] )
                    frags_atom_mask.append( chains_atom_mask[chain][frag_idx] )
                    frags_res_type.append( chains_res_type[chain][frag_idx] )
                    frags_res_idx.append( chains_res_idx[chain][frag_idx] )
                    frags_atom_bfactor.append( chains_atom_bfactor[chain][frag_idx] )

                frag_start_idx += len(fragment)

        print("# Sequence {} has {} frags".format(k, len(frags_atom_pos)))

        # reorder frag and renumber res idx
        sorted_order = np.argsort(frags_aligned_pos)

        frags_atom_pos = [frags_atom_pos[x] for x in sorted_order]
        frags_atom_mask = [frags_atom_mask[x] for x in sorted_order]
        frags_atom_bfactor = [frags_atom_bfactor[x] for x in sorted_order]
        frags_res_type = [frags_res_type[x] for x in sorted_order]
        frags_res_idx = [frags_res_idx[x] for x in sorted_order]
        frags_aligned_pos = [frags_aligned_pos[x] for x in sorted_order]

        # renumber res idx
        delta = 10
        last_res_idx = 0
        new_frags_res_idx = []
        for kk, atom_pos in enumerate(frags_atom_pos):
            new_res_idx = frags_res_idx[kk] - frags_res_idx[kk][0]
            #print(frags_res_idx[kk], new_res_idx)

            new_res_idx += last_res_idx

            new_frags_res_idx.append(new_res_idx)

            last_res_idx = new_res_idx[-1]
            last_res_idx = last_res_idx + delta

            #print(last_res_idx)

        frags_res_idx = new_frags_res_idx


        # concate all frags for current sequence
        new_atom_pos = np.concatenate(frags_atom_pos, axis=0)
        new_atom_mask = np.concatenate(frags_atom_mask, axis=0)
        new_res_type = np.concatenate(frags_res_type, axis=0)
        new_res_idx= np.concatenate(frags_res_idx, axis=0)
        new_atom_bfactor = np.concatenate(frags_atom_bfactor, axis=0)


        # append to chains
        new_chains_atom_pos.append(new_atom_pos)
        new_chains_atom_mask.append(new_atom_mask)
        new_chains_res_type.append(new_res_type)
        new_chains_res_idx.append(new_res_idx)
        new_chains_atom_bfactor.append(new_atom_bfactor)

    print("# {} fragments are assigned to {} chains, {} left".format(len(assigned_idxs), len(assigned_chains), n_chain - len(assigned_idxs)))

    # 3. add missing fragments
    for i in range(n_chain):
        if not (i in assigned_idxs):
            new_chains_atom_pos.append( chains_atom_pos[i] )
            new_chains_atom_mask.append( chains_atom_mask[i] )
            new_chains_res_type.append( chains_res_type[i] )
            new_chains_res_idx.append( chains_res_idx[i] )
            new_chains_atom_bfactor.append( chains_atom_bfactor[i] )

    # yield final result
    os.makedirs(args.out, exist_ok=True)
    fpdbout = pjoin(args.out, "reorder.cif")
  
    chains_atom_pos_to_pdb(
        fpdbout, 
        chains_atom_pos=new_chains_atom_pos,
        chains_atom_mask=new_chains_atom_mask,
        chains_res_types=new_chains_res_type,
        chains_res_idxs=new_chains_res_idx,
        chains_chain_idxs=None,
        chains_bfactors=new_chains_atom_bfactor,
    )

    t1 = time.time()
    print("# Time consumption {:.4f}".format(t1 - t0))
    print("# Output reordered chains to {}".format(fpdbout))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", "-p", required=True, help="Input structure")
    parser.add_argument("--seq", "-s", required=True, help="Input sequence")
    parser.add_argument("--out", "-o", default=".", help="Output dir")
    # solver controls
    parser.add_argument("--time_limit", default=600, help="Time limit for solving")
    parser.add_argument("--num_workers", default=2, help="Num of CPUs to use")
    parser.add_argument("--log_search_progress", default=True, help="Whether to log search process")
    args = parser.parse_args()
    main(args)
