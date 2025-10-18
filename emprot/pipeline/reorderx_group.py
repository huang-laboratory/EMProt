import os
import re
import time
import pickle
import tempfile
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
from emprot.utils.fix_idx_break import fix_idx_break
from emprot.utils.vrp import fragment_tracing
from emprot.utils.us_utils import run_USalign_ns

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

    # makr temp dir
    usalign_temp_dir = pjoin(args.out, "usalign")
    nwalign_temp_dir = pjoin(args.out, "nwalign")
    os.makedirs(usalign_temp_dir, exist_ok=True)
    os.makedirs(nwalign_temp_dir, exist_ok=True)

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

    lib_dir = os.path.join(abspath(os.path.dirname(__file__)), "..")

    ##############################################
    # split seq into groups of "hetero" and "homo"
    ##############################################

    seq_groups = dict()
    homo_seq_cutoff_A = 0.40 # for longer sequence
    homo_seq_cutoff_B = 0.80 # for shorter sequence
    homo_seq_cutoff_B_high = 0.95
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


            # Split to fragment according to alignment

            #n_align = sum([1 if x == ":" else 0 for x in align])
            #r_align_A = n_align / len(seqs[i])
            #r_align_B = n_align / len(seqs[k])

            #print(sA)
            #print(align)
            #print(sB)
            #print(idA, covA)
            #print(idB, covB)

            if (idA > homo_seq_cutoff_A and idB > homo_seq_cutoff_B) or \
                idB > homo_seq_cutoff_B_high:
                seq_groups[i].append(k)
                seq_is_used[k] = True

    for i, (k, v) in enumerate(seq_groups.items()):
        print("# Group {} has {} seqs ".format(i, len(v)))


    #########################################################
    # for each sequence group find the corresponding template
    #########################################################
    use_chain_guidance = True
    chain_groups = dict()
    if args.chain is None:
        print("# No chain guidance")
        use_chain_guidance = False
    else:
        fchain_list = args.chain
        print("# Using chain guidance")
        print("# Found {} chain templates".format(len(fchain_list)))
        chain_is_assigned = [False] * len(fchain_list)

        for k, k_group in seq_groups.items():
            seqA = seqs[k]

            best_match_ic = None
            best_match_fchain = None
            best_idA = -1.0
            best_covA = -1.0
            best_idB = -1.0
            best_covB = -1.0


            for ic, fchain in enumerate(fchain_list):
                if chain_is_assigned[ic]:
                    continue

                # read structure and sequence
                (
                    atom_pos, 
                    atom_mask, 
                    res_type, 
                    res_idx, 
                    chain_idx, 
                    bfactor, 
                ) = read_pdb(fchain, keep_valid=False, return_bfactor=True)

                seqB = "".join([index_to_restype_1[x] for x in res_type])
                new_seqA, align, new_seqB, idA, covA, idB, covB = nwalign_fast(
                    seqA, 
                    seqB,
                    temp_dir=nwalign_temp_dir,
                    lib_dir=lib_dir,
                    verbose=True,
                    fmt=True,
                )

                #print(idA, covA, idB, covB)
                #print(new_seqA)
                #print(align)
                #print(new_seqB)

                if idA > best_idA and covA > best_covA and idB > best_idB and covB > best_covB:
                    best_match_ic = ic
                    best_match_fchain = fchain
                    best_idA = idA
                    best_idB = idB
                    best_covA = covA
                    best_covB = covB

            if best_match_ic is None:
                print("# No chain match to seq {}".format(k))
                chain_groups[k] = None
            else:
                print("# Chain {} match to seq {}".format(best_match_ic, k))
                chain_is_assigned[best_match_ic] = True
                chain_groups[k] = (best_match_fchain, atom_pos, atom_mask, res_type, res_idx, bfactor)

    #exit()


    # read chains
    fpdb = args.pdb
    atom_pos, atom_mask, res_type, res_idx, chain_idx, atom_bfactor = read_pdb(fpdb, ignore_hetatm=True, keep_valid=False, return_bfactor=True)

    
    n_res = len(atom_pos)
    print("# Input structure has {} residues".format(n_res))
    if n_res > 5000:
        use_chain_guidance = False
    print("# Use template chain guidance {}".format(use_chain_guidance))


    # split chains
    n_chain = chain_idx.max() + 1
    chains_atom_pos = []
    chains_atom_mask = []
    chains_res_type = []
    chains_res_idx = []
    chains_atom_bfactor = []
    for i_chain in range(n_chain):
        mask = chain_idx == i_chain
        chains_atom_pos.append(atom_pos[mask])
        chains_atom_mask.append(atom_mask[mask])
        chains_res_type.append(res_type[mask])
        chains_res_idx.append(res_idx[mask])
        chains_atom_bfactor.append(atom_bfactor[mask])
    print("# Read {} chains from {}".format(n_chain, fpdb))


    # split each chain into fragments base on inter Ca distance
    frags_atom_pos = []
    frags_atom_mask = []
    frags_res_type = []
    frags_res_idx = []
    frags_atom_bfactor = []
    frags_original_chain_idx = []
    d_break = 8.0
    for i in range(len(chains_atom_pos)):
        last_res = 0
        n = 0
        for k in range(1, len(chains_atom_pos[i])):
            d = np.linalg.norm(chains_atom_pos[i][k - 1][1] - chains_atom_pos[i][k][1])
            if d > d_break or k == len(chains_atom_pos[i]) - 1:
                frag_idx = list(range(last_res, k))

                frags_atom_pos.append( chains_atom_pos[i][frag_idx] )
                frags_atom_mask.append( chains_atom_mask[i][frag_idx] )
                frags_res_type.append( chains_res_type[i][frag_idx] )
                frags_res_idx.append( chains_res_idx[i][frag_idx] )
                frags_atom_bfactor.append( chains_atom_bfactor[i][frag_idx] )

                # Save the original chain index for current frag
                frags_original_chain_idx.append( i )

                n += 1
                last_res = k

        print("# Split chain {} into {} frags".format(i, n))

            
    print("###################################")
    print("# Split {} chains to {} fragments #".format(n_chain, len(frags_atom_pos)))
    print("###################################")



    ######################################################
    # Step 1: Divide fragments into hetero and homo groups
    ######################################################

    ###################################################################
    # Step 1.1: For hetero group, reorder with the matched starting pos
    ###################################################################

    print("# Hetero groups")
    hetero_seq_idxs = []
    for k, v in seq_groups.items():
        if len(v) == 1:
            hetero_seq_idxs.append(k)


    ###################################################################
    # Step 1.2: For each target seq, find the match frags using USalign
    ###################################################################

    matches = {}
    min_res = 10
    id_cutoff = 0.80
    cov_cutoff = 0.80
    r_align_cutoff = 0.60

    if use_chain_guidance:
        frags_is_assigned = [False] * len(frags_atom_pos)
        seq_frags = dict()
        for k in hetero_seq_idxs:
            seq_frags[k] = list()

        plddt_cutoff = 85.00
        r_frag_cutoff = 0.60
        seq_frags_chain_align = dict()

        for k in hetero_seq_idxs:
            seq_frags_chain_align[k] = list()

        for k in hetero_seq_idxs:
            print("# Process seq {}".format(k))
            if chain_groups[k] is None:
                print("# No chain for seq {}".format(k))
                continue

            (chain_filename, _, _, _, _, chain_bfactor) = chain_groups[k]

            mean_plddt = chain_bfactor[..., 1].mean()

            #if not (100.00 > mean_plddt > plddt_cutoff):
            #    print("# Mean plddt is {:.4f}, skip this chain".format(mean_plddt))
            #    continue

            # Write all frags into one chain
            sel_frag_idx = []
            for i in range(len(frags_atom_pos)):
                if frags_is_assigned[i]:
                    continue
                sel_frag_idx.append(i)

            #print(sel_frag_idx)
            if len(sel_frag_idx) == 0:
                print("# No more frags can be aligned")
                continue

            fo = pjoin(usalign_temp_dir, f"frags_for_seq_{k}.cif")
            chains_atom_pos_to_pdb(
                fo,
                [np.concatenate([frags_atom_pos[x] for x in sel_frag_idx], axis=0)],
                [np.concatenate([frags_atom_mask[x] for x in sel_frag_idx], axis=0)],
                [np.concatenate([frags_res_type[x] for x in sel_frag_idx], axis=0)],
                chains_bfactors=[np.concatenate([frags_atom_bfactor[x] for x in sel_frag_idx], axis=0)],
            )

            print("# Write frag for seq {} to {}".format(k, fo))
            print("# Align {} and {}".format(fo, chain_filename))

            result = run_USalign_ns(
                chain_filename, fo, lib_dir=lib_dir, 
            )

           
            # analyze result
            res_idx_to_frag_idx = np.concatenate(
                [ [x] * len(frags_atom_pos[x]) for x in sel_frag_idx ],
                axis=0,
            )
            #print(res_idx_to_frag_idx)
            #print(type(res_idx_to_frag_idx))

            tm = []
            aligned_idxs = []
            for line in result:
                fields = line.strip().split()
                if len(fields) == 0:
                    continue

                if fields[0] == "TM-score=":
                    tm.append(float(fields[1]))

                if fields[0] == "CA":
                    if fields[-2][0] == 'A':
                        aligned_idxs.append(int(fields[-2][1:]))
                    else:
                        aligned_idxs.append(int(fields[-2]))

            aligned_idxs = np.asarray(aligned_idxs, dtype=np.int32)

            #print(tm)
            #print(aligned_idxs)

            for i in range(len(frags_atom_pos)):
                n0 = (res_idx_to_frag_idx[aligned_idxs] == i).sum()
                n1 = len(frags_atom_pos[i])

                r = n0 / n1

                #print(i, r, r_frag_cutoff)

                if r > r_frag_cutoff:
                    frags_is_assigned[i] = True
                    seq_frags_chain_align[k].append(i)
                    print("# Frag {} assigned to seq {} with r = {:.4f}".format(i, k, r))

        ########################################################################
        # Step 1.2.1 For the assigned frags, find the matched seq. start and end
        ########################################################################
        r_sub_cutoff = 0.80
        frags_is_assigned = [False] * len(frags_atom_pos)
        for k, v in seq_frags_chain_align.items():
            print(k, v)
            seqB = seqs[k]

            for i in v:
                seqA = "".join([index_to_restype_1[x] for x in frags_res_type[i]])
                # fix a bug in NWalign
                if seqA[0] == seqB[0]:
                    seqAx = "".join(seqA[1:])
                    seqBx = "".join(seqB[1:])
                else:
                    seqAx = seqA
                    seqBx = seqB

                new_seqA, align, new_seqB, idA, covA, _, _ = nwalign_fast(
                    seqAx, 
                    seqBx, 
                    temp_dir=nwalign_temp_dir, 
                    lib_dir=lib_dir,
                    verbose=True,
                    fmt=True, 
                )

                #print("=" * 100)
                #print(i)
                #print(new_seqA)
                #print(align)
                #print(new_seqB)
                #print(idA, covA, id_cutoff, cov_cutoff)

                if idA > id_cutoff and covA > cov_cutoff:


                    # If the continuous fragment has wrong sequence assignment
                    # Find the largest colon substring
                    start = None
                    end = None
                    max_len = -1

                    cnts = 0
                    for match in re.finditer(r':+', align):
                        cnts += len(match.group())

                        #print(f"Match: '{match.group()}', Start: {match.start()}, End: {match.end()}")
                        if len(match.group()) > max_len:
                            max_len = len(match.group())
                            start = match.start()
                            end = match.end() 

                    if start is None:
                        start = len(align)

                    if end is None:
                        end = len(align)

                    r_sub = max_len / (cnts + 1e-3)
                    print("Sub ratio {:.4f}".format(r_sub))


                    # If a good frag
                    if r_sub > r_sub_cutoff:
                        print("Found frag ", i)
                        frags_is_assigned[i] = True
                        seq_frags[k].append((i, start, end))
                    else:
                        start = find_first_of(align, ":")
                        if start == -1:
                            start = len(align)

                        end = find_last_of(align, ":")
                        if end == -1:
                            end = len(align)
                        else:
                            end += 1

                        n_align = sum([1 if x == ":" else 0 for x in align])
                        r_align = n_align / (end - start)

                        #print(len(seqA), end - start)
                        #print(n_align, end - start, r_align, r_align_cutoff)
                        #print("=" * 100)

                        if r_align > r_align_cutoff:
                            print("Found frag ", i)
                            frags_is_assigned[i] = True
                            seq_frags[k].append((i, start, end))

        """
        # if some sequence can not be aligned
        for k, v in seq_frags_chain_align.items():
            if len(v) >= 1 and len(seq_frags[k]) == 0:
                for i in v:
                    frags_is_assigned[i] = True
                    seq_frags[k].append((i, 0, 1))
        """
    else:
        print("# No using chain guidance for hetero")
        frags_is_assigned = [False] * len(frags_atom_pos)
        seq_frags = dict()
        for k in hetero_seq_idxs:
            seq_frags[k] = list()

    #print(seq_frags)
    #print(frags_is_assigned)
    #exit()

    #########################################################################
    # Step 1.3: For each fragment find the best matched seq. using seq. align
    #########################################################################

    for i in range(len(frags_atom_pos)):
        if frags_is_assigned[i]:
            print("# Frag {} is already assigned using USalign".format(i))
            continue

        # ignore too short fragment
        if len(frags_atom_pos[i]) < min_res:
            continue

        # get sequence from structure
        seqA = "".join([index_to_restype_1[x] for x in frags_res_type[i]])

        # find best match k
        best_idA = -1.0
        match_k = None
        match_start = None
        match_end = None

        for k in hetero_seq_idxs:
            # align
            seqB = seqs[k]

            # fix a bug in NWalign
            if seqA[0] == seqB[0]:
                seqAx = "".join(seqA[1:])
                seqBx = "".join(seqB[1:])
            else:
                seqAx = seqA
                seqBx = seqB

            new_seqA, align, new_seqB, idA, covA, _, _ = nwalign_fast(
                seqAx, 
                seqBx, 
                temp_dir=nwalign_temp_dir, 
                lib_dir=lib_dir,
                verbose=True,
                fmt=True, 
            )

            if idA > id_cutoff and covA > cov_cutoff:

                # Should be same as line 430, but now, ignore it

                start = find_first_of(align, ":")
                if start == -1:
                    start = len(align)

                end = find_last_of(align, ":")
                if end == -1:
                    end = len(align)
                else:
                    end += 1

                n_align = sum([1 if x == ":" else 0 for x in align])
                r_align = n_align / (end - start)

                #print(i)
                #print(new_seqA)
                #print(align)
                #print(new_seqB)
                #print(idA, covA)
                #print(len(seqA), end - start)
                #print(n_align, end - start, r_align)

                if r_align > r_align_cutoff and idA > best_idA:
                    print("Aligned")

                    best_idA = idA
                    match_k = k
                    match_start = start
                    match_end = end
            
        if match_k is not None:
            print("# Frag {} best match to hetero seq {}".format(i, match_k))
            seq_frags[match_k].append((i, match_start, match_end))

    #exit()

    for k in hetero_seq_idxs:
        print("# Seq {} frags".format(k))
        print(seq_frags[k])

        # Set the assigned fragments as "visited"
        for frag_info in seq_frags[k]:
            frags_is_assigned[frag_info[0]] = True

    print("# Frags is assigned")
    print(frags_is_assigned)
    #exit()

    ########################################
    # Step 1.2: Merge all assigned fragments
    ########################################
    hetero_final_atom_pos = []
    hetero_final_atom_mask = []
    hetero_final_res_type = []
    hetero_final_res_idx = []
    hetero_final_atom_bfactor = []

    for k in range(len(seqs)):
        hetero_chain_atom_pos = []
        hetero_chain_atom_mask = []
        hetero_chain_res_type = []
        hetero_chain_atom_bfactor = []

        starts = []

        # if sequence has not assigned frags
        if k not in seq_frags:
            continue

        for frag_info in seq_frags[k]:
            start = frag_info[1]
            starts.append(start)

        res_idx_start = 0
        for idx in np.argsort(starts):
            frag_info = seq_frags[k][idx]
            frag_idx = frag_info[0]

            hetero_chain_atom_pos.append( frags_atom_pos[frag_idx] )
            hetero_chain_atom_mask.append( frags_atom_mask[frag_idx] )
            hetero_chain_res_type.append( frags_res_type[frag_idx] )
            hetero_chain_atom_bfactor.append( frags_atom_bfactor[frag_idx] )

        if len(hetero_chain_atom_pos) > 0:
            hetero_chain_atom_pos = np.concatenate(hetero_chain_atom_pos, axis=0)
            hetero_chain_atom_mask = np.concatenate(hetero_chain_atom_mask, axis=0)
            hetero_chain_res_type = np.concatenate(hetero_chain_res_type, axis=0)
            hetero_chain_atom_bfactor = np.concatenate(hetero_chain_atom_bfactor, axis=0)

            hetero_chain_res_idx = fix_idx_break(hetero_chain_atom_pos, d_break=4.5)


            # append to final
            hetero_final_atom_pos.append( hetero_chain_atom_pos )
            hetero_final_atom_mask.append( hetero_chain_atom_mask )
            hetero_final_res_type.append( hetero_chain_res_type )
            hetero_final_res_idx.append( hetero_chain_res_idx )
            hetero_final_atom_bfactor.append( hetero_chain_atom_bfactor )





    ##################################################
    # Step 1.2: For homo, reorder with hybrid strategy
    ##################################################
    print("# Homo groups")
    homo_seq_idxs = []
    for k, v in seq_groups.items():
        if len(v) >= 2:
            homo_seq_idxs.append(k) # representative sequence


    #######################################################################
    # Align the homo seqs
    # Unlike hetero seqs, we first find those who CANNOT align to homo seqs
    # Then find the best match to the REPRESENTATIVE sequencs (to split frags into sequence groups)
    #######################################################################
    frags_is_excluded = [False] * len(frags_atom_pos)
    homo_group_frag_idxs = dict()
    match_info = dict()
    match_info_all = dict()
    for k in homo_seq_idxs:
        homo_group_frag_idxs[k] = list()

    for i in range(len(frags_atom_pos)):
        if frags_is_assigned[i]:
            continue

        # ignore too short fragment
        if len(frags_atom_pos[i]) < min_res:
            continue

        # get sequence from structure
        seqA = "".join([index_to_restype_1[x] for x in frags_res_type[i]])

        # find best match k
        best_idA = -1.0
        match_k = None
        match_start = None
        match_end = None
        match_new_seqA = None
        match_align = None
        match_new_seqB = None

        # only use representative sequence
        for k in homo_seq_idxs:
            # align
            seqB = seqs[k]

            # fix a bug in NWalign
            if seqA[0] == seqB[0]:
                seqAx = "".join(seqA[1:])
                seqBx = "".join(seqB[1:])
            else:
                seqAx = seqA
                seqBx = seqB

            new_seqA, align, new_seqB, idA, covA, _, _ = nwalign_fast(
                seqAx, 
                seqBx, 
                temp_dir=nwalign_temp_dir, 
                lib_dir=lib_dir,
                verbose=True,
                fmt=True, 
            )

            if idA > id_cutoff and covA > cov_cutoff:

                start = find_first_of(align, ":")
                if start == -1:
                    start = len(align)

                end = find_last_of(align, ":")
                if end == -1:
                    end = len(align)
                else:
                    end += 1

                n_align = sum([1 if x == ":" else 0 for x in align])
                r_align = n_align / (end - start)

                #print(new_seqA)
                #print(align)
                #print(new_seqB)
                #print(idA, covA)
                #print(len(seqA), end - start)
                #print(n_align, end - start, r_align)

                if r_align > r_align_cutoff and idA > best_idA:
                    best_idA = idA
                    match_k = k
                    match_start = start
                    match_end = end
                    match_new_seqA = new_seqA
                    match_align = align
                    match_new_seqB = new_seqB

        if match_k is None:
            print("# Frag {} cannot aligned to any seqs".format(i))
            frags_is_excluded[i] = True
        else:
            print("# Frag {} best aligns to seq {}".format(i, match_k))
            # group current frag to best match homo group
            homo_group_frag_idxs[match_k].append((i, match_start, match_end))
            match_info[i] = (match_start, match_end)
            match_info_all[i] = (match_k, match_new_seqA, match_align, match_new_seqB)

    #exit()

    print("# Frags is excluded")
    print(frags_is_excluded)


    homo_final_atom_pos = []
    homo_final_atom_mask = []
    homo_final_res_type = []
    homo_final_res_idx = []
    homo_final_atom_bfactor = []
    #####################
    # For each homo group
    #####################
    for k in homo_seq_idxs:
        homo_frag_idxs = [x[0] for x in homo_group_frag_idxs[k]]
        print("##########################################################################")
        print("# Searching for sequence {}".format(k))
        print("# Homo frags")
        print(homo_frag_idxs)

        if len(homo_frag_idxs) == 0:
            print("# Skip search")
            print("##########################################################################")
            continue


        if use_chain_guidance:
            ######################################################
            # Do USalign ns on homo frags to find same chain pairs
            ######################################################
            same_chain_pairs = set()
            n_copy = len(seq_groups[k])
            homo_frags_is_assigned = [False] * len(frags_atom_pos)

            for i_copy in range(n_copy):
                print("# USalign on copy {}".format(i_copy))
                if chain_groups[k] is None:
                    print("# No chain for seq {}".format(k))
                    continue
                (chain_filename, _, _, _, _, chain_bfactor) = chain_groups[k]
                mean_plddt = chain_bfactor[..., 1].mean()

                # select frag
                homo_frag_idxs_to_idxs = dict()

                sel_frag_idx = []
                for ii in range(len(homo_frag_idxs)):
                    i = homo_frag_idxs[ii]
                    homo_frag_idxs_to_idxs[i] = ii

                    if homo_frags_is_assigned[i]:
                        continue
                    sel_frag_idx.append(i)

                if len(sel_frag_idx) == 0:
                    print("# No more frags can be aligned")
                    continue

                print("# Homo   idxs ", homo_frag_idxs)
                print("# Select idxs ", sel_frag_idx)

                fo = pjoin(usalign_temp_dir, f"frags_for_homo_seq_{k}_copy_{i_copy}.cif")
                chains_atom_pos_to_pdb(
                    fo,
                    [np.concatenate([frags_atom_pos[x] for x in sel_frag_idx], axis=0)],
                    [np.concatenate([frags_atom_mask[x] for x in sel_frag_idx], axis=0)],
                    [np.concatenate([frags_res_type[x] for x in sel_frag_idx], axis=0)],
                    chains_bfactors=[np.concatenate([frags_atom_bfactor[x] for x in sel_frag_idx], axis=0)],
                )

                print("# Write frag for seq {} to {}".format(k, fo))
                print("# Align {} and {}".format(fo, chain_filename))
    
                result = run_USalign_ns(
                    chain_filename, fo, lib_dir=lib_dir,
                )
    
    
                # analyze result
                res_idx_to_frag_idx = np.concatenate(
                    [ [x] * len(frags_atom_pos[x]) for x in sel_frag_idx ],
                    axis=0,
                )
                #print(res_idx_to_frag_idx)
                #print(type(res_idx_to_frag_idx))
    
                tm = []
                aligned_idxs = []
                for line in result:
                    fields = line.strip().split()
                    if len(fields) == 0:
                        continue

                    if fields[0] == "TM-score=":
                        tm.append(float(fields[1]))
    
                    if fields[0] == "CA":
                        if fields[-2][0] == 'A':
                            aligned_idxs.append(int(fields[-2][1:]))
                        else:
                            aligned_idxs.append(int(fields[-2]))
    
                aligned_idxs = np.asarray(aligned_idxs, dtype=np.int32)


                temp_list = []
                # calculate ratio and determine whether to keep frags
                for i in range(len(frags_atom_pos)):
                    n0 = (res_idx_to_frag_idx[aligned_idxs] == i).sum()
                    n1 = len(frags_atom_pos[i])
                    r = n0 / n1
                    if r > r_frag_cutoff:
                        homo_frags_is_assigned[i] = True

                        temp_list.append(
                            homo_frag_idxs_to_idxs[i]
                        )
                        print("# Frag {} assigned to seq {} copy {} / {} with r = {:.4f}".format(i, k, i_copy + 1, n_copy, r))

                print(temp_list)


                # construct same chain pairs
                for iii in range(len(temp_list)):
                    for kkk in range(iii + 1, len(temp_list)):
                        same_chain_pairs.add((
                            temp_list[iii],
                            temp_list[kkk],
                        ))
                #exit()
            #print(same_chain_pairs)
            #exit()
        else:
            print("# No using chain guidance for homo")
            same_chain_pairs = None

        #######################################################
        # Do VRP Tracing on homo frags with res count restraint
        #######################################################
        center_pos = np.asarray([
            frags_atom_pos[i][..., 1, :].mean(axis=0) for i in homo_frag_idxs
        ], dtype=np.float32)
        c_ter_pos = np.asarray([
            frags_atom_pos[i][ -1, 1, :] for i in homo_frag_idxs
        ], dtype=np.float32)
        n_ter_pos = np.asarray([
            frags_atom_pos[i][  0, 1, :] for i in homo_frag_idxs
        ], dtype=np.float32)


        # 1. construct distance map
        distance_map_c_to_n = np.linalg.norm(
            c_ter_pos[:, None, :] - n_ter_pos[None, :, :],
            axis=-1,
        )
        np.fill_diagonal(distance_map_c_to_n, 1e6)

        distance_map_center = np.linalg.norm(center_pos[:, None, :] - center_pos[None, :, :], axis=-1) # (n, n)
        np.fill_diagonal(distance_map_center, 1e6)

        #distance_map_c_to_n = distance_map_c_to_n.astype(np.int32)
        #distance_map_center = distance_map_center.astype(np.int32)
        #print("# Distance map c to n")    
        #print(distance_map_c_to_n)
        #print("# Distance map center")
        #print(distance_map_center)

        #distance_map_center = distance_map_center * 0.90
        #distance_map_c_to_n = distance_map_c_to_n * 0.10

        distance_map_center = np.clip(distance_map_center, a_min=1.0, a_max=1e6)
        distance_map_c_to_n = np.clip(distance_map_c_to_n, a_min=1.0, a_max=1e6)

        # center distance should be more important
        distance_map = 2.0 / (1.0 / distance_map_c_to_n + 1.0 / distance_map_center)

        #distance_map = distance_map_center

        distance_map = np.clip(distance_map, a_min=1.0, a_max=1e6)
        distance_map = distance_map.astype(np.int32)
        distance_map = distance_map * 10

        #print("# Distance map")
        #print(distance_map)

        # 2. get residue counts for each fragment
        residue_counts = np.asarray([
            len(frags_atom_pos[i]) for i in homo_frag_idxs
        ], dtype=np.int32)

        print("# Residue counts")
        print(residue_counts)


        #################################
        # 3. determine the vehicle number
        #################################
        overlap_cutoff = 0.60
        #overlap_cutoff = 0.20
        conflict_pairs = set()
        for ii in range(len(homo_group_frag_idxs[k])):
            m_i = homo_group_frag_idxs[k][ii]
            for kk in range(ii + 1, len(homo_group_frag_idxs[k])):
                m_k = homo_group_frag_idxs[k][kk]
                # if match is inconsistent like the following
                # we should use the actual matched idxs

                n_i = len(find_colon_idx(match_info_all[m_i[0]][2]))
                n_k = len(find_colon_idx(match_info_all[m_k[0]][2]))

                if not (n_i > 0):
                    n_i = 1e-6

                if not (n_k > 0):
                    n_k = 1e-6

                n_common = len(find_common_colon_idx(match_info_all[m_i[0]][2], (match_info_all[m_k[0]][2])))

                overlap_min = min(
                    n_common / n_i,
                    n_common / n_k,
                )
                overlap_max = max(
                    n_common / n_i,
                    n_common / n_k,
                )

                overlap = overlap_max

                #print(overlap_min, overlap_max)
                #print("=" * 100)
                #print(overlap, n_i, n_k, n_common)
                #print(match_info_all[m_i[0]][1])
                #print(match_info_all[m_i[0]][2])
                #print(match_info_all[m_i[0]][3])
                #print(match_info_all[m_k[0]][1])
                #print(match_info_all[m_k[0]][2])
                #print(match_info_all[m_k[0]][3])
                #print("=" * 100)

                # has conflict
                if overlap > overlap_cutoff:
                    #print(overlap_min, overlap_max)
                    #print("=" * 100)
                    #print(overlap, n_i, n_k, n_common)
                    #print(match_info_all[m_i[0]][1])
                    #print(match_info_all[m_i[0]][2])
                    #print(match_info_all[m_i[0]][3])
                    #print(match_info_all[m_k[0]][1])
                    #print(match_info_all[m_k[0]][2])
                    #print(match_info_all[m_k[0]][3])
                    #print("=" * 100)

                    conflict_pairs.add((ii, kk))

        #exit()


        # check if idxs appears in both conflict pairs and same chain pairs
        if same_chain_pairs is not None:
            print("# Checking conflict table")
            for p in conflict_pairs:
                if p in same_chain_pairs:
                    print("# Found conflicts {}".format(p))
                    same_chain_pairs.discard(p)
            print("# Done checking conflict table")
            #exit()


        # solve minimum vehicle
        model = cp_model.CpModel()
        max_vehicle = len(homo_group_frag_idxs[k])
        vid = [model.NewIntVar(0, max_vehicle - 1, f'v_{i}') for i in range(max_vehicle)]


        # conflict pairs
        for p in conflict_pairs:
            model.Add(vid[p[0]] != vid[p[1]])

        # same chain pairs
        if same_chain_pairs is not None:
            for p in same_chain_pairs:
                model.Add(vid[p[0]] == vid[p[1]])

        max_used_vehicle = model.NewIntVar(0, max_vehicle - 1, "max_v")
        
        for v in vid:
            model.Add(v <= max_used_vehicle)

        model.Minimize(max_used_vehicle + 1)

        # Solve
        print("# Start solving")
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = args.time_limit
        solver.parameters.num_search_workers = 2
        solver.parameters.log_search_progress = args.log_search
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            n_vehicle = solver.Value(max_used_vehicle) + 1
            print("# Found solution n = {}".format(n_vehicle))
        else:
            n_vehicle = len(seq_groups[k])
            print("# No solution found")

        if n_vehicle < len(seq_groups[k]):
            n_vehicle = len(seq_groups[k])

        print("# End solving")
        print("# Num of frags = {} initial n = {} solved n = {}".format(
            len(homo_group_frag_idxs[k]),
            len(seq_groups[k]),
            n_vehicle, 
        ))


        #print(conflict_pairs) 
        #exit()
        #conflict_pairs = None

        # 4. tracing
        print("# Tracing with restraint")
        result = fragment_tracing(
            # num of fragments
            len(residue_counts),
            # num of vehicles
            n_vehicle,
            # distance map
            distance_map,
            residue_counts,
            args.alpha,
            time_limit=args.time_limit,
            log_search=args.log_search, 
            conflict_pairs=conflict_pairs, 
            same_chain_pairs=same_chain_pairs, 
        )

        if result is None:
            print("# No solution found")
            print("# Tracing without restraint")
            result = fragment_tracing(
                # num of fragments
                len(residue_counts),
                # num of vehicles
                n_vehicle,
                # distance map
                distance_map,
                residue_counts,
                args.alpha,
                time_limit=args.time_limit,
                log_search=args.log_search, 
                conflict_pairs=None, 
            )

        print(result)



        ##################################################
        # After grouping frags into sequence groups
        #   Next, reorder frags inside each sequence group
        #   Simply by sorting each frag's staring position
        ##################################################

        # 4. post-process
        for route in result['routes']:
            homo_chain_atom_pos = []
            homo_chain_atom_mask = []
            homo_chain_res_type = []
            homo_chain_res_idx = []
            homo_chain_atom_bfactor = []

            new_route = [route[x] for x in np.argsort(
                [match_info[ homo_frag_idxs[ii] ][0] for ii in route]
            )]

            print("# Original route")
            print(route)
            print("# Sorted route")
            print(new_route)

            for ii in new_route:
                frag_idx = homo_frag_idxs[ii]
                frags_is_assigned[frag_idx] = True

                homo_chain_atom_pos.append( frags_atom_pos[frag_idx] )
                homo_chain_atom_mask.append( frags_atom_mask[frag_idx] )
                homo_chain_res_type.append( frags_res_type[frag_idx] )
                homo_chain_atom_bfactor.append( frags_atom_bfactor[frag_idx] )

            if len(homo_chain_atom_pos) > 0:
                homo_chain_atom_pos = np.concatenate(homo_chain_atom_pos, axis=0)
                homo_chain_atom_mask = np.concatenate(homo_chain_atom_mask, axis=0)
                homo_chain_atom_bfactor = np.concatenate(homo_chain_atom_bfactor, axis=0)
                homo_chain_res_type = np.concatenate(homo_chain_res_type, axis=0)

                # fix res idx
                homo_chain_res_idx = fix_idx_break(homo_chain_atom_pos, d_break=4.5)

                # append
                homo_final_atom_pos.append( homo_chain_atom_pos )
                homo_final_atom_mask.append( homo_chain_atom_mask )
                homo_final_atom_bfactor.append( homo_chain_atom_bfactor )
                homo_final_res_type.append( homo_chain_res_type )
                homo_final_res_idx.append( homo_chain_res_idx )


        print("# End search")
        print("##########################################################################")



    ###############################
    # Step 3: Merge hetero and homo
    ###############################
    final_atom_pos = hetero_final_atom_pos + homo_final_atom_pos
    final_atom_mask = hetero_final_atom_mask + homo_final_atom_mask
    final_res_type = hetero_final_res_type + homo_final_res_type
    final_res_idx = hetero_final_res_idx + homo_final_res_idx
    final_atom_bfactor = hetero_final_atom_bfactor + homo_final_atom_bfactor


    #####################################
    # Step 4: Append the unassigned frags
    #####################################
    for i in range(len(frags_atom_pos)):
        if not frags_is_assigned[i]:

            final_atom_pos.append( frags_atom_pos[i] )
            final_atom_mask.append( frags_atom_mask[i] )
            final_res_type.append( frags_res_type[i] )
            final_atom_bfactor.append( frags_atom_bfactor[i] )

            # renum residue index
            final_res_idx.append(
                fix_idx_break(frags_atom_pos[i], d_break=4.5)
            )


    # yield final result
    os.makedirs(args.out, exist_ok=True)
    fpdbout = pjoin(args.out, "reorder.cif")
  
    chains_atom_pos_to_pdb(
        fpdbout, 
        chains_atom_pos=final_atom_pos,
        chains_atom_mask=final_atom_mask,
        chains_res_types=final_res_type,
        chains_res_idxs=final_res_idx,
        chains_chain_idxs=None,
        chains_bfactors=final_atom_bfactor,
    )

    t1 = time.time()
    print("# Time consumption {:.4f}".format(t1 - t0))
    print("# Output reordered chains to {}".format(fpdbout))



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", "-p", required=True, help="Input structure")
    parser.add_argument("--seq", "-s", required=True, help="Input sequence")
    parser.add_argument("--chain", nargs='+', help="Input chain template")
    parser.add_argument("--out", "-o", default=".", help="Output dir")
    # solver controls
    parser.add_argument("--alpha", default=0.5)
    parser.add_argument("--time_limit", default=10, help="Time limit for solving")
    parser.add_argument("--num_workers", default=2, help="Num of CPUs to use")
    parser.add_argument("--log_search", action='store_true', help="Whether to log search process")
    args = parser.parse_args()
    main(args)
