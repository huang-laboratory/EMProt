import os
import torch
import numpy as np

from typing import Dict, Callable, List

from emprot.utils.hmm_sequence_align import (
    fix_chains_pipeline,
    prune_and_connect_chains,
    FixChainsOutput,
)
from emprot.utils.match_to_sequence import MatchToSequence
from emprot.utils.misc_utils import list_select, pjoin, abspath
from emprot.utils.grid import get_grid_value
from emprot.utils.residue_constants import (
    restype_3_to_index, 
    restype_1_to_index, 
    restype_3to1, 
    restype_1to3, 
    select_torsion_angles, 
)

from emprot.io.seqio import read_fasta
from emprot.io.pdbio import (
    read_pdb,
    split_atoms_to_chains,
    read_pdb_chains,
    chains_atom_pos_to_pdb,
    convert_to_chains,
)

from emprot.utils.flood_fill import flood_fill
from emprot.utils.protein import (
    torsion_angles_to_frames, 
    frames_and_literature_positions_to_atomc_pos,
)

def average_by_groups(aa_logits):
    return aa_logits

def final_results_align_to_sequence(
    final_results,
    aa_map, 
    seq_dir, 
    output_dir, 
):

    # Setting tempdir
    out_dir = abspath(output_dir)
    hmm_temp_dir = pjoin(out_dir, "hmm")
    os.makedirs(hmm_temp_dir, exist_ok=True)
    print("# Setting output directory to {}".format(out_dir))
    print("# Setting HMM temp dir to {}".format(hmm_temp_dir))

    # Print seqs
    seqs = [seq.upper() for seq in read_fasta(seq_dir)]
    print("# Found {} sequences".format(len(seqs)))
    for k, seq in enumerate(seqs):
        print(f"# Seq {k}")
        print(f"# {seq}")

    # Read aa logits
    aa_logits_map = aa_map['map']
    aa_logits_origin = aa_map['origin']
    aa_logits_vsize = aa_map['voxel_size']


    #############################################
    ### Get dummy backbone coords for tracing ###
    #############################################
    dummy_aatype = np.ones(len(final_results['pred_affines'])).astype(np.int32) # (N, )
    pred_torsions = select_torsion_angles(
        torch.from_numpy(final_results["pred_torsions"]), 
        aatype=dummy_aatype,
    ) # (N, 8, 2)
    #print(pred_torsions.shape)

    pred_frames = torsion_angles_to_frames(
        dummy_aatype, 
        torch.from_numpy(final_results["pred_affines"]), 
        pred_torsions, 
    ) # (N, 9, 3, 4)
    #print(pred_frames.shape)

    pred_atom_pos = frames_and_literature_positions_to_atomc_pos(
        dummy_aatype, 
        pred_frames, 
    ).numpy() # (N, 23, 3)
    #print(pred_atom_pos.shape)

    # Tracing final results
    chains = flood_fill(pred_atom_pos, is_nucleotide=False)

    reordered_final_results = final_results.copy()
    for key in final_results.keys():
        reordered_final_results[key] = np.concatenate([final_results[key][idx] for idx in chains], axis=0)

    atom14_pos = [pred_atom_pos[idx] for idx in chains]
    atom14_pos = np.concatenate(atom14_pos, axis=0)

    chains_ca_pos = [pred_atom_pos[idx][..., 1, :] for idx in chains]
    print("# Trace to {} fragments a total of {} CAs".format(len(chains_ca_pos), len(atom14_pos)))


    # Assign aa_logits for each coords
    print("# Assign aa logits")
    chains_aa_logits = []
    ca_pos = np.concatenate(chains_ca_pos).reshape(-1, 3)
    chains = []
    chains_prot_mask = []

    bidirection = False
    start = 0
    for i, acoords in enumerate(chains_ca_pos):
        # AA logits
        aa_logits = []
        for k in range(len(acoords)):
            logits = get_grid_value(aa_logits_map, (acoords[k]-aa_logits_origin)/aa_logits_vsize)
            aa_logits.append(logits)

        aa_logits = np.asarray(aa_logits, dtype=np.float32)

        # TODO averaged by aa groups
        aa_logits = average_by_groups(aa_logits)

        chains_aa_logits.append(aa_logits)

        # Indices
        idxs = np.asarray(list(range(start, start + len(acoords))))
        chains.append(idxs)
        start += len(acoords)

        # Mask
        mask = np.asarray([True for i in range(len(acoords))])
        chains_prot_mask.append(mask)

        # Reverse the path
        if bidirection:
            rev_aa_logits = aa_logits[::-1]
            rev_idxs = idxs[::-1]
            rev_mask = mask[::-1]
            chains_aa_logits.append(rev_aa_logits)
            chains.append(rev_idxs)
            chains_prot_mask.append(rev_mask)

    # Fix chains pipeline
    fix_chains_output = fix_chains_pipeline(
        prot_sequences=seqs,
        rna_sequences=[],
        dna_sequences=[],
        chains=chains,
        chain_aa_logits=chains_aa_logits,
        ca_pos=ca_pos,
        chain_prot_mask=chains_prot_mask,
        base_dir=hmm_temp_dir,
        postprocess=False, # always false here for the first time
    )

    # Find better between consecutive
    if bidirection:
        score_cutoff = 0.0
        kept_idxs = []

        chains0 = []
        best_match_output0 = MatchToSequence()
        unmodelled_sequences0 = None

        for k in range(0, len(chains), 2):
            score0 = fix_chains_output.best_match_output.match_scores[k]
            score1 = fix_chains_output.best_match_output.match_scores[k+1]
            if score0 > score1:
                score = score0
                kk = k
            else:
                score = score1
                kk = k + 1
            if not (score > score_cutoff):
                continue
            kept_idxs.append(kk)

        # Only keep better
        chains0 = list_select(fix_chains_output.chains, kept_idxs)

        best_match_output0.new_sequences = list_select(fix_chains_output.best_match_output.new_sequences, kept_idxs)
        best_match_output0.residue_idxs = list_select(fix_chains_output.best_match_output.residue_idxs, kept_idxs)
        best_match_output0.sequence_idxs = list_select(fix_chains_output.best_match_output.sequence_idxs, kept_idxs)
        best_match_output0.key_start_matches = list_select(fix_chains_output.best_match_output.key_start_matches, kept_idxs)
        best_match_output0.key_end_matches = list_select(fix_chains_output.best_match_output.key_end_matches, kept_idxs)
        best_match_output0.match_scores = list_select(fix_chains_output.best_match_output.match_scores, kept_idxs)
        best_match_output0.hmm_output_match_sequences = list_select(fix_chains_output.best_match_output.hmm_output_match_sequences, kept_idxs)
        best_match_output0.exists_in_sequence_mask = list_select(fix_chains_output.best_match_output.exists_in_sequence_mask, kept_idxs)
        best_match_output0.is_nucleotide = list_select(fix_chains_output.best_match_output.is_nucleotide, kept_idxs)

        fix_chains_output = FixChainsOutput(
            chains=chains0.copy(),
            best_match_output=best_match_output0,
            unmodelled_sequences=unmodelled_sequences0,
        )

    # no bi-direction
    else:
        chains0 = fix_chains_output.chains


    # Update other chain infos too
    update_ca_pos = ca_pos.copy() # actually ca_pos does need to be updated
    update_chains = chains0.copy()
    update_chains_prot_mask = []
    update_chains_aa_logits = []

    for chain in chains0:
        acoords = ca_pos[chain]
        # AA logits
        aa_logits = []
        for k in range(len(acoords)):
            logits = get_grid_value(aa_logits_map, (acoords[k]-aa_logits_origin)/aa_logits_vsize)
            aa_logits.append(logits)

        aa_logits = np.asarray(aa_logits, dtype=np.float32)
        update_chains_aa_logits.append(aa_logits)

        # Mask
        mask = np.asarray([True for i in range(len(acoords))])
        update_chains_prot_mask.append(mask)


    # Do fix chains pipeline again but reorder chains
    fix_chains_output = fix_chains_pipeline(
        prot_sequences=seqs,
        rna_sequences=[],
        dna_sequences=[],
        chains=update_chains,
        chain_aa_logits=update_chains_aa_logits,
        ca_pos=ca_pos,
        chain_prot_mask=update_chains_prot_mask,
        base_dir=hmm_temp_dir,
        postprocess=True,
    )



    # Save the unpruned chains
    restypes = list(restype_1_to_index.keys())
    new_chains = fix_chains_output.chains
    new_sequences = fix_chains_output.best_match_output.new_sequences

    unpruned_chains_res_types = []
    unpruned_chains_atom_pos = []
    unpruned_chains_atom_mask = []
    unpruned_chains_res_idxs = []
    unpruned_chains_chain_idxs = []

    # For reordered
    reorder_chains = True
    sorted_idxs = list(range(0, len(fix_chains_output.chains)))
    if reorder_chains:
        # Sort in descending order
        sorted_idxs.sort(key=lambda x:len(fix_chains_output.chains[x]), reverse=True)

    match_score_cutoff = 0.00
    min_length_cutoff = 5
    for i in range(0, len(fix_chains_output.chains)):
        k = sorted_idxs[i]
        score = fix_chains_output.best_match_output.match_scores[k]
        if score < match_score_cutoff:
            continue

        new_sequences_str = "".join([  restypes[x] for x in new_sequences[k] ])
        if len(new_sequences_str) < min_length_cutoff:
            continue

        unpruned_chains_res_types.append(new_sequences[k])

        atom_pos = atom14_pos[new_chains[k]]

        unpruned_chains_atom_pos.append(atom_pos)

        # only use bb
        temp_mask = np.ones(atom_pos.shape[:2], dtype=bool)
        temp_mask[..., 4:] = False

        unpruned_chains_atom_mask.append(temp_mask)

        unpruned_chains_res_idxs.append( np.arange(0, len(atom_pos), dtype=np.int32) )
        unpruned_chains_chain_idxs.append( i )

    """
    new_sequences,
    residue_idxs, # starts from 1
    sequence_idxs,
    key_start_matches,
    key_end_matches,
    match_scores,
    hmm_output_match_sequences,
    exists_in_sequence_mask,
    is_nucleotide
    """

    # Fill unmatched gaps
    def fill_gaps(arr, upper_bound):
        mask = np.zeros_like(arr, dtype=bool) # label which one is updated

        lower_bound = 1
        length = len(arr)
        # scan from left to right until meets the first number not -1
        left_start = np.where(arr != -1)[0][0]
        for i in range(left_start - 1, -1, -1):
            next_value = arr[i + 1] - 1
            if next_value >= lower_bound:
                arr[i] = next_value
                mask[i] = True
            else:
                break

        # scan from right to left until meets the first number not -1
        right_start = np.where(arr != -1)[0][-1]
        for i in range(right_start + 1, length):
            next_value = arr[i - 1] + 1
            if next_value <= upper_bound:
                arr[i] = next_value
                mask[i] = True
            else:
                break
        return arr, mask


    def fix_match(best_match_output, seqs, match_score_cutoff=0.60, len_cutoff=10):
        (
            new_sequences,
            residue_idxs,
            sequence_idxs,
            key_start_matches,
            key_end_matches,
            match_scores,
            hmm_output_match_sequences,
            exists_in_sequence_mask,
            is_nucleotide_list,
        ) = ([], [], [], [], [], [], [], [], [])

        for i in range(len(best_match_output.residue_idxs)):

            # ignore bad small fragments
            if len(best_match_output.residue_idxs[i]) >= len_cutoff and \
                best_match_output.match_scores[i] >= match_score_cutoff:

                sequence_idx = best_match_output.sequence_idxs[i]
                # get new residue idx
                residue_idx, residue_idx_update_mask = fill_gaps(
                    best_match_output.residue_idxs[i].copy(),
                    upper_bound=len(seqs[sequence_idx]),
                )
                residue_idxs.append(residue_idx)

                # update calculated using new mask
                update_idx = np.where(residue_idx_update_mask == True)[0]
                new_sequence = best_match_output.new_sequences[i].copy()
                hmm_output_match_sequence = list(best_match_output.hmm_output_match_sequences[i])
                for k in update_idx:
                    new_sequence[k] = restype_1_to_index[seqs[sequence_idx][residue_idx[k] - 1]]
                    hmm_output_match_sequence[k] = seqs[sequence_idx][residue_idx[k] - 1]
                hmm_output_match_sequence = "".join(hmm_output_match_sequence)

                new_sequences.append(new_sequence)
                hmm_output_match_sequences.append(hmm_output_match_sequence)

                # simply update
                exists_in_sequence_mask.append((residue_idx != -1).astype(np.int32))
                match_scores.append((residue_idx != -1).sum() / len(residue_idx))
                key_start_matches.append(-1 if residue_idx[0] == -1 else residue_idx[0])
                key_end_matches.append(
                    residue_idx[np.where(residue_idx != -1)[0][-1]]
                )

                sequence_idxs.append(best_match_output.sequence_idxs[i])
                is_nucleotide_list.append(best_match_output.is_nucleotide[i])
            else:
                new_sequences.append(best_match_output.new_sequences[i])
                residue_idxs.append(best_match_output.residue_idxs[i])
                sequence_idxs.append(best_match_output.sequence_idxs[i])
                key_start_matches.append(best_match_output.key_start_matches[i])
                key_end_matches.append(best_match_output.key_end_matches[i])
                match_scores.append(best_match_output.match_scores[i])
                hmm_output_match_sequences.append(best_match_output.hmm_output_match_sequences[i])
                exists_in_sequence_mask.append(best_match_output.exists_in_sequence_mask[i])
                is_nucleotide_list.append(best_match_output.is_nucleotide[i])

        # return
        return MatchToSequence(
            new_sequences=new_sequences,
            residue_idxs=residue_idxs,
            sequence_idxs=sequence_idxs,
            key_start_matches=np.array(key_start_matches),
            key_end_matches=np.array(key_end_matches),
            match_scores=np.array(match_scores),
            hmm_output_match_sequences=hmm_output_match_sequences,
            exists_in_sequence_mask=exists_in_sequence_mask,
            is_nucleotide=is_nucleotide_list,
        )


    print("# Fixing gaps")
    new_best_match_output = fix_match(fix_chains_output.best_match_output, seqs)
    new_fix_chains_output = FixChainsOutput(
        chains=fix_chains_output.chains,
        best_match_output=new_best_match_output,
        unmodelled_sequences=None,
    )
    fix_chains_output = new_fix_chains_output


    # Prune and connect chains
    print("# Before prune and connect we have {} chains".format(len(fix_chains_output.chains)))
    best_match_output = fix_chains_output.best_match_output
    flag_prune_and_connect_chains = True
    if flag_prune_and_connect_chains:
        aggressive_pruning = True
        fix_chains_output = prune_and_connect_chains(
            chains=fix_chains_output.chains,
            best_match_output=fix_chains_output.best_match_output,
            ca_pos=ca_pos,
            aggressive_pruning=aggressive_pruning,
            chain_prune_length=4,
        )
    print("# After  prune and connect we have {} chains".format(len(fix_chains_output.chains)))


    restypes = list(restype_1_to_index.keys())
    new_chains = fix_chains_output.chains
    new_sequences = fix_chains_output.best_match_output.new_sequences



    ###################################
    ### Get True all-atom positions ###
    ###################################
    for i in range(0, len(fix_chains_output.chains)):
        idx = fix_chains_output.chains[i]

        aatype = fix_chains_output.best_match_output.new_sequences[i]

        pred_torsions = select_torsion_angles(
            torch.from_numpy(reordered_final_results['pred_torsions'][idx]),
            aatype=aatype,
            normalize=True, 
        ) # (L, 8, 2)

        pred_affines = torch.from_numpy(reordered_final_results['pred_affines'][idx])
    
        pred_frames = torsion_angles_to_frames(
            aatype,
            pred_affines,
            pred_torsions,
        ) # (L, 23, 4, 4)
    
        pred_atom_pos = frames_and_literature_positions_to_atomc_pos(
            aatype,
            pred_frames,
        ).numpy() # (L, 23, 3)

        atom14_pos[idx] = pred_atom_pos




    # Gather final results
    chains_res_types = []
    chains_atom_pos = []
    chains_atom_mask = []
    chains_res_idxs = []
    chains_chain_idxs = []

    # For reordered
    reorder_chains = True
    sorted_idxs = list(range(0, len(fix_chains_output.chains)))
    if reorder_chains:
        # Sort in descending order
        sorted_idxs.sort(key=lambda x:len(fix_chains_output.chains[x]), reverse=True)

    match_score_cutoff = 0.40
    min_length_cutoff = 8
    for i in range(0, len(fix_chains_output.chains)):
        k = sorted_idxs[i]
        score = fix_chains_output.best_match_output.match_scores[k]

        if score < match_score_cutoff:
            pass
            #continue

        new_sequences_str = "".join([  restypes[x] for x in new_sequences[k] ])

        chains_res_types.append(new_sequences[k])

        atom_pos = atom14_pos[new_chains[k]]
        chains_atom_pos.append(atom_pos)

        atom_mask = np.logical_not(np.all(np.abs(atom_pos) < 1e-3, axis=-1))
        chains_atom_mask.append(atom_mask)

        #chains_atom_mask.append(
        #    np.ones(atom_pos.shape[:2], dtype=bool),
        #)

        chains_res_idxs.append(fix_chains_output.best_match_output.residue_idxs[k])
        chains_chain_idxs.append(fix_chains_output.best_match_output.sequence_idxs[k])


    # If no chains after pruned
    if not (len(chains_atom_pos) > 0):
        print("WARNING no chains after pruning")
        print("WARNING use unpruned chains")
        chains_atom_pos = unpruned_chains_atom_pos
        chains_atom_mask = unpruned_chains_atom_mask
        chains_res_types = unpruned_chains_res_types
        chains_res_idxs = unpruned_chains_res_idxs


    # Output each individual chain
    fchainsout = os.path.join(out_dir, "denovo_chains")
    os.makedirs(fchainsout, exist_ok=True)
    for i in range(len(chains_atom_pos)):
        fchainout = os.path.join(fchainsout, f"denovo_chain_{i}.pdb")
        chains_atom_pos_to_pdb(
            filename=fchainout,
            chains_atom_pos=[chains_atom_pos[i]],
            chains_atom_mask=[chains_atom_mask[i]],
            chains_res_types=[chains_res_types[i]],
            chains_res_idxs=[chains_res_idxs[i]],
            suffix='pdb',
        )
    print("# Output individual chains to {}/denovo_chain_*.pdb".format(fchainsout))


    # Output final chains
    fpdbout = os.path.join(out_dir, "output.cif")
    chains_atom_pos_to_pdb(
        filename=fpdbout,
        chains_atom_pos=chains_atom_pos,
        chains_atom_mask=chains_atom_mask,
        chains_res_types=chains_res_types,
        chains_res_idxs=chains_res_idxs,
        suffix='cif',
    )
    print("# Output denovo chains to {}".format(fpdbout))


