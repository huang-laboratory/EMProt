import os
import numpy as np
import torch
from scipy.spatial import cKDTree
from collections import namedtuple

from emprot.io.pdbio import read_pdb, chains_atom_pos_to_pdb

def distance(a, b, sqrt=True):
    d2 = np.sum(np.power(np.subtract(a, b), 2), axis=-1)
    if sqrt:
        d = np.sqrt(d2)
        return d
    else:
        return d2

def remove_overlapping_ca(
    ca_positions: np.ndarray, radius_threshold: float = 0.5,
) -> np.ndarray:
    kdtree = cKDTree(ca_positions)
    existence_mask = np.ones(len(ca_positions), dtype=bool)

    for i in range(len(ca_positions)):
        if existence_mask[i]:
            too_close = np.array(
                kdtree.query_ball_point(ca_positions[i], r=radius_threshold,)
            )
            too_close = too_close[too_close != i]
            existence_mask[too_close] = False
    return existence_mask

def flood_fill(
    atomc_positions, b_factors=None, n_c_distance_threshold=2.1, is_nucleotide=False,
):
    # Check if b_factors is not provoded
    if b_factors is None:
        b_factors = np.ones(len(atomc_positions), dtype=np.float32)*100.0
    assert len(atomc_positions) == len(b_factors), "Shape no match"

    if is_nucleotide:
        raise "Error Not Implemented for nucleotides"
        """
        n_idx, c_idx = (
            restype3_to_atoms["A"].index("P"),
            restype3_to_atoms["A"].index("O3'"),
        )
        """
    else:
        n_idx, c_idx = (
            0, 2
        )

    n_positions = atomc_positions[:, n_idx]
    c_positions = atomc_positions[:, c_idx]
    kdtree = cKDTree(c_positions)
    b_factors_copy = np.copy(b_factors)

    chains = []
    chain_ends = {}
    while np.any(b_factors_copy != -1):
        idx = np.argmax(b_factors_copy)
        possible_indices = np.array(
            kdtree.query_ball_point(
                n_positions[idx], r=n_c_distance_threshold, return_sorted=True
            )
        )
        possible_indices = possible_indices[possible_indices != idx]

        got_chain = False
        if len(possible_indices) > 0:
            for possible_prev_residue in possible_indices:
                if possible_prev_residue == idx:
                    continue
                if possible_prev_residue in chain_ends:
                    chains[chain_ends[possible_prev_residue]].append(idx)
                    chain_ends[idx] = chain_ends[possible_prev_residue]
                    del chain_ends[possible_prev_residue]
                    got_chain = True
                    break
                elif b_factors_copy[possible_prev_residue] >= 0.0:
                    chains.append([possible_prev_residue, idx])
                    chain_ends[idx] = len(chains) - 1
                    b_factors_copy[possible_prev_residue] = -1
                    got_chain = True
                    break

        if not got_chain:
            chains.append([idx])
            chain_ends[idx] = len(chains) - 1

        b_factors_copy[idx] = -1

    og_chain_starts = np.array([c[0] for c in chains], dtype=np.int32)
    og_chain_ends = np.array([c[-1] for c in chains], dtype=np.int32)

    chain_starts = og_chain_starts.copy()
    chain_ends = og_chain_ends.copy()

    n_chain_starts = n_positions[chain_starts]
    c_chain_ends = c_positions[chain_ends]
    N = len(chain_starts)
    spent_starts, spent_ends = set(), set()

    kdtree = cKDTree(n_chain_starts)

    no_improvement = 0
    chain_end_match = 0

    while no_improvement < 2 * N:
        found_match = False
        if chain_end_match in spent_ends:
            no_improvement += 1
            chain_end_match = (chain_end_match + 1) % N
            continue

        start_matches = kdtree.query_ball_point(
            c_chain_ends[chain_end_match], r=n_c_distance_threshold, return_sorted=True
        )
        for chain_start_match in start_matches:
            if (
                chain_start_match not in spent_starts
                and chain_end_match != chain_start_match
            ):
                chain_start_match_reidx = np.nonzero(
                    chain_starts == og_chain_starts[chain_start_match]
                )[0][0]
                chain_end_match_reidx = np.nonzero(
                    chain_ends == og_chain_ends[chain_end_match]
                )[0][0]
                if chain_start_match_reidx == chain_end_match_reidx:
                    continue

                new_chain = (
                    chains[chain_end_match_reidx] + chains[chain_start_match_reidx]
                )

                chain_arange = np.arange(len(chains))
                tmp_chains = np.array(chains, dtype=object)[
                    chain_arange[
                        (chain_arange != chain_start_match_reidx)
                        & (chain_arange != chain_end_match_reidx)
                    ]
                ].tolist()
                tmp_chains.append(new_chain)
                chains = tmp_chains

                chain_starts = np.array([c[0] for c in chains], dtype=np.int32)
                chain_ends = np.array([c[-1] for c in chains], dtype=np.int32)

                spent_starts.add(chain_start_match)
                spent_ends.add(chain_end_match)
                no_improvement = 0
                found_match = True
                chain_end_match = (chain_end_match + 1) % N
                break

        if not found_match:
            no_improvement += 1
            chain_end_match = (chain_end_match + 1) % N

    return chains



def merge_chains(chains, atom_positions, lower=2.3, upper=4.1, ca_upper=4.0):
    # Calculate C and N atom positions for each chain
    c_positions = [atom_positions[chain[-1]][-1] for chain in chains]
    n_positions = [atom_positions[chain[ 0]][ 0] for chain in chains]

    ca_positions_head = [atom_positions[chain[ 0]][1] for chain in chains]
    ca_positions_tail = [atom_positions[chain[-1]][1] for chain in chains]

    merged_chains = chains.copy()
    
    d = lower
    delta = 0.2
    visited = set()
    while d < upper:
        ntree = cKDTree(n_positions)

        for i in range(len(c_positions)):
            if len(c_positions[i]) == 0:
                continue

            if i in visited:
                continue

            idxs = np.asarray(ntree.query_ball_point(c_positions[i], r=d), dtype=np.int32)
            idxs = idxs[idxs != i]

            # If no neighbor or more than one neighbor, ignore
            if len(idxs) > 1 or len(idxs) == 0:
                continue

            if idxs[0] in visited:
                continue

            # Check the Ca-Ca distance
            dcaca = distance(ca_positions_tail[i], ca_positions_head[idxs[0]])
            if dcaca > ca_upper:
                continue

            # Merge i and idxs[0]
            merged_chains[i] = merged_chains[i] + merged_chains[idxs[0]].copy()
            merged_chains[idxs[0]] = []

            c_positions[i] = c_positions[idxs[0]].copy()
            c_positions[idxs[0]] = []

            n_positions[idxs[0]] = c_positions[i].copy()

            visited.add(idxs[0])

        d += delta

    merged_chains = [chain for chain in merged_chains if chain]
    return merged_chains

def reverse_ncac(ncac):
    # ncac (L, 3, 3)
    d_ca_n = 1.459
    d_ca_c = 1.525
    # First reverse each residue
    new_ncac = np.zeros_like(ncac, dtype=np.float32)
    # Next reverse the N and C
    for ii in range(len(ncac)):
        ca = ncac[ii][1]
        n  = ncac[ii][0]
        c  = ncac[ii][2]

        v = c-ca
        v = v / (np.linalg.norm(v) + 1e-6)
        new_n = ca + v * d_ca_n

        v = n-ca
        v = v / (np.linalg.norm(v) + 1e-6)
        new_c = ca + v * d_ca_c

        new_ncac[len(ncac) - ii - 1][0] = new_n
        new_ncac[len(ncac) - ii - 1][1] = ca
        new_ncac[len(ncac) - ii - 1][2] = new_c

    return new_ncac


# Check if two residue is mergeable
def is_mergeable(c_pos, n_pos, d=2.3):
    return np.linalg.norm(c_pos - n_pos) < d

def is_ca_mergeable(ca_pos0, ca_pos1, d=4.5):
    return np.linalg.norm(ca_pos0 - ca_pos1) < d

def merge_chains_reversible(chains, atom_positions, lower=2.3, upper=3.0, ca_upper=4.0):
    # Sort chains
    merged_chains = chains.copy()
    merged_chains.sort(key=lambda x:len(x), reverse=False)

    # For each small chain
    visited = set()
    for i, chaini in enumerate(merged_chains):
        # Reverse shorter chaini
        atom_pos_i = reverse_ncac(atom_positions[chaini][..., :3, :])
        for k, chaink in enumerate(merged_chains):
            if i >= k:
                continue

            d = lower
            delta = 0.2

            atom_pos_k = atom_positions[chaink][..., :3, :]

            merged = False
            ca_flag = is_ca_mergeable(atom_pos_i[-1][1], atom_pos_k[0][1], ca_upper)

            if not ca_flag:
                continue

            while d < upper:
                if is_mergeable(atom_pos_i[-1][-1], atom_pos_k[0][0], d):
                    # Merge shorter to long
                    merged = True
                    atom_positions[chaini] = atom_pos_i
                    merged_chains[k] = chaini.copy() + merged_chains[k]
                    merged_chains[i] = []
                    break
                elif is_mergeable(atom_pos_i[0][0], atom_pos_k[-1][-1], d):
                    merged = True
                    atom_positions[chaini] = atom_pos_i
                    merged_chains[k] = merged_chains[k] + chaini.copy()
                    merged_chains[i] = []
                    break

                d += delta
            if merged:
                break
    merged_chains = [chain for chain in merged_chains if chain]
    return merged_chains, atom_positions



def thread_and_merge_ncac(ncac_pos):
    # ncac_pos (L, 3, 3)
    atom3_pos = ncac_pos

    # 0. Thread according to peptide-bonds
    chains = flood_fill(
        atom3_pos,
        n_c_distance_threshold=2.1,
    )
    print("# Found {:<4d} chains before merging".format(len(chains)))

    ca_upper = 4.5
    # 1. Merge to larger fragments according to peptide-bonds
    chains = merge_chains(chains, atom3_pos, lower=2.1, upper=2.4, ca_upper=ca_upper)

    # The network prediction is not perfect
    # It can reverse the N-CA-C sometimes
    # 2. Reverse a small fragment and see if it can link to a larger fragment
    chains, atom3_pos = merge_chains_reversible(chains, atom3_pos, lower=2.1, upper=2.4, ca_upper=ca_upper)

    # 3. Merge to larger fragments again
    chains = merge_chains(chains, atom3_pos, lower=2.1, upper=2.4, ca_upper=ca_upper)

    # Only keep >=3 residues
    #chains = [chain for chain in chains if len(chain) > 3]
    print("# Found {:<4d} chains after  merging".format(len(chains)))
    
    return chains


def main(argv):
    fpdb = argv[1]
    atom14_pos, atom14_mask, res_type, res_idx, chain_idx = read_pdb(fpdb)
    atom3_pos = atom14_pos[..., :3, :]
    atom3_mask = atom14_mask[..., :3]

    chains = thread_and_merge_ncac(atom3_pos)

    # Write chains to cif
    chains_atom_pos_to_pdb(
        filename="chains.cif",
        chains_atom_pos=[atom3_pos[chain] for chain in chains],
        chains_atom_mask=[atom3_mask[chain] for chain in chains],
        chains_res_types=[res_type[chain] for chain in chains],
    )

if __name__ == "__main__":
    import sys
    main(sys.argv)
