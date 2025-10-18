import os
import torch
import numpy as np
from typing import Dict, Callable, List

from emprot.io.pdbio import chains_atom_pos_to_pdb
from emprot.utils.protein import (
    torsion_angles_to_frames,
    frames_and_literature_positions_to_atomc_pos,
)

from emprot.utils.misc_utils import abspath
from emprot.utils.residue_constants import select_torsion_angles

def final_results_to_cif_refine(
    final_results,
    protein,
    output_dir, 
):
    output_dir = abspath(output_dir)

    # Get all-atom pos
    aatype = protein.aatype

    pred_torsions = select_torsion_angles(
        torch.from_numpy(final_results['pred_torsions']),
        aatype=aatype,
        normalize=True,
    ) # (L, 8, 2)

    pred_affines = torch.from_numpy(final_results['pred_affines'])

    pred_frames = torsion_angles_to_frames(
        aatype,
        pred_affines,
        pred_torsions,
    ) # (L, 23, 4, 4)

    pred_atom_pos = frames_and_literature_positions_to_atomc_pos(
        aatype,
        pred_frames,
    ).numpy() # (L, 23, 3)


    # Write to file
    chains_atom_pos = []
    chains_atom_mask = []
    chains_res_type = []
    chains_res_idx = []
    chains_bfactor = []
    n_chain = protein.chain_index.max() + 1

    for i in range(n_chain):
        mask = protein.chain_index == i

        chains_atom_pos.append(pred_atom_pos[mask])
        chains_atom_mask.append(protein.atomc_mask[mask])
        chains_res_type.append(protein.aatype[mask])
        chains_res_idx.append(protein.residue_index[mask])
        chains_bfactor.append(protein.b_factors[mask])


    fpdbout = os.path.join(output_dir, "output.cif")
    chains_atom_pos_to_pdb(
        fpdbout,
        chains_atom_pos,
        chains_atom_mask,
        chains_res_type,
        chains_res_idx,
        chains_bfactors=chains_bfactor, 
    )

    print("# Write refined structure to {}".format(fpdbout))
