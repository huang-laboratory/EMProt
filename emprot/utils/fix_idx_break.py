import numpy as np

def fix_idx_break(chain_atom_pos, d_break=4.5, delta=10):
    chain_res_idx = np.arange(0, len(chain_atom_pos))

    new_res_idx = []
    current_idx = 0

    new_res_idx = []
    cumulative_shift = 0

    for k in range(len(chain_atom_pos)):
        if k > 0:
            d = np.linalg.norm(chain_atom_pos[k][1] - chain_atom_pos[k - 1][1])
            if d > d_break:
                cumulative_shift += delta

        new_idx = chain_res_idx[k] + cumulative_shift
        new_res_idx.append(new_idx)

    #print(len(chain_atom_pos))
    #print(len(new_res_idx))
    #print(new_res_idx)

    return np.array(new_res_idx)

