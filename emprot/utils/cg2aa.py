import os
import tqdm
import tempfile
import subprocess
import numpy as np

from emprot.io.pdbio import write_atoms_as_pdb, read_pdb

def cg2aa(atom_pos, res_type=None, lib_dir="./", temp_dir=None, verbose=True):
    if res_type is None:
        # set to be all 'ALA'
        res_type = np.zeros( len(atom_pos), dtype=np.int32 )

    n_max_res_per_chain = 1000
    # chain (L, 3) or (L, 1, 3) or (L, 3, 3)
    with tempfile.TemporaryDirectory() as __temp_dir:
        if temp_dir is None:
            temp_dir = __temp_dir

        # if too long, split to many subarray
        all_atom14_pos = []
        all_atom14_mask = []
        for i in range(0, len(atom_pos), n_max_res_per_chain):
        #for i in tqdm.tqdm(range(0, len(atom_pos), n_max_res_per_chain)):
            # input
            ftemp1 = temp_dir + f"/cg_{i}.pdb"
            # output
            ftemp2 = temp_dir + f"/cg_{i}.rebuilt.pdb"
            write_atoms_as_pdb(
                ftemp1, 
                atom_pos[i:i+n_max_res_per_chain], 
                res_type=res_type, 
                ter=False,
            )

            cmd = f"cd {temp_dir} && " + f"{lib_dir}/bin/pulchra -f {ftemp1} -q"
            if verbose:
                print(f"# Running command {cmd}")
            result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0 and \
                os.path.exists(ftemp2):
                print(f"# Successfully recover full-atom structure to {ftemp2}")
            else:
                print(f"# Failed to recover full-atom structure")
                print(result.stderr.decode("utf-8"))
                raise Exception(f"# Failed to recover full-atom structure")

            atom14_pos, atom14_mask, _, _, _ = read_pdb(ftemp2, keep_valid=False)
            
            all_atom14_pos.append(atom14_pos)
            all_atom14_mask.append(atom14_mask)

        # concatenate all subarray
        all_atom14_pos = np.concatenate(all_atom14_pos, axis=0)
        all_atom14_mask = np.concatenate(all_atom14_mask, axis=0)

    return all_atom14_pos, all_atom14_mask


def ca2bb(atom_pos, res_type=None, lib_dir="./", temp_dir=None, verbose=True, check_bond=True):
    atom14_pos, atom14_mask = cg2aa(atom_pos, res_type, lib_dir=lib_dir, temp_dir=temp_dir, verbose=verbose)
    # keep ideal bond
    d_ca_n = 1.459
    d_ca_c = 1.525
    # Use a wide tolerance
    d_ca_n_tolerance = 0.50
    d_ca_c_tolerance = 0.50
    # Make sure an ideal bond is modeled    
    if check_bond:
        for i in range(len(atom_pos)):
            n  = atom14_pos[i][0]
            ca = atom14_pos[i][1]
            c  = atom14_pos[i][2]
            o  = atom14_pos[i][3]

            d = np.linalg.norm(n - ca)
            if not (d_ca_n - d_ca_n_tolerance <= d <= d_ca_n_tolerance + d_ca_n_tolerance):
                v = n - ca
                v /= np.linalg.norm(v) + 1e-6
                new_n = ca + v * d_ca_n
                atom14_pos[i][0] = new_n
 
            d = np.linalg.norm(c - ca)
            if not (d_ca_c - d_ca_c_tolerance <= d <= d_ca_c_tolerance + d_ca_c_tolerance):
                v = c - ca
                v /= np.linalg.norm(v) + 1e-6
                new_c = ca + v * d_ca_c

                delta = new_c - c

                atom14_pos[i][2] = c + delta
                atom14_pos[i][3] = o + delta

    return atom14_pos, atom14_mask

if __name__ == '__main__':
    pass
