import os
import time
import math
import tempfile
import numpy as np
import subprocess

from emprot.utils.grid import get_grid_value_interp, get_grid_value, get_grid_value_batch
from emprot.utils.cryo_utils import parse_map
from emprot.utils.flood_fill import thread_and_merge_ncac
from emprot.utils.misc_utils import abspath, pjoin

from emprot.io.pdbio import (
    read_pdb, 
    ca_to_atom3, 
    chains_atom_pos_to_pdb,
)

def distance(x, y):
    return np.sqrt(np.sum(np.power(np.subtract(x, y), 2)))

def pairwise_distances(a, b):
    """
    a, b : np.ndarray of shape [L, d], [M, d]
    """
    assert a.ndim == b.ndim
    assert a.shape[-1] == b.shape[-1]
    distances = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return distances

def set_tsp_file(mat, path='./', tsptype='TSP'):
    # Write to file
    filename = path + '/exp.tsp'
    f = open(filename, 'w')
    f.write('NAME : thread\nCOMMENT : thread\nTYPE : {}\nDIMENSION : {}\nEDGE_WEIGHT_TYPE : EXPLICIT\nEDGE_WEIGHT_FORMAT : FULL_MATRIX\nEDGE_WEIGHT_SECTION\n'.format(tsptype, len(mat)))

    for i in range(len(mat) - 1):
        for j in range(len(mat) - 1):
            f.write('{} '.format(int(mat[i][j])))
        f.write('0\n')
    for i in range(len(mat)):
        f.write('0 ')
    f.write('\nEOF\n')
    #print('Write to tsp file to {}'.format(path))


def set_tsp_params(depot=1, vehicles=1, path='./', time_limit=300):
    # Time limit is in seconds
    # Write params to file
    filename = path + '/exp.par'
    with open(filename, 'w') as f:
        f.write('PROBLEM_FILE = {}\nOPTIMUM = 99999999\nMOVE_TYPE = 5\nPATCHING_C = 3\nPATCHING_A = 2\nRUNS = 5\nVEHICLES = {}\nDEPOT = {}\nTOUR_FILE = {}\nTIME_LIMIT = {}\n'.format( os.path.join(path, 'exp.tsp'), vehicles, depot, os.path.join(path, 'exp.out'), time_limit))
    #print('Write tsp params to {}'.format(path))


def run_tsp(vehicles=1, path='./', lkh_dir='./'):
    filename = path + '/exp.par'
    cmd = lkh_dir + '/LKH ' + filename
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #print('Run LKH for num path = {}'.format(vehicles))


def distance(x, y):
    return np.sqrt(np.sum(np.power(np.subtract(x, y), 2)))

# vectorized
def get_pij(ca_pos_src, ca_pos_dst, ncac_map, ncac_origin, ncac_vsize, norm=False, n_sample=10):
    l = len(ca_pos_src)
    m = len(ca_pos_dst)

    # Calculate difference
    vij = ca_pos_src[:, np.newaxis, :] - ca_pos_dst[np.newaxis, :, :] # (L, M, 3)
    norm_arr = np.linalg.norm(vij, axis=2) + 1e-6 # (L, M)
    vij_norm = vij / norm_arr[:, :, np.newaxis] # (L, M, 3)
    delta = norm_arr[:, :, np.newaxis] / n_sample # (L, M)
    p = np.full_like(norm_arr, 1e6)  # Init p (L, M)
    for k in range(n_sample):
        k_pos = ca_pos_dst[np.newaxis, :, :] + (k * delta) * vij_norm # (L, M, 3)
        dens_arr = get_grid_value_batch(grid=ncac_map, coords=(k_pos-ncac_origin) / ncac_vsize)
        p = np.minimum(p, dens_arr, out=p)  # Update p

    return p


# Prepare cost matrix
def prepare_cost_matrix(ca_pos_head, ca_pos_tail, ncac_map, ncac_origin, ncac_vsize, sigma=1.0):
    l = len(ca_pos_tail)
    m = len(ca_pos_head)
    cost = np.zeros((l, m), dtype=np.float32)

    # Calculate pairwise distance
    dij = pairwise_distances(ca_pos_tail, ca_pos_head)
    # Get pij from ncac_map
    pij = get_pij(ca_pos_tail, ca_pos_head, ncac_map, ncac_origin, ncac_vsize, norm=False, n_sample=10)

    condition_1 = np.logical_or(dij > 10.0, pij < 0.1)
    condition_2 = np.logical_not(condition_1)

    # Get final
    cost[condition_1] = (100.0 + 10 * dij)[condition_1]
    cost[condition_2] = (100.0 - 100 * np.exp( -1 * (10 * dij - 38)**2 / 2 * sigma **2 ) * pij)[condition_2]
    
    return cost


def trace(points, cost=None, vehicles=1, lkh='./', time_limit=300):
    # Get abs dir for lkh
    lkh_dir = os.path.abspath(  os.path.expanduser(lkh)  )
    #print("Set lkh path to : ", lkh_dir)
    scale_factor = 10.
    # Make a temperary dir
    with tempfile.TemporaryDirectory() as temp_dir:
        #print("Temporary directory created : ", temp_dir)

        # Set lkh parameters
        L = len(points) + 1
        dmat = np.zeros((L, L), dtype=np.float32)

        if cost is None:
            # Use pairwise distances
            dmat[:L-1, :L-1] = pairwise_distances(points, points) * scale_factor
        else:
            dmat[:L-1, :L-1] = cost

        set_tsp_file(dmat, path=temp_dir)
        set_tsp_params(depot=len(points) + 1, vehicles=vehicles, path=temp_dir, time_limit=time_limit)
        #print("Set up tsp files")

        # Trace on lkh
        run_tsp(vehicles=vehicles, path=temp_dir, lkh_dir=lkh_dir)
        print("# Running lkh")

        # Get result
        with open(os.path.join(temp_dir, 'exp.out'), 'r') as f:
            lines = f.readlines()
        sol = list()
        flag = False
        for line in lines:
            if line.startswith('TOUR_SECTION'):
                flag = True
                continue
            if line.startswith('-1'):
                break
            if flag:
                x = int(line.strip().split()[0])
                sol.append(x)
        out_inds = list()
        all_out_inds = list()
        for i, ind in enumerate(sol):
            if 1 <= ind <= len(points):
                out_inds.append(ind - 1)
            else:
                all_out_inds.append(out_inds)
                out_inds = []
        all_out_inds.append(out_inds)
        all_out_inds = all_out_inds[1:]

    # Return traced points (indices)
    return all_out_inds


def residue_trace(residues, vehicles=1, lkh='./', time_limit=300):
    # Get abs dir for lkh
    lkh_dir = os.path.abspath(  os.path.expanduser(lkh)  )
    scale_factor = 10.
    L = len(residues) + 1
    n_positions = residues[:, 0, :]
    c_positions = residues[:, 2, :]
    dmat = np.zeros((L, L), dtype=np.float32)
    dmat[:L-1, :L-1] = pairwise_distances(c_positions, n_positions) * scale_factor
    # Make a temperary dir
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = './'
        #print("Temporary directory created : ", temp_dir)

        # Set lkh parameters
        set_tsp_file(dmat, path=temp_dir, tsptype='ATSP')
        set_tsp_params(depot=L, vehicles=vehicles, path=temp_dir, time_limit=time_limit)
        #print("Set up tsp files")

        # Trace on lkh
        run_tsp(vehicles=vehicles, path=temp_dir, lkh_dir=lkh_dir)
        print("Running fragment lkh")

        # Get result
        with open(os.path.join(temp_dir, 'exp.out'), 'r') as f:
            lines = f.readlines()

        sol = list()
        flag = False
        for line in lines:
            if line.startswith('TOUR_SECTION'):
                flag = True
                continue
            if line.startswith('-1'):
                break
            if flag:
                x = int(line.strip().split()[0])
                sol.append(x)
        out_inds = list()
        all_out_inds = list()
        for i, ind in enumerate(sol):
            if 1 <= ind <= L - 1:
                out_inds.append(ind - 1)
            else:
                all_out_inds.append(out_inds)
                out_inds = []
        all_out_inds.append(out_inds)
        all_out_inds = all_out_inds[1:]

    # Return traced
    return all_out_inds

# Trace fragment
def fragment_trace(
        frags, 
        cost=None,
        vehicles=1,
        lkh='./',
        ordered=True,
        time_limit=300,
        temp_dir=None,
    ):
    # frags : List[np.ndarray of shape (N, 3, 3)]

    # Special cases
    if len(frags) == 1:
        return [[0]]
    if len(frags) == 2:
        d0 = distance(frags[0][-1][-1], frags[-1][0][0])
        d1 = distance(frags[-1][-1][-1], frags[0][0][0])
        if d0 < d1:
            return [[0, 1]]
        else:
            return [[1, 0]]

    # Get abs dir for lkh
    lkh_dir = os.path.abspath(  os.path.expanduser(lkh)  )
    print("# Set lkh path to ", lkh_dir)

    # Use true distance
    if cost is None:
        print("# Use true distance map")
        # True distance between fragment terminals
        # Pseudo distance between 2 terminal in a fragment
        npoint = len(frags)
        dmat = np.zeros((npoint + 1, npoint + 1), dtype=np.float32)
        scale_factor = 10.
    
        # Only start from head to foot
        for i in range(len(frags)):
            fi = frags[i]
            for j in range(len(frags)):
                fj = frags[j]
                # i to j
                dmat[i, j] = distance(fi[-1][-1], fj[0][0]) * scale_factor
    # Use custimized distance
    else:
        print("# Use customized cost matrix")
        npoint = len(frags)
        dmat = np.zeros((npoint + 1, npoint + 1), dtype=np.float32)
        dmat[:npoint, :npoint] = cost


    # Make a temperary dir
    with tempfile.TemporaryDirectory() as __temp_dir:
        if temp_dir is None:
            temp_dir = __temp_dir

        print("# Set lkh temp dir to ", temp_dir)

        #temp_dir = './'
        #print("# Temporary directory created at ", temp_dir)

        # Set lkh parameters
        set_tsp_file(dmat, path=temp_dir, tsptype='ATSP')

        set_tsp_params(depot=npoint + 1, vehicles=vehicles, path=temp_dir, time_limit=time_limit)
        #print("Set up tsp files")

        # Trace on lkh
        run_tsp(vehicles=vehicles, path=temp_dir, lkh_dir=lkh_dir)
        print("# Running fragment lkh")

        # Get result
        with open(os.path.join(temp_dir, 'exp.out'), 'r') as f:
            lines = f.readlines()

        sol = list()
        flag = False
        for line in lines:
            if line.startswith('TOUR_SECTION'):
                flag = True
                continue
            if line.startswith('-1'):
                break
            if flag:
                x = int(line.strip().split()[0])
                sol.append(x)
        out_inds = list()
        all_out_inds = list()
        for i, ind in enumerate(sol):
            if 1 <= ind <= npoint:
                out_inds.append(ind - 1)
            else:
                all_out_inds.append(out_inds)
                out_inds = []
        all_out_inds.append(out_inds)
        all_out_inds = all_out_inds[1:]

    # Return traced
    return all_out_inds



def split_chain(coords, d=5.0):
    chains = []
    last = -1
    for k in range(len(coords) - 1):
        d0 = distance(coords[k], coords[k+1])
        if d0 > d:
            chain = []
            for kk in range(last + 1, k + 1):
                chain.append(kk)

            if len(chain) > 0:
                chains.append(chain)
            last = k

    chain = []
    for kk in range(last + 1, len(coords)):
        chain.append(kk)
    if len(chain) > 0:
        chains.append(chain)
    return chains



def main(args):
    ts = time.time()

    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)

    fpdb = args.pdb
    lib_dir = args.lib
    lib_dir = abspath(lib_dir)
    out_dir = args.out
    out_dir = abspath(out_dir)
    lkh_lib_dir = lib_dir + "/bin/trace"
    lkh_lib_dir = abspath(lkh_lib_dir)

    # Read map
    fncacmap = args.mcmap 
    ncac_map, ncac_origin, _, ncac_vsize = parse_map(fncacmap, False, None)
    # Normalize
    pmax = np.percentile(ncac_map, 99.5)
    ncac_map = np.clip(ncac_map, 0.0, pmax)
    ncac_map = ncac_map / (pmax + 1e-6)

    # Read N-Ca-C
    atom3_pos, _, _, _, _ = read_pdb(fpdb)
    atom3_pos = atom3_pos[..., :3, :]
    print("# Done reading {} CAs".format(len(atom3_pos)))

    # Thread individual residues into fragments
    atom3_mask = np.ones( atom3_pos.shape[:-1], dtype=bool )
    chains = thread_and_merge_ncac(
        atom3_pos,
    )
    fchains = pjoin(out_dir, "init_frags.cif")
    chains_atom_pos_to_pdb(
        filename=fchains,
        chains_atom_pos=[atom3_pos[chain] for chain in chains],
        chains_atom_mask=[atom3_mask[chain] for chain in chains],
        suffix='cif',
    )
    print("# Done initial threading")
    print("# Write initial chains to {}".format(fchains))






    # Read initial chains
    atom14_pos, atom14_mask, _, _, chain_idx = read_pdb(fchains, keep_valid=True)
    coords = atom14_pos[..., 1, :]
    print("# Done reading {} CAs".format(len(coords)))

    chains = []
    n_max = np.max(chain_idx) + 1
    idxs = np.arange(len(coords), dtype=np.int32)
    for i in range(n_max):
        chains.append(idxs[ chain_idx == i ])
    print("# Split to {:>4d} chains".format(len(chains)))

    # Calculate cost matrix
    print("# Calculating cost matrix")
    ca_pos_tail = []
    ca_pos_head = []
    frag_coords = []
    for chain in chains:
        ca_pos_tail.append(atom14_pos[chain][-1][1][None])
        ca_pos_head.append(atom14_pos[chain][ 0][1][None])
        frag_coords.append(atom14_pos[chain][..., 1, :][..., None, :]) # (N, 1, 3)
        #ca_pos_tail.append(coords[chain][-1][None])
        #ca_pos_head.append(coords[chain][ 0][None])
        #frag_coords.append(coords[chain][..., None, :]) # (N, 1, 3)

    ca_pos_head = np.concatenate(ca_pos_head, axis=0)
    ca_pos_tail = np.concatenate(ca_pos_tail, axis=0)

    cost = prepare_cost_matrix(ca_pos_head, ca_pos_tail, ncac_map=ncac_map, ncac_origin=ncac_origin, ncac_vsize=ncac_vsize)

    # TSP Trace
    #vehicles = [1, 5, 10, 20]
    #vehicles = [1, 5, 10] 
    #vehicles = [1] 

    if args.vehicle is not None:
        vehicles = [ args.vehicle ]
        time_limit = 1800
    else:
        n_res_per_vehicle = 300
        n_trial = int(math.ceil( len(coords) / n_res_per_vehicle))
        max_trial = 30
        if n_trial > max_trial:
            n_trial = max_trial
        vehicles = [ n_trial ]

        if n_trial < int(max_trial * 1 / 3):
            time_limit = 200
        elif n_trial < int(max_trial * 2 / 3):
            time_limit = 400
        else:
            time_limit = 600

    print("# Routing with vehicles = {}".format(vehicles))

    print("# Start TSP routing...")
    routes = []
    for iv, v in enumerate(vehicles):
        tstart = time.time()
        chains0 = []
        start = time.time()
        print("# Routing with n vehicle = {} with time limit = {:.4f} seconds".format(v, time_limit))

        # temp dir
        temp_dir = os.path.join(args.out, "route_{}".format(iv))
        os.makedirs(temp_dir, exist_ok=True)

        #inds = trace(coords, cost=cost, vehicles=v, lkh=lkh_lib_dir, time_limit=time_limit)
        inds = fragment_trace(frag_coords, cost=cost, vehicles=v, lkh=lkh_lib_dir, time_limit=time_limit, temp_dir=temp_dir)

        end = time.time()
        print("# Routing finished with {:.4f} seconds".format(end-start))
        print("# Routing report")
        for k in range(0, v):
            chains0.append(
                np.concatenate([chains[ind] for ind in inds[k]], axis=0, )
            )
            print("# Path {} / {} has {} CAs".format(k+1, v, len(chains0[k])))

        # Get ordered traces
        routes.append(chains0)

    # Update traced chains
    chains = chains0

    # For each routes, split at large break and determine the chain direction
    for r, chains in enumerate(routes):
        new_chains = []
        for chain in chains:
            idxs = split_chain(coords[chain], d=4.5)
            for idx in idxs:
                new_chains.append([chain[i] for i in idx])

        frags = new_chains

        # Write
        frags_ca_pos = [coords[idxs] for idxs in frags if len(idxs) > 10]

        frags_ca_pos.sort(key=lambda x:len(x), reverse=True)

        frags_atom3_pos = []
        frags_atom3_mask = []
        for ca_pos in frags_ca_pos:
            pos, mask = ca_to_atom3(ca_pos)
            frags_atom3_pos.append(pos)
            frags_atom3_mask.append(mask)

        ffrag = pjoin(out_dir, f"frag_{r}.cif")
        chains_atom_pos_to_pdb(
            ffrag,
            frags_atom3_pos,
            frags_atom3_mask,
            suffix='cif',
        )
        print("# Write traced chains to {}".format(ffrag))


    te = time.time()
    print("# Time consuming {:.4f}".format(te-ts))


if __name__ == '__main__':
    # If executed directly
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", "-p", type=str, help="Input coords in .pdb format")
    parser.add_argument("--mcmap", "-mcmap", type=str, help="Input main-chain map")
    parser.add_argument("--lib", "-l", type=str, help="Directory to lib")
    parser.add_argument("--vehicle", "-v", type=int, help="Input vehicles")
    parser.add_argument("--out", "-o", type=str, default="./", help="Output dir")
    args = parser.parse_args()
    main(args)
