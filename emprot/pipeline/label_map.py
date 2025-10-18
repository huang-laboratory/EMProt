import numpy as np

from emprot.io.pdbio import read_pdb_simple
from emprot.utils.grid import label_map
from emprot.utils.cryo_utils import parse_map, write_map

def main(args):
    assert args.pdb is not None and len(args.pdb) > 0, "# No PDBs are found"
    coords = []
    probs = []

    # read input coords
    for fp in args.pdb:
        atom_pos, atom_prob = read_pdb_simple(fp, res_type=False, bfactor=True)
        coords.append(atom_pos)
        probs.append(atom_prob)

    coords = np.concatenate(coords, axis=0)
    probs  = np.concatenate(probs, axis=0)

    print("# Read {} coords".format(len(coords)))
    assert np.all(np.logical_not(np.isnan(coords))), "# Input coords have NaN"

    # read reference map
    apix = 1.0
    grid, origin, nxyz, vsize = parse_map(args.map, False, apix)
    del grid

    # label map by CA position and density
    grid = label_map(origin, nxyz, apix, coords, probs, args.resolution)

    write_map(args.output, grid.astype(np.float32), origin=origin, voxel_size=[apix, apix, apix])
    print("# Write labeled map to {}".format(args.output))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", "-m", required=True, help="Reference map")
    parser.add_argument("--pdb", "-p", required=True, nargs='+', help="Input coords")
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--resolution", "-r", type=float, default=4.0)
    args = parser.parse_args()
    main(args)

