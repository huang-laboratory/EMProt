import numpy as np
from collections import OrderedDict


from emprot.io.fileio import getlines
from emprot.io.pdbio import write_atoms_as_pdb

def main(args):
    fpdb = args.pdb
    fmat = args.matrix

    # get the transformation matrix
    lines = getlines(fmat)
    seg_Ts = OrderedDict()
    for line in lines:
        if line.startswith("REMARK chain"):
            fields = line.strip().split()

            chain_idx = int(fields[2])
            seg_idx = int(fields[4])
            label = "{}_{}".format(chain_idx, seg_idx)

            center = fields[6:9]
            if center[0] == '-':
                seg_Ts[label] = None
                continue

            center = [float(x) for x in center]
            center = np.asarray(center, dtype=np.float32) # (3, )

            R = [float(x) for x in fields[10:19]] # (9, )
            t = [float(x) for x in fields[20:23]] # (3, )
            R = np.asarray(R, dtype=np.float32).reshape(3, 3)
            t = np.asarray(t, dtype=np.float32)

            seg_Ts[label] = (center, R, t)


    # read pdb
    lines = getlines(fpdb)
    seg_coords = OrderedDict()
    seg_lines = OrderedDict()
    for line in lines:
        if line.startswith("ATOM") and len(line) > 76:
            if line[12:16] != " CA ":
                continue

            chain_idx = int(line[74:76])
            seg_idx = int(line[70:74])
            label = "{}_{}".format(chain_idx, seg_idx)

            xyz = [float(x) for x in [line[i:i+8] for i in [30, 38, 46]]]
            seg_coords.setdefault(label, [])
            seg_coords[label].append(xyz)

            seg_lines.setdefault(label, [])
            seg_lines[label].append(line[:-1])

    for k, v in seg_coords.items():
        seg_coords[k] = np.asarray(v, dtype=np.float32)

    transformed_seg_coords = dict()
    for label, T in seg_Ts.items():
        center, R, t = T
        transformed_seg_coords[label] = (seg_coords[label] - center) @ R.T + t + center
   
    for i, (label, coords) in enumerate(transformed_seg_coords.items()):
        write_atoms_as_pdb("seg_{}.pdb".format(i), coords, ter=False, final_ter=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", "-p", help="PDB file")
    parser.add_argument("--matrix", "-m", help="Transformation matrix")
    args = parser.parse_args()
    main(args)
