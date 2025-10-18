import re
import pydssp
import numpy as np

sec_type_1_to_index = {
    "H": 0,
    "E": 1,
    "-": 2,
}

def assign(atom_pos):
    """
        atom_pos: (L, 4, 3) of N, Ca, C, O coords or 
                  (L, 14/23, 3) of all coords
    """
    atom4_pos = atom_pos[..., :4, :]
    assert np.all(np.logical_not(np.isnan(atom4_pos))), "# Input coords contain NaN"

    result = pydssp.assign(atom4_pos, out_type='c3')

    sec_type = [sec_type_1_to_index[x] for x in result]
    sec_type = np.asarray(sec_type, dtype=np.int32)

    return sec_type


def get_hbond_map(atom_pos):    
    """
        atom_pos: (L, 4, 3) of N, Ca, C, O coords or 
                  (L, 14/23, 3) of all coords
    """
    atom4_pos = atom_pos[..., :4, :]
    assert np.all(np.logical_not(np.isnan(atom4_pos))), "# Input coords contain NaN"
    result = pydssp.get_hbond_map(atom4_pos)
    return result


def segment_secstr(data):
    if len(data) == 0:
        return []

    current_value = data[0]
    start_index = 0
    n_frag = 0
    l = len(data)
    ret = [0] * l
    for i in range(1, len(data)):
        if data[i] != current_value:
            for k in range(start_index, i):
                ret[k] = n_frag
            current_value = data[i]
            start_index = i
            n_frag += 1

    for k in range(start_index, len(data)):
        ret[k] = n_frag
    return ret



def update_partition(a: list, b: list):
    """
        a: sec idx
    """
    a = "".join([str(x) for x in a])
    # find all substr of '2'
    matches = list(re.finditer(r"2+", a))

    for match in matches:
        start, end = match.start(), match.end()
        mid = start + (end - start) // 2  # mid point

        left_partition = b[start - 1] if start > 0 else None
        right_partition = b[end] if end < len(a) else None

        if start == 0:
            update_partition = right_partition
            for i in range(start, end):
                b[i] = update_partition
        elif end == len(a):
            update_partition = left_partition
            for i in range(start, end):
                b[i] = update_partition
        else:
            for i in range(start, mid):
                if left_partition:
                    b[i] = left_partition
            for i in range(mid, end):
                if right_partition:
                    b[i] = right_partition

    return b

def remap_partition(values):
    unique_values = sorted(set(values))
    value_to_new_id = {value: idx for idx, value in enumerate(unique_values)}
    remapped_values = [value_to_new_id[value] for value in values]
    return remapped_values



if __name__ == '__main__':
    import sys
    from emprot.io.pdbio import read_pdb
    atom_pos, _, _, _, _ = read_pdb(sys.argv[1], keep_valid=True)
    sec_idx = assign(atom_pos)
    sec_idx_str = "".join([str(x) for x in sec_idx])
    print(sec_idx_str)

    frag_idx = segment_secstr(sec_idx)
    print(frag_idx)
    frag_idx = update_partition(sec_idx, frag_idx)
    print(frag_idx)
    frag_idx = remap_partition(frag_idx)
    frag_idx = np.asarray(frag_idx, dtype=np.int32)

    print(frag_idx)
