import numpy as np
from scipy.spatial import KDTree

def clash_ratio(a, b, r_clash=1.8):
    # a, b of shape (L, 3), (M, 3)
    # determine the ratio of clashed atoms
    # if a pair of atoms are within distance <= r_clash, they are clashed

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    assert (a.ndim == 2 and b.ndim == 2)
    atree = KDTree(a)
    btree = KDTree(b)

    idxs_a_in_b = atree.query_ball_point(b, r=r_clash)
    idxs_b_in_a = btree.query_ball_point(a, r=r_clash)

    idxs_a_in_b = np.concatenate(idxs_a_in_b, axis=0).astype(np.int32)
    idxs_b_in_a = np.concatenate(idxs_b_in_a, axis=0).astype(np.int32)

    # del repeated
    idxs_a_in_b = np.unique(idxs_a_in_b)
    idxs_b_in_a = np.unique(idxs_b_in_a)

    ra = len(idxs_a_in_b) / len(a)
    rb = len(idxs_b_in_a) / len(b)

    return ra, rb


def clash_flag(a, b, r_clash=1.8):
    # a, b of shape (L, 3), (M, 3)
    # determine the ratio of clashed atoms
    # if a pair of atoms are within distance <= r_clash, they are clashed

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    assert (a.ndim == 2 and b.ndim == 2)
    atree = KDTree(a)
    btree = KDTree(b)

    idxs_a_in_b = atree.query_ball_point(b, r=r_clash)
    idxs_b_in_a = btree.query_ball_point(a, r=r_clash)

    idxs_a_in_b = np.concatenate(idxs_a_in_b, axis=0).astype(np.int32)
    idxs_b_in_a = np.concatenate(idxs_b_in_a, axis=0).astype(np.int32)

    # del repeated
    idxs_a_in_b = np.unique(idxs_a_in_b)
    idxs_b_in_a = np.unique(idxs_b_in_a)

    ia = np.zeros(len(a), dtype=np.int32)
    ib = np.zeros(len(b), dtype=np.int32)

    ia[idxs_a_in_b] = 1
    ib[idxs_b_in_a] = 1

    return ia, ib



if __name__ == '__main__':
    a = np.random.rand(256, 3)
    b = np.random.rand(384, 3) + 1.4

    cr = clash_ratio(a, b)
    print(cr)

    ia, ib = clash_flag(a, b)
    print(ia)
    print(ib)
