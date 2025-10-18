import numpy as np
from scipy.spatial import KDTree

def filter_with_distance(coords, density, r0=2.):
    max_iter = 10
    n_iter = 0
    while n_iter < max_iter:
        tree = KDTree(coords)
        keep = [False for _ in range(len(coords))]
        for i in range(len(coords)):
            inds = tree.query_ball_point(coords[i], r0)
            dens = density[inds]

            ii = np.argmax(dens)
            ii = inds[ii]
            keep[ii] = True
        if np.all(keep):
            break

        coords = coords[keep]
        density = density[keep]

        n_iter += 1

    return coords

def main(argv):
    fp = argv[1]
    pass

if __name__ == '__main__':
    import sys
    main(sys.argv)
