import numpy as np
from scipy.spatial import cKDTree

from emprot.utils.cryo_utils import parse_map, enlarge_grid
from emprot.io.pdbio import write_atoms_as_pdb

def get_lattice_meshgrid_np(shape, no_shift=False):
    linspace = np.linspace(
        0.5 if not no_shift else 0, shape - (0.5 if not no_shift else 1), shape,
    )
    mesh = np.stack(np.meshgrid(linspace, linspace, linspace, indexing="ij"), axis=-1,)
    return mesh

def grid_to_points(
    grid, threshold, neighbour_distance_threshold, prune_distance=1.1,
):
    lattice = np.flip(get_lattice_meshgrid_np(grid.shape[-1], no_shift=True), -1)

    output_points_before_pruning = np.copy(lattice[grid > threshold, :].reshape(-1, 3))

    points = lattice[grid > threshold, :].reshape(-1, 3)
    probs = grid[grid > threshold]

    for _ in range(3):
        kdtree = cKDTree(np.copy(points))
        n = 0

        new_points = np.copy(points)
        for p in points:
            neighbours = kdtree.query_ball_point(p, prune_distance)
            selection = list(neighbours)
            if len(neighbours) > 1 and np.sum(probs[selection]) > 0:
                keep_idx = np.argmax(probs[selection])
                prob_sum = np.sum(probs[selection])

                new_points[selection[keep_idx]] = (
                    np.sum(probs[selection][..., None] * points[selection], axis=0)
                    / prob_sum
                )
                probs[selection] = 0
                probs[selection[keep_idx]] = prob_sum

            n += 1

        points = new_points[probs > 0].reshape(-1, 3)
        probs = probs[probs > 0]

    kdtree = cKDTree(np.copy(points))
    for point_idx, point in enumerate(points):
        d, _ = kdtree.query(point, 2)
        if d[1] > neighbour_distance_threshold:
            points[point_idx] = np.nan

    points = points[~np.isnan(points).any(axis=-1)].reshape(-1, 3)

    output_points = points
    return output_points, output_points_before_pruning



def main(args):
    data, origin, nxyz, vsize = parse_map(args.map, False, None)
    data = enlarge_grid(data)
    
    maximum = np.percentile(data, 99.999)
    data = np.clip(data, 0.0, maximum)
    data = data / (data.max() + 1e-6)

    points, _ = grid_to_points(data, args.ratio, 6.0, prune_distance=1.6)

    write_atoms_as_pdb(args.output, points + origin)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", "-m", help="Input map", type=str)
    parser.add_argument("--ratio", "-r", help="Lower bound ratio", type=float, default=0.10)
    parser.add_argument("--output", "-o", help="Output file", type=str, default="output.pdb")
    args = parser.parse_args()
    main(args)

