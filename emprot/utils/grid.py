import math
import numpy as np
import torch
from scipy.interpolate import interpn

def get_grid_value(grid : np.ndarray, coord : np.ndarray, origin=None, r_voting=None):
    if origin is not None:
        coord = np.subtract(coord, origin)
    x, y, z = [int(x) for x in coord]

    if grid.ndim == 3:
        n0, n1, n2 = grid.shape
        if x >= n2 or x < 0 or \
           y >= n1 or y < 0 or \
           z >= n0 or z < 0:
            return 0.0
        else:
            return grid[z, y, x]
    elif grid.ndim == 4:
        c, n0, n1, n2 = grid.shape
        if x >= n2 or x < 0 or \
           y >= n1 or y < 0 or \
           z >= n0 or z < 0:
            return np.full((20,), -2.95, dtype=np.float32)
        else:
            return grid[:, z, y, x]
    else:
        raise "Error grid.ndim must be 3 or 4"

def get_grid_value_interp(grid : np.ndarray, coord : np.ndarray, origin=None):
    n0, n1, n2 = grid.shape

    if origin is not None:
        coord = np.subtract(coord, origin)

    z = 0.0

    kx0 = int(coord[0])
    kx1 = kx0 + 1

    ky0 = int(coord[1])
    ky1 = ky0 + 1

    kz0 = int(coord[2])
    kz1 = kz0 + 1

    if  kx0 < 0 or kx0 >= n2 or \
        kx1 < 0 or kx1 >= n2 or \
        ky0 < 0 or ky0 >= n1 or \
        ky1 < 0 or ky1 >= n1 or \
        kz0 < 0 or kz0 >= n0 or \
        kz1 < 0 or kz1 >= n0:
        return 0.0

    x0 = coord[0] - kx0
    y0 = coord[1] - ky0
    z0 = coord[2] - kz0

    txy = x0*y0
    tyz = y0*z0
    txz = x0*z0
    txyz = x0*y0*z0

    # z, y, x
    v000 = grid[kz0, ky0, kx0]
    v100 = grid[kz1, ky0, kx0]
    v010 = grid[kz0, ky1, kx0]
    v001 = grid[kz0, ky0, kx1]
    v101 = grid[kz1, ky0, kx1]
    v011 = grid[kz0, ky1, kx1]
    v110 = grid[kz1, ky1, kx0]
    v111 = grid[kz1, ky1, kx1]

    temp1 = v000*(1.0-x0-y0-z0+txy+tyz+txz-txyz);
    temp2 = v100*(x0-txy-txz+txyz);
    temp3 = v010*(y0-txy-tyz+txyz);
    temp4 = v001*(z0-txz-tyz+txyz);
    temp5 = v101*(txz-txyz);
    temp6 = v011*(tyz-txyz);
    temp7 = v110*(txy-txyz);
    temp8 = v111*txyz;

    z = temp1 + temp2 + temp3 + temp4 + temp5 + temp6 + temp7 + temp8

    return z


def get_grid_value_batch(grid : np.ndarray, coords : np.ndarray):
    coords = np.asarray(coords, dtype=np.int32)
    n0, n1, n2 = grid.shape[-3:]
    coords[..., 0] = np.clip(coords[..., 0], 0, n2)
    coords[..., 1] = np.clip(coords[..., 1], 0, n1)
    coords[..., 2] = np.clip(coords[..., 2], 0, n0)
    values = grid[..., coords[..., 2], coords[..., 1], coords[..., 0]]
    return values

# Get grid value using scipy.interpolate.interpn
def grid_value_interp(points, grid, origin=None, vsize=None):
    # grid (n0, n1, n2)
    # points (..., 3)

    if origin is None:
        origin = np.zeros(3, dtype=np.float32)
    if vsize is None:
        vsize = np.ones(3, dtype=np.float32)

    n0 = np.arange(grid.shape[0])
    n1 = np.arange(grid.shape[1])
    n2 = np.arange(grid.shape[2])
    # Transform to grid
    p = (points - origin) / vsize
    # Flip last dim
    p = np.flip(p, axis=-1)

    # Boundary check
    p[..., 0] = np.clip(p[..., 0], 0.0, grid.shape[0] - 1)
    p[..., 1] = np.clip(p[..., 1], 0.0, grid.shape[1] - 1)
    p[..., 2] = np.clip(p[..., 2], 0.0, grid.shape[2] - 1)

    #print(p)

    values = interpn(
        (n0, n1, n2), grid,
        p,
        method='linear',
        #method='nearest',
        bounds_error=False, fill_value=0.0
    )
    return values.astype(np.float32)


# pytorch version fully differentiable
def grid_value_sample(points, grid, origin=None, vsize=None):
    device = grid.device
    assert points.device == device

    if origin is None:
        origin = torch.zeros((3, ), dtype=torch.float32, device=device)
    if vsize is None:
        vsize = torch.ones((3, ), dtype=torch.float32, device=device)

    points = (points - origin) / vsize # (N, 3)
    points = torch.flip(points, dims=[-1])

    n0, n1, n2 = grid.shape

    clipped_points = points.clone()

    clipped_points[..., 0] = torch.clip(points[..., 0], min=0, max=n0-1)
    clipped_points[..., 1] = torch.clip(points[..., 1], min=0, max=n1-1)
    clipped_points[..., 2] = torch.clip(points[..., 2], min=0, max=n2-1)

    rescaled = torch.zeros_like(clipped_points, dtype=torch.float32, device=device) # (N, 3)
    rescaled[..., 0] = 2 * clipped_points[..., 0] / (n0 - 1) - 1
    rescaled[..., 1] = 2 * clipped_points[..., 1] / (n1 - 1) - 1
    rescaled[..., 2] = 2 * clipped_points[..., 2] / (n2 - 1) - 1

    # input (N, H_out, W_out, D_out, 3)
    # grid  (N, C, H_in, W_in, D_in)
    # out   (N, C, H_out, W_out, D_out)
    # using pytorch grid_sample
    # transform input ranges to [-1, 1] such that
    # (-1, -1, -1) -> (0, 0, 0)
    # ( 1,  1,  1) -> (n0-1, n1-1, n2-1)

    # [0, n-1] -> [-1,  1] using y = (2 * x / (n-1)) - 1
    # [-1,  1] -> [0, n-1] using x = (y + 1) * (n-1) / 2

    values = torch.nn.functional.grid_sample(
        # permute(2, 1, 0)
        input=grid.permute(2, 1, 0)[None, None, ...].expand(len(rescaled), -1, -1, -1, -1), # (N, 1, ...)
        grid=rescaled[..., None, None, None, :], # (N, 1, 1, 1, 3)
        padding_mode='zeros',
        mode='bilinear',
        #mode='nearest',
        align_corners=True,
    ) # (N, 1, 1, 1, 1)

    return values[:, 0, 0, 0, 0] # (N, )







from numba import jit
@jit(nopython=True)
def label_map(origin, nxyz, apix, coords, dens, resolution):
    # equavalent resolution
    #k = (np.pi / (2.4 + 0.8 * resolution)) ** 2
    k = (np.pi / resolution) ** 2
    C = (k / np.pi) ** 1.5
    bw2 = 6.0 / k
    bw = np.sqrt(bw2)
    label_map = np.full((nxyz[2], nxyz[1], nxyz[0]), 0.0, dtype=np.float32)

    for i, coord in enumerate(coords):
        coord_shifted = (coord - origin) / apix
        coord_lower = np.floor(coord_shifted - bw).astype(np.int32)
        coord_upper =  np.ceil(coord_shifted + bw).astype(np.int32)
        for x in range(coord_lower[0], coord_upper[0]):
            for y in range(coord_lower[1], coord_upper[1]):
                for z in range(coord_lower[2], coord_upper[2]):
                    if nxyz[0] > x >= 0 and nxyz[1] > y >= 0 and nxyz[2] > z >= 0:
                        distance = np.array([x, y, z], dtype=np.float32) - coord_shifted
                        distance2 = np.dot(distance, distance)
                        if distance2 < bw2:
                            prob = C * np.exp(-k * distance2) * dens[i]
                            label_map[z, y, x] = max(label_map[z, y, x], prob * 1.0)
    label_map /= C
    return label_map




# pytorch version
def label_map_torch(origin, nxyz, apix, coords, dens, resolution=5.0):
    device = coords.device
    # init output map
    out_map = torch.zeros((nxyz[0], nxyz[1], nxyz[2]), dtype=torch.float32).to(device)

    # constants
    pi = 3.1415926535
    k = (pi / resolution) ** 2
    C = (k / pi) ** 1.5
    bw2 = 6.0 / k
    bw = math.sqrt(bw2)

    # calculate potential map
    l = 7
    s = l // 2
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(-s, s + 1), torch.arange(-s, s + 1), torch.arange(-s, s + 1),
            indexing='ij'
        ), 
        dim=-1
    ).reshape(-1, 3).to(device)

    #print(grid)
    for offset in grid:
        displaced_coords = coords.long() + offset
        #d2 = torch.sum((coords[:, None, :] - displaced_coords[None, :, :]) ** 2, dim=-1)
        d2 = torch.sum((coords - displaced_coords) ** 2, dim=-1)
        exp_term = torch.exp(-k * d2)

        # Ensure indices are within bounds
        displaced_coords = ((displaced_coords - origin) / apix).long()
        #displaced_coords = torch.round(displaced_coords).long()

        displaced_coords = torch.clamp(displaced_coords, torch.tensor(0), torch.tensor(out_map.shape)-1)

        # Create indices for `scatter_add`
        linear_indices = displaced_coords[:, 0] * out_map.size(1) * out_map.size(2) + \
                         displaced_coords[:, 1] * out_map.size(2) + \
                         displaced_coords[:, 2]

        # Sum contributions using scatter_add
        out_map.view(-1).scatter_add_(0, linear_indices, exp_term.view(-1))
    out_map = out_map.permute(2, 1, 0)
    return out_map


def gen_mask_torch(origin, nxyz, apix, coords, dens, resolution=5.0):
    device = coords.device
    # init output map
    out_map = torch.zeros((nxyz[0], nxyz[1], nxyz[2]), dtype=torch.float32).to(device)

    # constants
    pi = 3.1415926535
    k = (pi / resolution) ** 2
    C = (k / pi) ** 1.5
    bw2 = 6.0 / k
    bw = math.sqrt(bw2)

    # calculate potential map
    l = 7
    s = l // 2
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(-s, s + 1), torch.arange(-s, s + 1), torch.arange(-s, s + 1),
            indexing='ij'
        ), 
        dim=-1
    ).reshape(-1, 3).to(device)

    #print(grid)
    for offset in grid:
        displaced_coords = coords.long() + offset
        #d2 = torch.sum((coords[:, None, :] - displaced_coords[None, :, :]) ** 2, dim=-1)
        d2 = torch.sum((coords - displaced_coords) ** 2, dim=-1)
        #exp_term = torch.exp(-k * d2)
        exp_term = torch.ones_like(d2).float()

        # Ensure indices are within bounds
        displaced_coords = ((displaced_coords - origin) / apix).long()
        #displaced_coords = torch.round(displaced_coords).long()

        displaced_coords = torch.clamp(displaced_coords, torch.tensor(0), torch.tensor(out_map.shape)-1)

        # Create indices for `scatter_add`
        linear_indices = displaced_coords[:, 0] * out_map.size(1) * out_map.size(2) + \
                         displaced_coords[:, 1] * out_map.size(2) + \
                         displaced_coords[:, 2]

        # Sum contributions using scatter_add
        out_map.view(-1).scatter_add_(0, linear_indices, exp_term.view(-1))

    out_map = out_map.permute(2, 1, 0)
    out_map[out_map > 0.0] = 1.0
    return out_map




if __name__ == '__main__':
    #grid = np.random.rand(2, 10, 10, 10)
    #coords = np.random.randint(0, 10, (5, 4, 3))
    #values = get_grid_value_batch(grid, coords)
    #print(values.shape)
    #print(values)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", "-p")
    parser.add_argument("--map", "-m")
    args = parser.parse_args()
    from emprot.utils.cryo_utils import parse_map, write_map
    from emprot.io.pdbio import read_pdb

    apix = 1.0
    _, origin, nxyz, vsize = parse_map(args.map, False, None)
    
    atom_pos, _, _, _, _ = read_pdb(args.pdb)
    coords = atom_pos[..., :3, :].reshape(-1, 3)[:2000]

    import time
    ts = time.time()
    #grid = label_map_differentiable(
    grid = gen_mask_differentiable(
        origin=torch.from_numpy(origin).float(),
        nxyz=nxyz, 
        apix=apix, 
        coords=torch.from_numpy(coords).float().requires_grad_(True), 
        dens=None, 
        resolution=5.0,
    )
    a = grid.mean()
    #a.backward()
    te = time.time()
    print("{:.4f}".format(te-ts))

    write_map("test.mrc", grid.cpu().detach().numpy().astype(np.float32), origin=origin, voxel_size=[apix, apix, apix])

