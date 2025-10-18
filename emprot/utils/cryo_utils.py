import os
import sys
import mrcfile
import warnings
import numpy as np
from math import floor, ceil
from collections import namedtuple
from emprot.interp.interp3d import Interp3d

'''parse_map, pad_map, split_map_into_overlapped_chunks, get_map_from_overlapped_chunks, write_map'''
def split_map_into_overlapped_chunks(map, box_size, stride, dtype=np.float32, padding=0.0):
    assert stride < box_size
    map_shape = np.shape(map)
    padded_map = np.full((map_shape[0] + 2 * box_size, map_shape[1] + 2 * box_size, map_shape[2] + 2 * box_size), padding, dtype=dtype)
    padded_map[box_size : box_size + map_shape[0], box_size : box_size + map_shape[1], box_size : box_size + map_shape[2]] = map
    chunk_list = list()
    start_point = box_size - stride
    cur_x, cur_y, cur_z = start_point, start_point, start_point
    while (cur_z + stride < map_shape[2] + box_size):
        next_chunk = padded_map[cur_x:cur_x + box_size, cur_y:cur_y + box_size, cur_z:cur_z + box_size]
        cur_x += stride
        if (cur_x + stride >= map_shape[0] + box_size):
            cur_y += stride
            cur_x = start_point # Reset
            if (cur_y + stride  >= map_shape[1] + box_size):
                cur_z += stride
                cur_y = start_point # Reset
                cur_x = start_point # Reset
        chunk_list.append(next_chunk)
    n_chunks = len(chunk_list)
    ncx, ncy, ncz = [ceil(map_shape[i] / stride) for i in range(3)]
    assert(n_chunks == ncx * ncy * ncz)
    chunks = np.asarray(chunk_list, dtype=dtype)
    return chunks, ncx, ncy, ncz

# old version
#def pad_map(map, box_size, core_size, dtype=np.float32, padding=0):
#    map_shape = np.shape(map)
#    ncx, ncy, ncz = [ceil(map_shape[i] / core_size) for i in range(3)]
#    padded_map = np.full((ncx * core_size, ncy * core_size, ncz * core_size), padding, dtype=dtype)
#    padded_map[:map_shape[0], :map_shape[1], :map_shape[2]] = map
#    return padded_map
    
# new version, same as the first several lines in 'split_map_into_overlapped_chunks'
def pad_map(map, box_size, dtype=np.float32, padding=0.0):
    map_shape = np.shape(map)
    padded_map = np.full((map_shape[0] + 2 * box_size, map_shape[1] + 2 * box_size, map_shape[2] + 2 * box_size), padding, dtype=dtype)
    padded_map[box_size : box_size + map_shape[0], box_size : box_size + map_shape[1], box_size : box_size + map_shape[2]] = map
    return padded_map

def get_map_from_overlapped_chunks(chunks, ncx, ncy, ncz, n_classes, box_size, stride, nxyz, dtype=np.float32, crop=0):
    assert stride > crop
    map = np.zeros((n_classes, \
                    (ncx - 1) * stride + box_size, \
                    (ncy - 1) * stride + box_size, \
                    (ncz - 1) * stride + box_size), dtype=dtype)
    denominator = np.zeros((n_classes, \
                            (ncx - 1) * stride + box_size, \
                            (ncy - 1) * stride + box_size, \
                            (ncz - 1) * stride + box_size), dtype=dtype) # should clip to 1
    i = 0
    for z_steps in range(ncz):
        for y_steps in range(ncy):
            for x_steps in range(ncx):
                if crop > 0:
                    # Crop chunk
                    chunk0 = chunks[i]
                    chunk = np.zeros_like(chunk0, dtype=dtype)
                    chunk[crop : box_size - crop, 
                          crop : box_size - crop, 
                          crop : box_size - crop] = chunk0[crop : box_size - crop, crop : box_size - crop, crop : box_size - crop]
                else:
                    chunk = chunks[i]

                map[:, x_steps * stride : x_steps * stride + box_size,
                       y_steps * stride : y_steps * stride + box_size,
                       z_steps * stride : z_steps * stride + box_size] += chunk
                denominator[:, x_steps * stride : x_steps * stride + box_size,
                               y_steps * stride : y_steps * stride + box_size,
                               z_steps * stride : z_steps * stride + box_size] += 1

                i += 1
    return (map / denominator.clip(min=1))[:, stride : nxyz[2] + stride, stride : nxyz[1] + stride, stride : nxyz[0] + stride]

def parse_map(map_file, ignorestart, apix=None, origin_shift=None):

    ''' parse mrc '''
    mrc = mrcfile.open(map_file, mode='r')

    map = np.asfarray(mrc.data.copy(), dtype=np.float32)
    voxel_size = np.asarray([mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z], dtype=np.float32)
    ncrsstart = np.asarray([mrc.header.nxstart, mrc.header.nystart, mrc.header.nzstart], dtype=np.float32)
    origin = np.asarray([mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z], dtype=np.float32)
    ncrs = (mrc.header.nx, mrc.header.ny, mrc.header.nz)
    angle = np.asarray([mrc.header.cellb.alpha, mrc.header.cellb.beta, mrc.header.cellb.gamma], dtype=np.float32)

    ''' check orthogonal '''
    try:
        assert(angle[0] == angle[1] == angle[2] == 90.0)
    except AssertionError:
        print("# Input grid is not orthogonal. EXIT.")
        mrc.close()
        sys.exit()

    ''' reorder axes '''
    mapcrs = np.subtract([mrc.header.mapc, mrc.header.mapr, mrc.header.maps], 1)
    sort = np.asarray([0, 1, 2], dtype=np.int32)
    for i in range(3):
        sort[mapcrs[i]] = i
    nxyzstart = np.asarray([ncrsstart[i] for i in sort])
    nxyz = np.asarray([ncrs[i] for i in sort])

    map = np.transpose(map, axes=2-sort[::-1])
    mrc.close()

    ''' shift origin according to n*start '''
    if not ignorestart:
        origin += np.multiply(nxyzstart, voxel_size)

    ''' shift by decimal '''
    if origin_shift is not None: 
        origin_shift = origin_shift - origin + np.floor(origin)

    ''' interpolate grid interval '''
    interp3d = Interp3d()
    if apix is not None:
        try:
            assert(voxel_size[0] == voxel_size[1] == voxel_size[2] == apix and origin_shift is None)
        except AssertionError:
            interp3d.del_mapout()
            target_voxel_size = np.asarray([apix, apix, apix], dtype=np.float32)
            print("# Rescale voxel size from {} to {}, shift origin by {}".format(voxel_size, target_voxel_size, origin_shift))
            if origin_shift is not None:
                interp3d.cubic(map, voxel_size[2], voxel_size[1], voxel_size[0], apix, origin_shift[2], origin_shift[1], origin_shift[0], nxyz[2], nxyz[1], nxyz[0])
                origin += origin_shift
            else:
                interp3d.cubic(map, voxel_size[2], voxel_size[1], voxel_size[0], apix, 0.0, 0.0, 0.0, nxyz[2], nxyz[1], nxyz[0])
                
            map = interp3d.mapout
            nxyz = np.asarray([interp3d.pextx, interp3d.pexty, interp3d.pextz], dtype=np.int32)
            voxel_size = target_voxel_size

    assert(np.all(nxyz == np.asarray([map.shape[2], map.shape[1], map.shape[0]], dtype=np.int32)))

    return map, origin, nxyz, voxel_size

def write_map(file_name, map, voxel_size, origin=(0.0, 0.0, 0.0), nxyzstart=(0, 0, 0)):
    mrc = mrcfile.new(file_name, overwrite=True)
    mrc.set_data(map)
    (mrc.header.nxstart, mrc.header.nystart, mrc.header.nzstart) = nxyzstart
    (mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z) = origin
    mrc.voxel_size = [voxel_size[i] for i in range(3)]

    mrc.close()




###############################################################################
######### The following codes are provided without use of interp3d ############
###############################################################################
def read_mrc(filename, ignorestart=False):
    return read_map(filename, ignorestart)

def read_map(filename, ignorestart=False):
    mrc = mrcfile.open(filename, mode='r')
    data = np.asarray(mrc.data.copy(), dtype=np.float32)
    voxel_size = np.asarray([mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z], dtype=np.float32)
    ncrsstart = np.asarray([mrc.header.nxstart, mrc.header.nystart, mrc.header.nzstart], dtype=np.float32)
    origin = np.asarray([mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z], dtype=np.float32)
    ncrs = (mrc.header.nx, mrc.header.ny, mrc.header.nz)
    angle = np.asarray([mrc.header.cellb.alpha, mrc.header.cellb.beta, mrc.header.cellb.gamma], dtype=np.float32)
    mapcrs = np.asarray([mrc.header.mapc, mrc.header.mapr, mrc.header.maps], dtype=np.int32)
    mrc.close()

    assert(np.all(angle == 90.0))

    ''' reorder axes

        mapcrs-1    sort        transpose
        0, 1, 2 --> 0, 1, 2 --> 0, 1, 2
        0, 2, 1 --> 0, 2, 1 --> 1, 0, 2
        1, 0, 2 --> 1, 0, 2 --> 0, 2, 1
        1, 2, 0 --> 2, 0, 1 --> 1, 2, 0
        2, 0, 1 --> 1, 2, 0 --> 2, 0, 1
        2, 1, 0 --> 2, 1, 0 --> 2, 1, 0

    '''
    sort = np.asarray([0, 1, 2], dtype=np.int32)
    for i in range(3):
        sort[mapcrs[i] - 1] = i
    nxyzstart = np.asarray([ncrsstart[i] for i in sort], dtype=np.int32)
    nxyz = np.asarray([ncrs[i] for i in sort], dtype=np.int32)
    data = np.transpose(data, axes=2-sort[::-1])

    ''' shift map origins '''
    if not ignorestart:
        origin += np.multiply(nxyzstart, voxel_size)

    #     (L, M, N) (3, ) (3, )
    return data, origin, voxel_size


def zone_box(coords, data, origin=None, voxel_size=None, r_zone=10):
    if origin is None:
        origin = np.array([0.0] * 3, dtype=np.float32)
    if voxel_size is None:
        voxel_size = np.array([1.0] * 3, dtype=np.float32)
    if coords.ndim == 1:
        coords = coords[None, ...]

    nxyz = data.shape[::-1]

    # Shift origin
    coords = coords - origin
    nxyz_min = np.maximum(np.floor((np.min(coords, axis=0) - r_zone) / voxel_size), 0).astype(np.int32)
    nxyz_max = np.minimum(np.ceil((np.max(coords, axis=0) + r_zone) / voxel_size), nxyz).astype(np.int32)

    assert(np.all(nxyz_min < nxyz_max))
    origin += np.multiply(nxyz_min, voxel_size)
    nxyz = np.subtract(nxyz_max, nxyz_min)
    zone_data = data[nxyz_min[2] : nxyz_max[2], nxyz_min[1] : nxyz_max[1], nxyz_min[0] : nxyz_max[0]]
    assert(np.all(np.shape(zone_data) == nxyz[::-1]))
    assert np.all(np.abs(np.round(origin/voxel_size) - origin/voxel_size) < 1e-4)
    return zone_data, origin


def enlarge_grid(grid, pad=0.0, pad_to_even=True):
    n0, n1, n2 = grid.shape
    n = np.max([n0, n1, n2])
    if pad_to_even:
        if n % 2 == 1:
            n += 1
    #if n < 100:
    #    n = 100
    gridx = np.full((n, n, n), pad, dtype=np.float32)
    gridx[:n0, :n1, :n2] = grid
    return gridx



def iterative_percentile(data, lower_bound=0.0, delta=10.0):
    p_start = 5.0
    val = 0.0
    while p_start < 99.0:
        val = np.percentile(data, q=p_start)
        print("# Percentile = {:.4f} val = {:.6f}".format(p_start, val))
        if val > lower_bound:
            break
        else:
            pass
        p_start += delta
    return val

###########################################
### Written by Jiahua He to save memory ###
###########################################
# generator version
def chunk_generator(padded_map, maximum, box_size, stride):
    assert stride <= box_size
    padded_map_shape = np.shape(padded_map)
    start_point = box_size - stride
    cur_x, cur_y, cur_z = start_point, start_point, start_point
    while (cur_z + stride < padded_map_shape[2] - box_size):
        next_chunk = padded_map[cur_x:cur_x + box_size, cur_y:cur_y + box_size, cur_z:cur_z + box_size]
        cur_x0, cur_y0, cur_z0 = cur_x, cur_y, cur_z
        cur_x += stride
        if (cur_x + stride >= padded_map_shape[0] - box_size):
            cur_y += stride
            cur_x = start_point # Reset X
            if (cur_y + stride  >= padded_map_shape[1] - box_size):
                cur_z += stride
                cur_y = start_point # Reset Y
                cur_x = start_point # Reset X

        if next_chunk.max() <= 0.0:
            continue
        else:
            yield cur_x0, cur_y0, cur_z0, next_chunk.clip(min=0.0, max=maximum) / maximum * 100.0

# get a batch of chunks from generator
def get_batch_from_generator(generator, batch_size, dtype=np.float32):
    positions = list()
    batch = list()
    for _ in range(batch_size):
        try:
            output = next(generator)
            positions.append(output[:3])
            batch.append(output[3])
        except StopIteration:
            break
    return positions, np.asarray(batch, dtype=dtype)

# map the batch of chunks to the map
def map_batch_to_map(pred_map, denominator, positions, batch, box_size):
    for position, chunk in zip(positions, batch):
        pred_map[
            :, # add a channel dim
            position[0]:position[0] + box_size, 
            position[1]:position[1] + box_size, 
            position[2]:position[2] + box_size
        ] += chunk
        denominator[
            :, # add a channel dim
            position[0]:position[0] + box_size, 
            position[1]:position[1] + box_size, 
            position[2]:position[2] + box_size
        ] += 1
    return pred_map, denominator

MRCObject = namedtuple("MRCObject", ["grid", "voxel_size", "origin"])

if __name__ == "__main__":
    pass
