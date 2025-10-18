from cryo_utils import parse_map, write_map
import argparse
import sys

def main(args):
    grid, origin, nxyz, voxel_size = parse_map(args.i, ignorestart=args.ignore_start, apix=args.a)
    write_map(
        args.o,
        grid,
        voxel_size=voxel_size,
        origin=origin,
    )
    if args.verbose:
        print("# Write formated map to {}".format(args.o))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=True, help="Input map dir")
    parser.add_argument("-o", required=True, help="Output map dir")
    parser.add_argument("-a", type=float, default=1.0, help="Target voxel size")
    parser.add_argument("--ignore_start", action='store_true', help="Ignore nxyz start")
    parser.add_argument("--verbose", action='store_true', help="Whether to log")
    args = parser.parse_args()
    main(args)
