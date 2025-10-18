"""
Evaluate EMProt built model
"""
import os
import time
import argparse
from emprot.utils.misc_utils import pjoin

def main(args):
    start = time.time()

    #output_dir = os.path.abspath(os.path.dirname(args.structure))
    output_dir = os.path.abspath(args.output)

    temp_dir = pjoin(output_dir, "temp")

    from emprot.pipeline import eval_localx

    eval_args = argparse.Namespace()
    eval_args.output = args.output
    eval_args.pdb = args.structure
    eval_args.map = pjoin(temp_dir, "format_map.mrc")
    eval_args.ca = pjoin(temp_dir, "ncac", "raw_ca.pdb")
    eval_args.resolution = 5.0
    eval_args.smooth = True

    eval_localx.main(eval_args)

    end = time.time()
    print("# Time consumption {:.4f}. Done eval".format(end - start))

def add_args(parser):
    parser.add_argument("--structure", "-s", help="Target structure", required=True)
    parser.add_argument("--output", "-o", help="Output dir", default="./")
    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = add_args(parser).parse_args()
    main(args)
