"""
Refine of built models
"""
import os
import shutil
import argparse

def main(args):
    from emprot.pipeline import post_refine
    post_refine.main(args)

def add_args(parser):
    parser.add_argument("--pdb", "-p", help="Input pdbs")
    parser.add_argument("--map", "-m", help="Input map")
    parser.add_argument("--output", "-o", help="Output directory")
    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = add_args(parser).parse_args()
    main(args)


