"""
Semi-flexible ftting of stucture templates to map
"""
import os
import shutil
import argparse

def main(args):
    pass

def add_args(parser):
    parser.add_argument("--chain", "-c", help="Input pdbs")
    parser.add_argument("--map", "-m", help="Input map")
    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = add_args(parser).parse_args()
    main(args)

