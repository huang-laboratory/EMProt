import os
import sys
import numpy as np

def main(args):
    pass

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", nargs='+', help="Input structure")
    args = parser.parse_args()
    main(args)
