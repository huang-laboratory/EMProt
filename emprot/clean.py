"""
Clean the useless intermediate files
"""
import os
import shutil
import argparse
from emprot.utils.misc_utils import pjoin

def main(args):
    output_dir = args.output

    temp_dir = pjoin(output_dir, "temp")
    if os.path.exists(temp_dir):
        files_to_be_removed = [
            #pjoin(temp_dir, "format_map.mrc"),
            #pjoin(temp_dir, "ncac", "n.mrc"),
            #pjoin(temp_dir, "ncac", "ca.mrc"),
            #pjoin(temp_dir, "ncac", "c.mrc"),
            pjoin(temp_dir, "ncac", "aa_logits.npz"),
        ]

        for f in files_to_be_removed:
            if os.path.exists(f):
                os.remove(f)
                print("# Removing {}".format(f))
            else:
                print("# Not found {} skip this file".format(f))
    else:
        print("# No temp file is found in {} will not do anything".format(output_dir))

def add_args(parser):
    parser.add_argument("--output", "-o", help="Output directory that contains dir 'temp'", required=True)
    parser.add_argument("--remove_all", action='store_true', 
        help="Remove all files in 'temp' thoroughly. By default only predicted maps are removed", 
    )
    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = add_args(parser).parse_args()
    main(args)
