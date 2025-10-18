import shutil
import numpy as np

def getlines(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    return lines

def extract_lines_by_ca(lines):
    return [line for line in lines if line.startswith("ATOM") and line[12:16] == " CA "]

def writelines(filename, lines):
    with open(filename, 'w') as f:
        for line in lines:
            f.write(line.strip("\n") + "\n")

def extract_lines_by_res_idx(lines, res_idxs):
    ret = []
    for line in lines:
        if line.startswith("ATOM"):
            res_idx = int(line[22:26])
            if res_idx in res_idxs:
                ret.append(line)
    return ret


def copy_file(src, dst):
    try:
        shutil.copy(src, dst)
        return True
    except Exception as e:
        return False
