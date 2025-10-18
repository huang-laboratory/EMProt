import os
import sys
import tempfile
import subprocess
from Bio import pairwise2
from Bio.Seq import Seq
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from emprot.io.fileio import writelines, extract_lines_by_ca

from emprot.utils.residue_constants import (
    restype_3to1, index_to_restype_1,
    restype_1to3, index_to_restype_3,
)
from emprot.utils.misc_utils import pjoin, get_temp_dir

def readlines(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines if line.strip()]
    return lines

def read_lines(filename):
    lines = readlines(filename)
    return lines

def read_fasta(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    seqs = []
    seq = ""
    for line in lines:
        if line.startswith('>'):
            if len(seq) > 0:
                seqs.append(seq)
            seq = ""
            continue
        seq += line.strip()
    if len(seq) > 0:
        seqs.append(seq)
    return seqs

def std_aa_seq(seq, outlier_filling='X'):
    seq0 = []
    for i in range(len(seq)):
        ch = seq[i].upper()
        if ch in index_to_restype_1[:20]:
            seq0.append(ch)
        else:
            seq0.append(outlier_filling)
    return "".join(seq0)

def write_lines_to_file(lines, filename):
    with open(filename, 'w') as f:
        for line in lines:
            f.write(line.strip())
            f.write("\n")

def write_seq_to_file(seq, filename):
    lines = [">seq"]
    lines.append(seq.strip())
    write_lines_to_file(lines, filename)

def read_secstr(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    seqs = []
    seq = ""
    for line in lines:
        if line.startswith('>'):
            if len(seq) > 0:
                seqs.append(seq)
            seq = ""
            continue
        seq += line.strip()
    if len(seq) > 0:
        seqs.append(seq)
    return seqs
    '''
    with open(filename, 'r') as f:
        lines = f.readlines()
    secs = []
    for line in lines:
        if len(line.strip()) > 0:
            secs.append(line.strip())
    return secs
    '''

def toupper(seq):
    return seq.upper()

def tolower(seq):
    return seq.lower()

def is_na(seq):
    nucs = ['A', 'G', 'C', 'U', 'T', 'I', 'N', 'X', 'a', 'g', 'c', 'u', 't', 'i', 'n', 'x']
    for s in seq:
        if s not in nucs:
            return False
    return True

def is_dna(seq):
    return is_na(seq) and ('T' in seq or 't' in seq)

def is_rna(seq):
    return is_na(seq) and 'T' not in seq and 't' not in seq




def seq_identity(a, b):
    assert len(a) == len(b)
    s = 0
    for i in range(len(a)):
        if a[i] == '-' or b[i] == '-':
            continue
        if a[i] == b[i]:
            s += 1
    return s / len(a)


def get_sequence_from_pdb_lines(lines):
    seq = []
    for line in lines:
        if line.startswith("ATOM") and line[12:16] == " CA ":
            try:
                resname3 = line[17:20]
                resname1 = restype_3to1[resname3]
            except:
                resname3 = "XXX"
                resname1 = "X"
            # keep only protein sequence
            if resname1 not in index_to_restype_1[:20]:
                resname1 = "X"
            seq.append(resname1)
    return "".join(seq)

def update_sequence_to_pdb_lines(lines, seq):
    # since the sequence is updated
    # for a convience, only Ca coords are kept
    # a pdb line looks like
    # ATOM      1  CA  TYR A   1     126.118 180.522  68.870  1.00100.00         1 C
    # 0         0         0         0
    ca_lines = extract_lines_by_ca(lines)
    assert len(ca_lines) == len(seq)
    new_lines = []
    for i, line in enumerate(ca_lines):
        resname1 = seq[i]
        if resname1 not in index_to_restype_1[:20]:
            resname1 = "X"
        resname3 = restype_1to3[resname1]
        new_line = line[:17] + "{:>3s}".format(resname3) + line[20:]
        new_lines.append(new_line)
    return new_lines


# Format alignment
def format_alignment(A, align, B):
    # remove the '-' in B
    new_A = []
    new_align = []
    new_B = []

    for i, ch in enumerate(B):
        if ch != '-':
            new_A.append(A[i])
            new_align.append(align[i])
            new_B.append(ch)

    new_A = "".join(new_A)
    new_B = "".join(new_B)
    new_align = "".join(new_align)

    return new_A, align, new_B


def nwalign_fast(a, b, lib_dir="./", temp_dir=None, verbose=False, namea=None, nameb=None, fmt=False):
    """
    with tempfile.TemporaryDirectory() as __temp_dir:
        if temp_dir is None:
            temp_dir = __temp_dir
    """

    with get_temp_dir(temp_dir) as temp_dir:

        if namea is None:
            namea = "seq_a"
        if nameb is None:
            nameb = "seq_b"

        f1 = pjoin(temp_dir, f"{namea}.fasta")
        f2 = pjoin(temp_dir, f"{nameb}.fasta")

        writelines(f1, [f">{namea}", a])
        writelines(f2, [f">{nameb}", b])

        cmd = lib_dir + "/bin/NWalign {} {}".format(f1, f2)
        if verbose:
            print(f"Running command {cmd}")
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if result.returncode == 0:
            result = result.stdout.decode('utf-8').split('\n')[:3]
            seqA, align, seqB = result

            if fmt:
                seqA, align, seqB = format_alignment(seqA, align, seqB)

            seqid = 0
            seqcov = 0
            for i in range(len(seqA)):
                if seqA[i] != '-' and seqA[i] == seqB[i]:
                    seqid += 1
                if seqA[i] != '-' and seqB[i] != '-':
                    seqcov += 1
            denoma = len(a)
            denomb = len(b)
            if len(a) == 0:
                denoma += 1e-8
            if len(b) == 0:
                denomb += 1e-8

            seqAid = seqid / denoma
            seqAcov = seqcov / denoma
            seqBid = seqid / denomb
            seqBcov = seqcov / denomb
            return (seqA, align, seqB, seqAid, seqAcov, seqBid, seqBcov)
        else:
            return (None, None, None, 0.0, 0.0, 0.0, 0.0)

def multi_align(a, b, mode='global', match=2, mismatch=-1, gap_open=-0.5, gap_extend=-0.1, n_classes=4, top=20):
    seq1 = Seq(a)
    seq2 = Seq(b)
    # Get match dict
    match_dict = {
        ('A', 'A'): 2, ('A', 'G'): 1, ('A', 'C'):-1, ('A', 'U'):-1, ('A', 'T'):-1, 
        ('G', 'A'): 1, ('G', 'G'): 2, ('G', 'C'):-1, ('G', 'U'):-1, ('G', 'T'):-1, 
        ('C', 'A'):-1, ('C', 'G'):-1, ('C', 'C'): 2, ('C', 'U'): 1, ('C', 'T'): 1, 
        ('U', 'A'):-1, ('U', 'G'):-1, ('U', 'C'): 1, ('U', 'U'): 2, ('U', 'T'): 1, 
        ('T', 'A'):-1, ('T', 'G'):-1, ('T', 'C'): 1, ('T', 'U'): 1, ('T', 'T'): 2, 
    }
    if mode == 'global':
        alignments = pairwise2.align.globalds(seq1, seq2, match_dict, gap_open, gap_extend, one_alignment_only=False)
    elif mode == 'local':
        alignments =  pairwise2.align.localds(seq1, seq2, match_dict, gap_open, gap_extend, one_alignment_only=False)
    else:
        raise ValueError
    print(len(alignments))
    return alignments[:top] if top > 1 else [alignments[top]]



def read_era_result(filename):
    result = readlines(filename)
    if len(result) < 5:
        raise "ERA result is not valid"
    return result[-5:]


def read_usalign_result(filename):
    result = readlines(filename)
    if len(result) < 3:
        raise "USalign result is not valid"
    return result[-4:-1]



'''
    input
    seqA : ---AAAGASG-ASGASG--ASFASF--
    seqB : ASDASO--S-ASF--ASFAFSF-ASAD
    return
    seqA : AAAGASG-ASGASG--ASFASF
    seqB : ASO--S-ASF--ASFAFSF-AS
'''
def remove_bar(seq1, seq2, sec2=None):
    assert len(seq1) == len(seq2)
    if sec2 is not None:
        assert len(seq2) == len(sec2)

    # Remove bar both terminus in seq1
    n = len(seq1)
    ret_seq1 = []
    ret_seq2 = []
    ret_sec2 = []
    for i in range(n):
        if seq1[i] != '-':
            ret_seq1.append(seq1[i])
            ret_seq2.append(seq2[i])
            if sec2 is not None:
                ret_sec2.append(sec2[i])

    # s1 does not have '-'
    # s2 may have '-'
    n = len(ret_seq2)
    for i in range(n):
        if ret_seq2[i] == '-':
            ret_seq2[i] = ret_seq1[i]
            if sec2 is not None:
                ret_sec2[i] = '.'

    ret_seq1 = "".join(ret_seq1)
    ret_seq2 = "".join(ret_seq2)
    ret_sec2 = "".join(ret_sec2)
    if sec2 is not None:
        return ret_seq1, ret_seq2, ret_sec2
    else:
        return ret_seq1, ret_seq2


# Simple removes '-' in seq2
def remove_bar_simple(seq1, seq2, sec2=None):
    assert len(seq1) == len(seq2)
    if sec2 is not None:
        assert len(seq2) == len(sec2)

    ret_seq1 = "".join([x for x in seq1 if x != '-'])

    ret_seq2 = []
    ret_sec2 = []

    for i in range(len(seq2)):
        if seq2[i] == '-':
            continue
        ret_seq2.append(seq2[i])
        if sec2 is not None:
            ret_sec2.append(sec2[i])

    ret_seq2 = "".join(ret_seq2)
    ret_sec2 = "".join(ret_sec2)

    if sec2 is not None:
        return ret_seq1, ret_seq2, ret_sec2
    else:
        return ret_seq1, ret_seq2


# If seq1[i] != '-' and seq2[i] != '-'
# seq1[i] = seq2[i]
def remove_bar_match(seq1, seq2):
    assert len(seq1) == len(seq1)
    seqA = []
    seqB = []
    for i in range(len(seq1)):
        if seq1[i] != '-' and seq2[i] != '-':
            seqA.append(seq1[i])
            seqB.append(seq2[i])
        else:
            seqA.append(seq1[i])
            seqB.append(seq1[i])

    seqA = "".join(seqA)
    seqB = "".join(seqB)

    seqA = seqA.replace('-', '')
    seqB = seqB.replace('-', '')

    return seqA, seqB


# Parse a second structure
def ss_to_pairs(ss):
    ret = []
    s = []

    for i in range(len(ss)):
        if ss[i] == '.':
            continue

        # Find bracket
        if ss[i] in ['{', '[', '<', '(']:
            s.append( (ss[i], i) )
            continue
        else:
            symbol = '.';
            if ss[i] == ')':
                symbol = '('
            elif ss[i] == ']':
                symbol = '['
            elif ss[i] == '}':
                symbol = '{'
            elif ss[i] == '>':
                symbol = '<'

            # Find nearest left bracket
            s0 = []
            while s:
                ch = s[-1][0]
                ii = s[-1][1]
                if ch == symbol:
                    ret.append( (ii, i) )
                    s.pop()
                    break
                else:
                    s0.append( s[-1])
                    s.pop()

            # Recover s
            while s0:
                s.append( s0[-1] )
                s0.pop()

    # Sort by first number
    ret.sort(key=lambda x:x[0])
    return ret


def align_seq_and_sec(seq, sec):
    assert len(seq) >= len(sec)
    ret = []
    i, k = 0, 0
    while i < len(seq):
        if seq[i] != '-':
            ret.append(sec[k])
            i+=1
            k+=1
        else:
            ret.append('&')
            i+=1
    # Postprocess
    if len(ret) > len(seq):
        ret = ret[:len(seq)]
    elif len(ret) < len(seq):
        ret.extend(['&']*(len(seq)-len(ret)))
    return "".join(ret)


def convert_to_rna_seq(seq):
    return seq.replace('T', 'U')

def convert_to_dna_seq(seq):
    return seq.replace('U', 'T')


def get_multi_seq_align_among_candidates(seq1, seqs, clean_gap=True):
    # Determine which sequence best fits the segment
    score0 = -1e6
    seqA0 = None
    seqB0 = None
    k0 = None

    for k, seq2 in enumerate(seqs):
        # Target sequence is >= query sequence
        '''
        if len(seq2) < len(seq1):
            continue
        '''

        alignments = multi_align(seq1, seq2)
        for i, alignment in enumerate(alignments):
            seqA, seqB, score, start, end = alignment[:5]
            print(seqA)
            print(seqB)
            print(score)

    seqA = seqA0
    seqB = seqB0






def get_seq_align_among_candidates(seq1, seqs, **kwargs):
    # Determine which sequence best fits the segment
    score0 = -1e6
    seqA0 = None
    seqB0 = None
    k0 = None

    for k, seq2 in enumerate(seqs):
        # Target sequence is >= query sequence
        '''
        if len(seq2) < len(seq1):
            continue
        '''

        alignment = align(seq1, seq2, gap_open=-1.0, gap_extend=-0.5, **kwargs)[0]
        seqA, seqB, score, start, end = alignment[:5]

        if score > score0:
            score0 = score
            seqA0 = seqA[start:end+1]
            seqB0 = seqB[start:end+1]
            k0 = k

    seqA = seqA0
    seqB = seqB0
    #print(seqA)
    #print(seqB)

    # In case that given sequence is invalid
    if seqA is None or \
       seqB is None:
        seqA = seq1
        seqB = seq1

    seqA, seqB = remove_bar(seqA, seqB)
    #seqA, seqB = remove_bar_match(seqA, seqB)

    # In case that seqB is shorted
    seqB = seqB + 'U'*(len(seq1)-len(seqB))
    return seqA, seqB, score0, k0


def format_seqs(seqs):
    # 0 Convert to upper
    seqs = [seq.upper() for seq in seqs]

    # 1. Convert chars not in "AGCUT" to "AGCUT"
    seqs0 = []
    for seq in seqs:
        seq0 = ""
        for s in seq:
            if s in ['A', 'G', 'C', 'U', 'T']:
                seq0 += s
            else:
                seq0 += 'U'
        seqs0.append(seq0)
    return seqs0


# some utils for ranges
def is_valid_interval(length, interval):
    start, end = interval
    return 0 <= start < length and 0 <= end < length and start < end

def valid_interval(length, interval):
    start, end = interval
    if start < 0:
        start = 0
    if end > length:
        end = length
    return (start, end)

def remove_intervals(length, intervals):
    intervals = [valid_interval(length, x) for x in intervals]
    flag = [True] * length
    for interval in intervals:
        start, end = interval
        for k in range(start, end):
            flag[k] = False
    ret = []
    k = 0
    while k < length:
        start = k
        interval = []
        while start < length and not flag[start]:
            start += 1
        while start < length and flag[start]:
            interval.append(start)
            start += 1
        if len(interval) >= 1:
            ret.append([interval[0], interval[-1]])
        k = start
    return ret


# Find chars

def find_first_of(s, chars, start=0, end=None):
    return next((i for i, c in enumerate(s[start:end or len(s)], start) if c in chars), -1)

def find_first_not_of(s, chars, start=0, end=None):
    return next((i for i, c in enumerate(s[start:end or len(s)], start) if c not in chars), -1)

def find_last_of(s, chars, start=None, end=None):
    substr = s[start:end]
    return max((substr.rfind(c) + (start or 0) for c in chars), default=-1)

def find_last_not_of(s, chars, start=None, end=None):
    if end is None:
        end = len(s)
    if start is None:
        start = 0
    return next((i for i in range(end-1, start-1, -1) if s[i] not in chars), -1)


if __name__ == '__main__':
    pass
