def main(args):
    from seqio import read_fasta
    seqs = read_fasta(args.i)
    
    n = 0
    repeat = set()
    with open(args.o, 'w') as f:
        for seq in seqs:
            if seq not in repeat:
                f.write(">{}\n".format(n))
                f.write("{}\n".format(seq))
                n += 1
                repeat.add(seq)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, help="Input sequence(s)")
    parser.add_argument("-o", type=str, help="Output sequence(s)")
    args = parser.parse_args()
    main(args)
