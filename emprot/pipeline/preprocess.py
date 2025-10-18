# process templates going the following steps
# 0. remove invalid coordinates record and keep only "ATOM" record
# 1. align template sequence with given sequence and cut head and tail loops
# 2. renumber template residue indices such that they start from 0 and no gap inside
# 3. if template only have CA coords, make it full-atom
# 4. make sure at least 80 characters at each line (for unidoc)
# 5. do sequence alignment between given seq and template

import os
import sys
import shutil
import numpy as np

from emprot.io.fileio import getlines, writelines
from emprot.io.pdbio import read_pdb, chains_atom_pos_to_pdb
from emprot.io.seqio import read_fasta

from emprot.utils.misc_utils import pjoin, abspath
from emprot.utils.residue_constants import index_to_restype_1

def main(args):
    fseq = args.seq
    out_dir = args.output
    out_dir = abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    chain_templs_out_dir = pjoin(out_dir, "chain_templs")
    complex_templs_out_dir = pjoin(out_dir, "complex_templs")
    templs_out_dir = pjoin(out_dir, "templs")

    os.makedirs(chain_templs_out_dir, exist_ok=True)
    os.makedirs(complex_templs_out_dir, exist_ok=True)
    os.makedirs(templs_out_dir, exist_ok=True)

    # check input seq
    if fseq is not None:
        seqs = read_fasta(fseq)
        fseqout = pjoin(out_dir, "format_seq.fasta")
        seq_lines = []
        for i, seq in enumerate(seqs):
            # remove non-std residues
            seq0 = "".join([x for x in seq if x in index_to_restype_1[:20]])
            seq_lines.append(">chain_{}".format(i))
            seq_lines.append(seq0)
        writelines(fseqout, seq_lines)
        print("Write formatted seq to {}".format(fseqout))
    else:
        print("WARNING you do not input sequence(s)")
        fseqout = pjoin(out_dir, "format_seq.fasta")
        seq_lines = [">dummy_chain_0", "AAA"]
        writelines(fseqout, seq_lines)


    # read single chain and complex structure template
    ftempls_chain = args.chain
    if ftempls_chain is None:
        ftempls_chain = []


    # first, try using complex template
    ftempls_complex = args.complex
    has_complex_template = False
    if ftempls_complex is not None:
        print("User input {} complex structure template".format(1))
        try:
            atom_pos, atom_mask, res_type, res_idx, chain_idx, bfactor = read_pdb(
                ftempls_complex,
                keep_valid=False, 
                return_bfactor=True,
            )

            # Split to chains
            for k in range(0, chain_idx.max() + 1):
                sel_mask = chain_idx == k

                fout = pjoin(complex_templs_out_dir, f"templ_{k}.pdb")
                chains_atom_pos_to_pdb(
                    filename=fout,
                    chains_atom_pos=[atom_pos[sel_mask]],
                    chains_atom_mask=[atom_mask[sel_mask]],
                    chains_res_types=[res_type[sel_mask]],
                    chains_res_idxs=[np.arange(0, len(sel_mask), dtype=np.int32)],
                    chains_bfactors=[bfactor[sel_mask]],
                    suffix='pdb',
                )
                print(f"Rewrite chain {k} from complex to {fout}")
           
            if len(atom_pos) > 0:
                has_complex_template = True

        # handle exception
        except Exception as e:
            print(f"Error occurs -> {e}")
            print(f"WARNING cannot rewrite complex template")
            if os.path.exists(ftempls_complex):
                print("WARNING the template file exists, maybe no ATOM is recorded")
            else:
                print("WARNING the template file not exists")
            



    # Write specified chain templates
    has_chain_template = False
    if ftempls_chain is not None:
        # Use single-chain structures
        if len(ftempls_chain) == 0:
            print("User input no single-chain structure templates")
            print("Using no structure templates")
        else:
            print("User input {} single-chain structure templates".format(len(ftempls_chain)))
            print("Using single-chain structure as templates")
            for k, ftempl in enumerate(ftempls_chain):
                try:
                    atom_pos, atom_mask, res_type, res_idx, chain_idx, bfactor = read_pdb(ftempl, keep_valid=False, return_bfactor=True)
                    
                    # check invalid coordinates
                    pass

                    fout = pjoin(chain_templs_out_dir, f"templ_{k}.pdb")
                    chains_atom_pos_to_pdb(
                        filename=fout,
                        chains_atom_pos=[atom_pos],
                        chains_atom_mask=[atom_mask],
                        chains_res_types=[res_type],
                        chains_res_idxs=[np.arange(0, len(atom_pos), dtype=np.int32)],
                        chains_bfactors=[bfactor],
                        suffix='pdb',
                    )
                    print(f"Rewrite template {k} from single-chain to {fout}")

                    has_chain_template = True

                # handle exception
                except Exception as e:
                    print(f"Error occurs -> {e}")
                    print(f"WARNING cannot rewrite template {k}")
                    if os.path.exists(ftempl):
                        print("WARNING the template file exists, maybe no ATOM is recorded")
                    else:
                        print("WARNING the template file not exists")
    

    # If have complex structure, we ignore the single-chain structures
    print("# Has chain   template = {}".format(has_chain_template))
    print("# has complex template = {}".format(has_complex_template))

    if has_complex_template:
        print("# Has complex template, use chain from complex as template")
        shutil.copytree(
            complex_templs_out_dir, 
            templs_out_dir, 
            dirs_exist_ok=True,
        )
        
        # write flag
        with open(pjoin(templs_out_dir, "type.txt"), 'w') as f:
            f.write("complex\n")
            
    elif has_chain_template:
        print("# No complex template, use chain from single as template")
        shutil.copytree(
            chain_templs_out_dir,
            templs_out_dir,
            dirs_exist_ok=True, 
        )

        # write flag
        with open(pjoin(templs_out_dir, "type.txt"), 'w') as f:
            f.write("chain\n")
        
    else:
        print("# No template found")

        # write flag
        with open(pjoin(templs_out_dir, "type.txt"), 'w') as f:
            f.write("none\n")
        

    # interp map
    if args.map is not None:
        from emprot.utils.cryo_utils import parse_map, write_map
        data, origin, _, vsize = parse_map(args.map, False, 1.0)
        map_out = pjoin(args.output, "format_map.mrc")
        write_map(
            map_out, 
            data, 
            origin=origin, 
            voxel_size=[1., 1., 1.],
        )
        print("# Write formated map to {}".format(map_out))



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", "-s", help="Input sequence")
    parser.add_argument("--chain", nargs='*', help="Input single chain structure predicted by AF2", required=False)
    parser.add_argument("--complex", help="Input complex structure predicted by AF3", required=False)
    parser.add_argument("--map", "-m", help="Input map")
    parser.add_argument("--output", "-o", help="Output directory of processed models", default="./")
    args = parser.parse_args()
    main(args)
