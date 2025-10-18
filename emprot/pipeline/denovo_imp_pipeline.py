import os
import re
import sys
import time
import tqdm
import tempfile
import argparse
import numpy as np

from scipy.spatial import KDTree

from emprot.io.fileio import (
    getlines,
    writelines,
)

from emprot.io.seqio import (
    read_fasta,
    get_sequence_from_pdb_lines,
    nwalign_fast,
)

from emprot.io.pdbio import (
    read_pdb,
    chains_atom_pos_to_pdb,
)

from emprot.utils.misc_utils import (
    abspath,
    pjoin,
)

from emprot.utils.tm_utils import (
    run_TMalign,
)

from emprot.utils.geo import (
    rmsd,
    kabsch,
    apply,
)

from emprot.utils.residue_constants import (
    index_to_restype_3,
    index_to_restype_1,
    restype_3_to_index,
    restype_1_to_index,
)

from emprot.utils.shift_field import create_shift_field

def split_gaps_seq(sequence):
    gaps = []
    for m in re.finditer(r' +', sequence):
        gaps.append((m.start(), m.group()))
    gaps.sort(
        key=lambda x:x[0],
    )
    return gaps
    

def split_frags_and_gaps_seq(sequence):
    frags_and_gaps = []
    for m in re.finditer(r'-+', sequence):
        frags_and_gaps.append((m.start(), m.group()))
    for m in re.finditer(r'[^-]+', sequence):
        frags_and_gaps.append((m.start(), m.group()))
    frags_and_gaps.sort(
        key=lambda x:x[0],
    )
    return frags_and_gaps

def idx_aligned_to_original(seq):
    n = 0
    idxs = np.full(len(seq), -1, dtype=np.int32)
    for i, s in enumerate(seq):
        if s != '-':
            idxs[i] = n
            n += 1
    return idxs

def main(args):
    ts = time.time()
    try:
        lib_dir = args.lib
        lib_dir = abspath(lib_dir)
        out_dir = args.output
        out_dir = abspath(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        fseq = args.seq
        seqs = read_fasta(fseq)
        print("Found {} seqs".format(len(seqs)))

        verbose = args.verbose
        # the script is to fix structure gaps inside each chains

        # preprocess templates
        # split template models into domains
        ftempls = args.template

        # process chains
        fchains = args.chain

        # set up temp dir
        temp_dir = pjoin(out_dir, "alignments")
        os.makedirs(temp_dir, exist_ok=True)
        print(f"Input {len(fchains)} chains")
        print(f"Input {len(ftempls)} templates")


        # for each chain, find the best match templ
        chain_seqs = []
        templ_seqs = []

        chain_templ_relation = dict()
        chain_templ_seq_align = dict()

        seq_temp_dir = pjoin(temp_dir, "seqs")
        os.makedirs(seq_temp_dir, exist_ok=True)
        for i in range(len(fchains)): 
            # find the best matched template according to seqid
            # convert X to G to avoid Error from protein sequence align
            chain_lines = getlines(fchains[i])

            # get sequene from pdbfile
            chain_seq = get_sequence_from_pdb_lines(chain_lines)
            chain_seq = "".join([x if x != "X" else "G" for x in chain_seq])

            # record chain seq
            chain_seqs.append(chain_seq)

            seqid_best = -1e6
            idx_best = -1
            for k in range(len(ftempls)):
                templ_lines = getlines(ftempls[k])
                templ_seq = get_sequence_from_pdb_lines(templ_lines)
                templ_seq = "".join([x if x != "X" else "G" for x in templ_seq])

                # record templ seq
                if i == 0:
                    templ_seqs.append(templ_seq)

                result = nwalign_fast(chain_seq, templ_seq, lib_dir=lib_dir, temp_dir=seq_temp_dir, verbose=verbose, namea=f"chain_{i}", nameb=f"templ_{k}")
                seqAid = result[3]
                seqBid = result[5]

                if seqAid > seqid_best:
                    seqid_best = seqAid
                    idx_best = k

                # record alignment
                key = "{}_{}".format(i, k)
                chain_templ_seq_align[key] = result

            chain_templ_relation[i] = idx_best
        for k, v in chain_templ_relation.items():
            print("Best template fit for chain {} is {}".format(fchains[k], ftempls[v]))


        templs_atom_pos = []
        templs_atom_mask = []
        templs_res_type = []
        templs_res_idx = []
        templs_chain_idx = []
        for ftempl in ftempls:
            atom_pos, atom_mask, res_type, res_idx, chain_idx = read_pdb(ftempl, keep_valid=False)
            templs_atom_pos.append(atom_pos)
            templs_atom_mask.append(atom_mask)
            templs_res_type.append(res_type)
            templs_res_idx.append(res_idx)
            templs_chain_idx.append(chain_idx)

        shift_field_refine = False
        seqid_cutoff = 0.60
        seqcov_cutoff = 0.60
        rms_cutoff = 2.0
        gap_ratio_chain_cutoff = 0.50
        gap_ratio_templ_cutoff = 0.20
        fix_gap_at_terminus = True
        all_fixed_atom_pos = []
        all_fixed_atom_mask = []
        all_fixed_res_type = []
        all_is_fixed = []
        for i, fchain in enumerate(fchains):
            # find gaps
            chain_atom_pos, chain_atom_mask, chain_res_type, chain_res_idx, _ = read_pdb(fchain, keep_valid=False)
        
            tidx = chain_templ_relation[i]
            templ_atom_pos = templs_atom_pos[tidx]
            templ_atom_mask = templs_atom_mask[tidx]
            templ_res_type = templs_res_type[tidx]

            chain_seq = "".join( [index_to_restype_1[x] for x in chain_res_type] )
            templ_seq = "".join( [index_to_restype_1[x] for x in templ_res_type] )

            key = "{}_{}".format(i, tidx)
            result = chain_templ_seq_align[key]

            # only fix frag with high identity
            if not (result[3] > seqid_cutoff and result[4] > seqcov_cutoff):
                all_fixed_atom_pos.append(chain_atom_pos)
                all_fixed_atom_mask.append(chain_atom_mask)
                all_fixed_res_type.append(chain_res_type)
                all_is_fixed.append(False)
                print('-'*80)
                print("Skip chain {} with seqid = {:.4f} seqcov = {:.4f}".format(i, result[3], result[4]))
                continue
            else:
                all_is_fixed.append(True)
                print('-'*80)
                print("Fix  chain {} with seqid = {:.4f} seqcov = {:.4f}".format(i, result[3], result[4]))

            # find frags and gaps
            frags_and_gaps = split_frags_and_gaps_seq(result[0])

            # original idxs
            idx_a2o_chain = idx_aligned_to_original(result[0])
            idx_a2o_templ = idx_aligned_to_original(result[2])

            # for each gap, find the neighbor frags
            fixed_atom_pos = []
            fixed_atom_mask = []
            fixed_res_type = []
            gap_start = 0
            while gap_start < len(frags_and_gaps):
                if '-' in frags_and_gaps[gap_start][1]:
                    break
                gap_start += 1

            gap_end = len(frags_and_gaps) - 1
            while gap_end >= 0:
                if '-' in frags_and_gaps[gap_end][1]:
                    break
                gap_end -= 1


            for idx, s in enumerate(frags_and_gaps):
                has_correspondance = True
                if '-' in s[1]:
                    gap_at_terminus = (idx == gap_start or idx == gap_end)
                    gap_ratio_chain = len(s[1]) / len(chain_atom_pos) # maybe  > 1.0
                    gap_ratio_templ = len(s[1]) / len(templ_atom_pos) # always < 1.0
                    print("Gap ratio chain = {:.4f}".format(gap_ratio_chain))
                    print("Gap ratio templ = {:.4f}".format(gap_ratio_templ))
                    # if gap is at both terminus do not fix too large gap
                    if gap_at_terminus:
                        if not fix_gap_at_terminus:
                            print("Not fix terminal gaps")
                            continue

                        if gap_ratio_chain > gap_ratio_chain_cutoff or \
                            gap_ratio_templ > gap_ratio_templ_cutoff:
                            continue
                        print("Fix terminal gap")

                    # if gap
                    chain_frag_atom_pos = []
                    templ_frag_atom_pos = []
                    gap_atom_pos = []
                    gap_atom_mask = []
                    gap_res_type = []
                    if idx - 1 >= 0:
                        start = frags_and_gaps[idx - 1][0]
                        end = start + len(frags_and_gaps[idx - 1][1])

                        sel_idxs_chain = np.asarray([idx_a2o_chain[x] for x in range(start, end)], dtype=np.int32)
                        sel_idxs_templ = np.asarray([idx_a2o_templ[x] for x in range(start, end)], dtype=np.int32)
                        mask = np.logical_and(sel_idxs_chain != -1, sel_idxs_templ != -1)

                        chain_frag_atom_pos.append(
                            chain_atom_pos[sel_idxs_chain[mask]]
                        )

                        templ_frag_atom_pos.append(
                            templ_atom_pos[sel_idxs_templ[mask]]
                        )

                    if idx + 1 < len(frags_and_gaps):
                        start = frags_and_gaps[idx + 1][0]
                        end = start + len(frags_and_gaps[idx + 1][1])

                        sel_idxs_chain = np.asarray([idx_a2o_chain[x] for x in range(start, end)], dtype=np.int32)
                        sel_idxs_templ = np.asarray([idx_a2o_templ[x] for x in range(start, end)], dtype=np.int32)
                        mask = np.logical_and(sel_idxs_chain != -1, sel_idxs_templ != -1)

                        chain_frag_atom_pos.append(
                            chain_atom_pos[sel_idxs_chain[mask]]
                        )
                        
                        templ_frag_atom_pos.append(
                            templ_atom_pos[sel_idxs_templ[mask]]
                        )

                    gap_atom_pos.append(
                        templ_atom_pos[
                            s[0] : s[0] + len(s[1])
                        ]
                    )

                    gap_atom_mask.append(
                        templ_atom_mask[
                            s[0] : s[0] + len(s[1])
                        ]
                    )

                    gap_res_type.append(
                        templ_res_type[
                            s[0] : s[0] + len(s[1])
                        ]
                    )

                    chain_frag_atom_pos = np.concatenate(chain_frag_atom_pos, axis=0)
                    templ_frag_atom_pos = np.concatenate(templ_frag_atom_pos, axis=0)
                    gap_atom_pos = np.concatenate(gap_atom_pos, axis=0)
                    gap_atom_mask = np.concatenate(gap_atom_mask, axis=0)
                    gap_res_type = np.concatenate(gap_res_type, axis=0)

                    # if no correspondance can be found
                    if len(chain_frag_atom_pos) == 0 or len(templ_frag_atom_pos) == 0:
                        has_correspondance = False
                        print("Fix chain {} gap {} failed continue on next".format(i, idx))
                        continue

                    # superpose
                    print("Intend to align {} residues for fixing a gap with {} residues".format(len(chain_frag_atom_pos), len(gap_atom_pos)))
                    R, t = kabsch(
                        templ_frag_atom_pos[..., 1, :],
                        chain_frag_atom_pos[..., 1, :],
                    )

                    t_templ_frag_atom_pos = apply(templ_frag_atom_pos, R, t)
                    t_gap_atom_pos = apply(gap_atom_pos, R, t)
                    rms = rmsd(t_templ_frag_atom_pos[..., 1, :], chain_frag_atom_pos[..., 1, :])
                    print("RMSD between matched frags = {:.4f}".format(rms))

                    if rms < rms_cutoff:
                        print("Do fix")
                        # apply shift-field 
                        if shift_field_refine:
                            shift_field = create_shift_field(
                                fixing=chain_frag_atom_pos[..., 1, :],
                                moving=t_templ_frag_atom_pos[..., 1, :],
                                u=15.,
                            )
                            shift_vector = shift_field(t_gap_atom_pos[..., 1, :])
                            t_s_gap_atom_pos = t_gap_atom_pos + shift_vector[..., None, :]
                            rms = rmsd(t_s_gap_atom_pos[..., 1, :], t_gap_atom_pos[..., 1, :])
                            print("RMSD between shifted and original = {:.4f}".format(rms))
                        else:
                            t_s_gap_atom_pos = t_gap_atom_pos

                        # insert the shifted gap atom pos to chain
                        fixed_atom_pos.append(t_s_gap_atom_pos)
                        fixed_atom_mask.append(gap_atom_mask)
                        fixed_res_type.append(gap_res_type)
                    else:
                        print("Ignore fix because RMSD is too large")
                else:
                    # if not gap, use the original atom pos
                    start = frags_and_gaps[idx][0]
                    end = start + len(frags_and_gaps[idx][1])
                    sel_idxs_chain = np.asarray([idx_a2o_chain[x] for x in range(start, end)], dtype=np.int32)
                    mask = sel_idxs_chain != -1

                    fixed_atom_pos.append(chain_atom_pos[sel_idxs_chain[mask]])
                    fixed_atom_mask.append(chain_atom_mask[sel_idxs_chain[mask]])
                    fixed_res_type.append(chain_res_type[sel_idxs_chain[mask]])

            fixed_atom_pos = np.concatenate(fixed_atom_pos, axis=0)
            fixed_atom_mask = np.concatenate(fixed_atom_mask, axis=0)
            fixed_res_type = np.concatenate(fixed_res_type, axis=0)

            # only keep CA available
            #fixed_atom_mask[..., :] = False
            #fixed_atom_mask[..., 1] = True

            # append to final result
            all_fixed_atom_pos.append(fixed_atom_pos)
            all_fixed_atom_mask.append(fixed_atom_mask)
            all_fixed_res_type.append(fixed_res_type)

            print("Done fix chain {}".format(i))
            print('-'*80)



        # trim fixed chains to keep only the matched residues
        n_res_gap_cutoff = 10
        d_caca_cutoff = 6.0
        trimmed_atom_pos = []
        trimmed_atom_mask = []
        trimmed_res_type = []
        trimmed_res_idx = []
        chain_seq_align = dict()
        for i in range(len(all_fixed_atom_pos)):
            chain_seq = "".join([index_to_restype_1[x] for x in all_fixed_res_type[i]])

            seqid_best = -1e6
            idx_best = -1
            for k in range(len(seqs)):
                result = nwalign_fast(
                    chain_seq, 
                    seqs[k], 
                    lib_dir=lib_dir, temp_dir=seq_temp_dir, verbose=verbose, namea=f"fixed_chain_{i}", nameb=f"seq_{k}"
                )
                seqAid = result[3]
                seqBid = result[5]

                if seqAid > seqid_best:
                    seqid_best = seqAid
                    idx_best = k

                # record alignment
                key = "{}_{}".format(i, k)
                chain_seq_align[key] = result

            key = "{}_{}".format(i, idx_best)
            result = chain_seq_align[key]

            # get the aligned residue index
            gaps = split_gaps_seq(
                result[1],
            )
            keep_idxs = np.ones(len(result[0]), dtype=bool)
            for gap in gaps:
                start = gap[0]
                l = len(gap[1])
                if l > n_res_gap_cutoff:
                    for k in range(start, start + l):
                        keep_idxs[k] = False

            idxs_a2o = idx_aligned_to_original(result[0])
            keep_idxs_a2o = idxs_a2o[keep_idxs]
            keep_idxs_a2o = keep_idxs_a2o[keep_idxs_a2o != -1]

            # add trim atom pos, atom mask and res type
            atom_pos = all_fixed_atom_pos[i][keep_idxs_a2o]
            trimmed_atom_pos.append(atom_pos)
            atom_mask = all_fixed_atom_mask[i][keep_idxs_a2o]
            trimmed_atom_mask.append(atom_mask)
            res_type = all_fixed_res_type[i][keep_idxs_a2o]
            trimmed_res_type.append(res_type)

            # for each residue, if the distance to the next residue is larger than 6A add a gap index
            res_idx = keep_idxs_a2o.copy()
            for k in range(len(atom_pos) - 1):
                d_caca = np.linalg.norm(atom_pos[k][1] - atom_pos[k+1][1])
                if d_caca > d_caca_cutoff:
                    res_idx[k+1:] += 1
                    print("Found large Ca-Ca gap of distance = {:.4f}".format(d_caca))

            trimmed_res_idx.append(res_idx)

        # if not fixed some chains
        if not len(trimmed_atom_pos) > 0:
            raise Exception("Unable to fix any chains by imp")

        # write un-trimmed chains
        fout = pjoin(out_dir, 'imp_chains_untrimmed.cif')
        chains_atom_pos_to_pdb(
            filename=fout,
            chains_atom_pos=all_fixed_atom_pos,
            chains_atom_mask=all_fixed_atom_mask,
            chains_res_types=all_fixed_res_type,
            suffix='cif',
        )
        print("Write untrimmed chains to {}".format(fout))

        # write trimmed chains
        fout = pjoin(out_dir, 'imp_chains_trimmed.cif')
        chains_atom_pos_to_pdb(
            filename=fout,
            chains_atom_pos=trimmed_atom_pos,
            chains_atom_mask=trimmed_atom_mask,
            chains_res_types=trimmed_res_type,
            chains_res_idxs=trimmed_res_idx,
            suffix='cif',
        )
        print("Write trimmed chains to {}".format(fout))
    except Exception as e:
        fchains = args.chain
        print("Error occurs -> {}".format(e))
        print("WARNING cannot imp chains by templates")
        print("WARNING will write denovo built chains instead")

        denovo_atom_pos = []
        denovo_atom_mask = []
        denovo_res_type = []
        denovo_res_idx = []
        for k, fchain in enumerate(fchains):
            atom_pos, atom_mask, res_type, res_idx, _ = read_pdb(fchain, keep_valid=False)
            denovo_atom_pos.append(atom_pos)
            denovo_atom_mask.append(atom_mask)
            denovo_res_type.append(res_type)
            denovo_res_idx.append(res_idx)

        #denovo_atom_pos = np.concatenate(denovo_atom_pos, axis=0)
        #denovo_atom_mask = np.concatenate(denovo_atom_mask, axis=0)
        #denovo_res_type = np.concatenate(denovo_res_type, axis=0)
        #denovo_res_idx = np.concatenate(denovo_res_idx, axis=0)

        fout = pjoin(out_dir, "imp_chains_untrimmed.cif")
        chains_atom_pos_to_pdb(
            filename=fout,
            chains_atom_pos=denovo_atom_pos,
            chains_atom_mask=denovo_atom_mask,
            chains_res_types=denovo_res_type,
            chains_res_idxs=denovo_res_idx,
            suffix=fout.split('.')[-1],
        )
        print(f"Write original chains to {fout}")

        fout = pjoin(out_dir, "imp_chains_trimmed.cif")
        chains_atom_pos_to_pdb(
            filename=fout,
            chains_atom_pos=denovo_atom_pos,
            chains_atom_mask=denovo_atom_mask,
            chains_res_types=denovo_res_type,
            chains_res_idxs=denovo_res_idx,
            suffix=fout.split('.')[-1],
        )
        print(f"Write original chains to {fout}")

    te = time.time()
    #print("Time consuming = {:.4f}".format(te - ts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", "-s", help="Input sequence")
    parser.add_argument("--chain", "-c", nargs='+', help="Denovo built chains")
    parser.add_argument("--template", "-t", type=str, nargs='+', help="Input template, could be models predicted by AlphaFold/ESMFold")
    parser.add_argument("--lib", "-l", help="Lib directory")
    parser.add_argument("--output", "-o", help="Output directory", default="./")
    parser.add_argument("--verbose", "-v", action='store_true', help="Whether to print log to stdout")
    args = parser.parse_args()
    main(args)
