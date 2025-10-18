import os
import re
import sys
import time
import tqdm
import tempfile
import argparse
import numpy as np

from scipy.spatial import KDTree
from collections import deque

from emprot.io.fileio import (
    getlines,
    writelines,
    extract_lines_by_ca,
    extract_lines_by_res_idx,
)

from emprot.io.seqio import (
    seq_identity,
    get_sequence_from_pdb_lines,
    update_sequence_to_pdb_lines,
    nwalign_fast,
)

from emprot.io.pdbio import (
    read_pdb,
    chains_atom_pos_to_pdb,
)

from emprot.utils.misc_utils import (
    abspath,
    pjoin,
    find_first_not_of,
    find_last_not_of,
    split_array,
)

from emprot.utils.tm_utils import (
    run_TMalign,
)

from emprot.utils.domain import (
    run_stride,
    run_unidoc,
    parse_unidoc_result,
    convert_domains_to_1d_repr,
    detect_large_loops,
    extract_secstr_1d,
)

from emprot.utils.geo import (
    distance,
    apply,
)

from emprot.utils.residue_constants import (
    index_to_restype_3,
    restype_3_to_index,
)

def prune_align(align):
    i = 0
    k = len(align) - 1
    # find first non-space
    while i < len(align):
        if align[i] != ' ':
            break
        i += 1
    # find last non-space
    while k >= 0:
        if align[k] != ' ':
            break
        k -= 1
    "   ADASDS BSFGED D"
    "(i, k) = (3, 17)"
    return (i, k)

def find_matched_frag(seq1, align, seq2, tolerance=2, min_sub_num=5, min_score=0.80, min_seq_id=0.80, verbose=False):
    assert tolerance >= 1
    assert min_sub_num >= 1
    assert min_score >= 0.0
    assert min_seq_id >= 0.0
    assert len(seq1) == len(align) == len(seq2)

    # find fragments and score each fragment
    data = seq2
    pattern = re.compile(r"[^-]+(?:-{1," + str(tolerance) + r"}[^-]+)*")
    matches = list(re.finditer(pattern, data))
    # filter
    matches = [match for match in matches if len(match.group(0)) >= min_sub_num]
   
    seqids = [] 
    scores = []
    for match in matches:
        substr = match.group(0)
        idx = match.start()

        n = 0
        seqid = 0
        score = 0.0
        for k in range(idx, idx + len(substr)):
            if align[k] == ':':
                score += 1.0
            elif align[k] == '.':
                score += 0.2
            if seq1[k] != '-' and seq2[k] != '-':
                n += 1
                if seq1[k] == seq2[k]:
                    seqid += 1
        if n == 0:
            seqid = 0.0
        else:
            seqid = seqid / n
        score = score / len(substr)
        scores.append(score)
        seqids.append(seqid)

        if verbose:
            print(f"sub-frag '{match.group(0)}' start position {match.start()} len {len(substr):d} seqid {seqid:.4f} score {score:.4f}")

    # merge well-matched fragments
    idxs = np.argsort(scores, kind='stable')
    idxs = idxs[::-1]
    idxs = [idx for idx in idxs if scores[idx] >= min_score and seqids[idx] >= min_seq_id]

    # find left fragments
    left_idxs = [i for i in range(len(matches)) if i not in idxs]
    #print(left_idxs)

    # return all fragments
    return [
        (matches[idx].group(0), matches[idx].start(), seqids[idx], scores[idx]) for idx in idxs
    ], [
        (matches[idx].group(0), matches[idx].start(), seqids[idx], scores[idx]) for idx in left_idxs
    ]


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

        verbose = args.verbose
        # the script is to flexibly transform the templates' domains by the following steps
        # 0. Do TMalign between a chain (n fragment) and a template
        # 1. Get the best fit fragment, merge it to other well-matched fragment
        # 2. Keep the minimum "domain" that just include the well-matched fragment(s)
        # 3. Recursively do the unmatched domains

        # preprocess templates
        # split template models into domains
        ftempls = args.template

        # process chains
        fchains = args.chain

        # set up temp dir
        temp_dir = pjoin(out_dir, "alignments")
        os.makedirs(temp_dir, exist_ok=True)
        print(f"Found {len(fchains)} chains")
        print(f"Found {len(ftempls)} templates")


        chain_templ_relation = dict()
        seq_temp_dir = pjoin(temp_dir, "seqs")
        os.makedirs(seq_temp_dir, exist_ok=True)
        for i in range(len(fchains)): 
            # find the best matched template according to seqid
            # convert X to G to avoid Error from protein sequence align
            chain_lines = getlines(fchains[i])
            chain_seq = get_sequence_from_pdb_lines(chain_lines)
            chain_seq = "".join([x if x != "X" else "G" for x in chain_seq])

            seqid_best = -1e6
            idx_best = -1
            for k in range(len(ftempls)):
                templ_lines = getlines(ftempls[k])
                templ_seq = get_sequence_from_pdb_lines(templ_lines)
                templ_seq = "".join([x if x != "X" else "G" for x in templ_seq])

                result = nwalign_fast(chain_seq, templ_seq, lib_dir=lib_dir, temp_dir=seq_temp_dir, verbose=verbose, namea=f"chain_{i}", nameb=f"templ_{k}")
                seqAid = result[3]
                seqBid = result[5]
                #print(result)
                #print(seqAid, seqBid)
                if seqAid > seqid_best:
                    seqid_best = seqAid
                    idx_best = k
            chain_templ_relation[i] = idx_best
        for k, v in chain_templ_relation.items():
            print("Best sequence fit for chain {} is {}".format(fchains[k], ftempls[v]))


        # split template into domains
        # further, if a domain contains too many loop regions, like, > 30 consecutive residues
        # split the domain into fragments too
        templs_domains = []
        templs_domains_1d = []
        for i in range(len(ftempls)):
            # init domain
            templ_lines0 = getlines(ftempls[i])
            ftempl = pjoin(temp_dir, f"templ_{i}_all.pdb")
            writelines(ftempl, templ_lines0)

            # parse domain
            unidoc_result = run_unidoc(ftempl, chain='A', lib_dir=lib_dir, temp_dir=temp_dir, domain_type='unmerged')
            templ_domains = parse_unidoc_result(unidoc_result)
            templ_domains_1d = convert_domains_to_1d_repr(templ_domains)

            if args.verbose:
                print("Unidoc domains {}".format(unidoc_result))

            # assign secondary structure for each domain
            new_templ_domains = []
            n_extend = 5
            for k, domain in enumerate(templ_domains):
                # write to file
                domain_res_idxs = []
                for subdomain in domain:
                    start, end = subdomain
                    domain_res_idxs.extend(list(range(start, end + 1)))
                domain_lines = extract_lines_by_res_idx(templ_lines0, domain_res_idxs)
                ftempl_domain = pjoin(temp_dir, f"templ_{i}_domain_{k}.pdb")
                writelines(ftempl_domain, domain_lines)
                #print(domain_res_idxs)

                # run stride on file
                slines = run_stride(ftempl_domain, lib_dir=lib_dir, temp_dir=None, verbose=verbose)
                ftempl_secstr = extract_secstr_1d(slines)
                large_loops = detect_large_loops(ftempl_secstr)

                # further split the domain into small fragments if has large loops
                # for AF2 predicted models, there are many loops if you provide a full-length sequence
                # so this step can help reduce the loop residues inside a domain
                if large_loops:
                    new_domain = []
                    seps = []
                    for loop in large_loops:
                        # add a small extension on both end
                        loop_start_idx = loop[0] + n_extend
                        loop_end_idx = loop[0] + len(loop[1]) - 1 - n_extend

                        # normally, impossible to happen
                        if not loop_start_idx <= loop_end_idx:
                            continue

                        seps.append(loop_start_idx)
                        seps.append(loop_end_idx)
                        #print(loop_start_idx, loop_end_idx)

                    new_domain_res_idxs = split_array(domain_res_idxs, seps)
                    new_domain = [[[x[0], x[-1]]] for x in new_domain_res_idxs]

                    # update
                    new_templ_domains.extend(new_domain)
                else:
                    new_templ_domains.append(domain)
            print("Original domains {}".format(templ_domains))
            templ_domains = new_templ_domains
            templ_domains_1d = convert_domains_to_1d_repr(templ_domains)
            print("Updated  domains {}".format(templ_domains))

            # append to list
            templs_domains.append(templ_domains)
            templs_domains_1d.append(templ_domains_1d)
        #print(templs_domains)
        #print(templs_domains_1d)        
        #exit() 

        # Should sort the domains
        #[[[1034, 1074]], [[1075, 1105]], [[1125, 1310]], [[1021, 1033], [1106, 1124], [1311, 1388]], [[82, 120]], [[0, 81], [121, 256]], [[257, 373]], [[374, 403]], [[425, 464]], [[466, 509]], [[422, 424], [465, 465], [510, 545]], [[404, 421], [546, 659]], [[679, 716]], [[660, 678], [717, 889]], [[890, 1020], [1389, 1639]], [[1640, 1687]]]


        # save fixed template coordinates
        fix_chains_atom_pos = []
        fix_chains_atom_mask = []
        fix_chains_res_type = []
        fix_chains_res_idx = []

        # save chain lines
        denovo_chain_lines = []

        # loop for each chain
        max_layer = 20
        for cidx in range(len(fchains)):
            print("-"*80)
            print(f"Start searching for chain {cidx}")
            print("-"*80)

            # find best fit template
            tidx = chain_templ_relation[cidx]

            templ_domains = templs_domains[tidx]
            templ_domains_1d = templs_domains_1d[tidx]

            # read pdb
            templ_atom_pos, templ_atom_mask, templ_res_type, templ_res_idx, _ = read_pdb(ftempls[tidx])

            # iterative alignment
            # init lines
            templ_lines0 = getlines(ftempls[tidx])
            ca_templ_lines0 = extract_lines_by_ca(templ_lines0)
            chain_lines0 = getlines(fchains[cidx])
            ca_chain_lines0 = extract_lines_by_ca(chain_lines0)
            q = deque()
            q.append((ca_templ_lines0, ca_chain_lines0))

            fixed_domains = []
            layer = 0
            while q:
                N = len(q)
                for n in range(N):
                    if args.verbose:
                        print("Start search at layer {} node {}".format(layer, n))

                    # pop element in queue
                    e = q.pop()
                    templ_lines = e[0]
                    chain_lines = e[1]

                    # dump chains to files
                    templ_prefix = f"templ_l_{layer}_n_{n}"
                    chain_prefix = f"chain_l_{layer}_n_{n}"
                    ftempl = pjoin(temp_dir, templ_prefix + ".pdb")
                    fchain = pjoin(temp_dir, chain_prefix + ".pdb")
                    writelines(ftempl, templ_lines)
                    writelines(fchain, chain_lines)

                    # run initial TMalign
                    result, R, t = run_TMalign(ftempl, fchain, lib_dir=lib_dir, d=2.0, description=templ_prefix + "_onto_" + chain_prefix, temp_dir=temp_dir, verbose=verbose)
                    # check if we have run TMalign succesfully
                    # if not continue on next node
                    if result is None:
                        print("No TMalign result for {} and {}".format(templ_prefix, chain_prefix))
                        continue

                    align_result = result[-6:-3]
                    #print(align_result)

                    # save idxs
                    idxs0_a2o = np.asarray(idx_aligned_to_original(align_result[0]), dtype=np.int32)
                    idxs1_a2o = np.asarray(idx_aligned_to_original(align_result[2]), dtype=np.int32)

                    # score each fragments
                    frags_good, frags_bad = find_matched_frag(align_result[0], align_result[1], align_result[2], verbose=args.verbose)
                    # if no good fragments, continue on next node
                    if not frags_good:
                        print(f"No good fragments on layer {layer} node {n}")
                        continue

                    start = int(1e6)
                    end = -1
                    #print("Good fragments")
                    for frag in frags_good:
                        length = len(frag[0])
                        start_idx = frag[1]
                        idxs0 = idxs0_a2o[start_idx : start_idx + length]
                        #print(idxs0)

                        # find start and end for structure 0
                        s = find_first_not_of(idxs0, -1)
                        e = find_last_not_of(idxs0, -1)
                        if s == -1 or e == -1:
                            continue
                        s = idxs0[s]
                        e = idxs0[e]
                        start = min(s, start)
                        end = max(e, end)
                        
                    # get the matched minimum "domain"
                    select_doms = []
                    if end > start:
                        doms, counts = np.unique(templ_domains_1d[start:end+1], return_counts=True)
                        for dom, count in zip(doms, counts):
                            if count / (end - start + 1) > 0.10:
                                select_doms.append(dom)
                    print("Select domains {}".format(select_doms))


                    # run TMalign again to determine the final transform
                    sel_idxs = []
                    for frag in frags_good:
                        length = len(frag[0])
                        start_idx = frag[1]
                        sel_idxs.extend(list(range(start_idx, start_idx + length)))
                    origial_idxs0 = [idxs0_a2o[x] for x in sel_idxs]
                    origial_idxs1 = [idxs1_a2o[x] for x in sel_idxs]
                    #print(origial_idxs0)
                    #print(origial_idxs1)
                    # should not have -1 idx
                    sel_idxs = [i for i in range(len(sel_idxs)) if origial_idxs0[i] != -1 and origial_idxs1[i] != -1]
                    original_idxs0 = [origial_idxs0[i] for i in sel_idxs]
                    original_idxs1 = [origial_idxs1[i] for i in sel_idxs]
                    #print(origial_idxs0)
                    #print(origial_idxs1)
                    sel_templ_lines = [templ_lines[x] for x in original_idxs0]
                    sel_chain_lines = [chain_lines[x] for x in original_idxs1]
                    #print(sel_templ_lines)
                    #print(sel_chain_lines)
                    sel_ca_templ_lines = extract_lines_by_ca(sel_templ_lines)
                    sel_ca_chain_lines = extract_lines_by_ca(sel_chain_lines)
                    #print(sel_ca_templ_lines)
                    #print(sel_ca_chain_lines)
                    sel_templ_seq = get_sequence_from_pdb_lines(sel_ca_templ_lines)
                    sel_chain_seq = get_sequence_from_pdb_lines(sel_ca_chain_lines)

                    seq_id = seq_identity(sel_templ_seq, sel_chain_seq)

                    if args.verbose:
                        print("Templ seq {}".format(sel_templ_seq))
                        print("Chain seq {}".format(sel_chain_seq))
                        print("Seq id {:.4f}".format(seq_id))

                    # mutate matched residues in chain
                    mutate_chain_seq = sel_templ_seq
                    mutate_chain_lines = update_sequence_to_pdb_lines(sel_ca_chain_lines, mutate_chain_seq)

                    # save the chain lines
                    denovo_chain_lines.extend(mutate_chain_lines + ["TER\n"])

                    writelines(pjoin(temp_dir, f"sel_templ_l_{layer}_n_{n}.pdb"), sel_templ_lines)
                    writelines(pjoin(temp_dir, f"sel_chain_l_{layer}_n_{n}.pdb"), sel_chain_lines)
                    _, R0, t0 = run_TMalign(
                                    pjoin(temp_dir, f"sel_templ_l_{layer}_n_{n}.pdb"),
                                    pjoin(temp_dir, f"sel_chain_l_{layer}_n_{n}.pdb"),
                                    d=2.0,
                                    lib_dir=lib_dir,
                                    temp_dir=None,
                                    verbose=verbose,
                                )
                    #print(R)
                    #print(R0)
                    #print(t)
                    #print(t0)
                    if R0 is not None:
                        print("Re-align to refine transform")
                        R = R0
                        t = t0

                    # save the corresponding domain structure
                    if select_doms:
                        # 2025-04-20 reorder domain
                        select_templ_mask = np.zeros_like(templ_res_type).astype(bool)

                        for dom in select_doms:
                            mask = templ_domains_1d == dom
                            select_templ_mask[mask] = True

                        select_atom_pos = templ_atom_pos[select_templ_mask]
                        select_atom_mask = templ_atom_mask[select_templ_mask]
                        select_res_type = templ_res_type[select_templ_mask]
                        select_res_idx = templ_res_idx[select_templ_mask]

                        #print(select_res_idx)
                        #exit()

                        # Previous
                        #select_atom_pos = []
                        #select_atom_mask = []
                        #select_res_type = []
                        #select_res_idx = []
                        #for dom in select_doms:
                        #    mask = templ_domains_1d == dom
                        #    select_atom_pos.append(templ_atom_pos[mask])
                        #    select_atom_mask.append(templ_atom_mask[mask])
                        #    select_res_type.append(templ_res_type[mask])
                        #    select_res_idx.append(templ_res_idx[mask])
                        #select_atom_pos = np.concatenate(select_atom_pos, axis=0)
                        #select_atom_mask = np.concatenate(select_atom_mask, axis=0)
                        #select_res_type = np.concatenate(select_res_type, axis=0)
                        #select_res_idx = np.concatenate(select_res_idx, axis=0)

                        # apply
                        if len(select_atom_pos) > 0:
                            select_atom_pos = apply(select_atom_pos, R, t)
                            fix_chains_atom_pos.append(select_atom_pos)
                            fix_chains_atom_mask.append(select_atom_mask)
                            fix_chains_res_type.append(select_res_type)
                            fix_chains_res_idx.append(select_res_idx)

                    # exclude the matched frags and put into queue
                    #print("Bad fragments")
                    bad_idxs = []
                    for frag in frags_bad:
                        length = len(frag[0])
                        start_idx = frag[1]
                        idxs1 = idxs1_a2o[start_idx : start_idx + length]
                        bad_idxs.extend(idxs1)
                    bad_idxs = [idx for idx in bad_idxs if idx != -1]

                    # TMalign requires at least >= 3 residues
                    # we further tighten this restraint
                    if len(bad_idxs) >= 5:
                        bad_lines = [chain_lines[x] for x in bad_idxs]
                        q.append((templ_lines, bad_lines))

                # add up layer
                layer += 1

                # to avoid to stuck in dead loop
                # nearly impossible to happen
                if layer > max_layer:
                    print("WARNING Too deep layer end search now")
                    break

            print("-"*80)
            print(f"End searching for chain {cidx}")
            print("-"*80)


        # if not fixed some chains
        if not len(fix_chains_atom_pos) > 0:
            raise Exception("Unable to fix any chains by fix")

        # output templs
        fout = pjoin(out_dir, "fix_templs.cif")
        chains_atom_pos_to_pdb(
            filename=fout,
            chains_atom_pos=fix_chains_atom_pos,
            chains_atom_mask=fix_chains_atom_mask,
            chains_res_types=fix_chains_res_type,
            chains_res_idxs=fix_chains_res_idx,
            suffix=fout.split('.')[-1],
        )
        print(f"Write fixed templates to {fout}")

        # the following has very little help to the RMSD performance
        # but keep it anyway
        # use predicted Ca coords for templates
        chain_ca_pos = []
        for i in range(len(fchains)):
            chain_lines0 = getlines(fchains[i])
            ca_chain_lines = extract_lines_by_ca(chain_lines0)
            for line in ca_chain_lines:
                chain_ca_pos.append( [float(line[k:k+8]) for k in [30, 38, 46]] )
        chain_ca_pos = np.asarray(chain_ca_pos)
        print("Total {} denovo coords".format(len(chain_ca_pos)))
        d0 = 1.0

        visited = set()
        tree = KDTree(chain_ca_pos)
        for i in range(len(fix_chains_atom_pos)):
            idxs = tree.query_ball_point(fix_chains_atom_pos[i][..., 1, :], d0)
            for k, idx in enumerate(idxs):
                # if none is near or more than 1 is near
                if len(idx) == 0 or len(idx) >= 2:
                    continue
                if idx[0] in visited:
                    continue
                visited.add(idx[0])

                # shift vector
                v = chain_ca_pos[idx[0]] - fix_chains_atom_pos[i][k][1]
                # shift
                fix_chains_atom_pos[i][k] += v

        # if not fixed some chains
        if not len(fix_chains_atom_pos) > 0:
            raise Exception("Unable to fix any chains by fix")

        fout = pjoin(out_dir, "fix_chains_templs.cif")
        chains_atom_pos_to_pdb(
            filename=fout,
            chains_atom_pos=fix_chains_atom_pos,
            chains_atom_mask=fix_chains_atom_mask,
            chains_res_types=fix_chains_res_type,
            chains_res_idxs=fix_chains_res_idx,
            suffix=fout.split('.')[-1],
        )
        print(f"Write fixed templates to {fout}")

    # at least write input denovo chains
    except Exception as e:
        fchains = args.chain
        print("Error occurs -> {}".format(e))
        print("WARNING cannot fix chains by templates")
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

        fout = pjoin(out_dir, "fix_chains_templs.cif")
        chains_atom_pos_to_pdb(
            filename=fout,
            chains_atom_pos=denovo_atom_pos,
            chains_atom_mask=denovo_atom_mask,
            chains_res_types=denovo_res_type,
            chains_res_idxs=denovo_res_idx,
            suffix=fout.split('.')[-1],
        )
        print(f"Write original chains to {fout}")
        
        fout = pjoin(out_dir, "fix_templs.cif")
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
    script_dir = abspath(os.path.dirname(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", "-s", help="Input sequence")
    parser.add_argument("--chain", "-c", nargs='+', help="Denovo built chains")
    parser.add_argument("--template", "-t", type=str, nargs='+', help="Input template, could be models predicted by AlphaFold/ESMFold")
    parser.add_argument("--lib", "-l", help="Lib directory", default=pjoin(script_dir, ".."))
    parser.add_argument("--output", "-o", help="Output directory", default="./")
    parser.add_argument("--verbose", "-v", action='store_true', help="Whether to print log to stdout")
    args = parser.parse_args()
    main(args)
