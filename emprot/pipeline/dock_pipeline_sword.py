import os
import sys
import time
import glob
import argparse
import tempfile
import numpy as np

from emprot.io.pdbio import read_pdb, chains_atom_pos_to_pdb

from emprot.utils.misc_utils import abspath, pjoin
from emprot.utils.domain import (
    run_unidoc, 
    parse_unidoc_result, 
    annotate_pdb_with_domains,
    convert_domains_to_1d_repr,
    merge_domains_simple,
)

from emprot.utils.dock_and_refine import run_dock_and_refine_pipeline

def main(args):
    ts = time.time()

    fpdbs = args.pdb
    fmap = args.map
    lib_dir = abspath(args.lib)
    out_dir = abspath(args.output)
    os.makedirs(out_dir, exist_ok=True)
    verbose = args.verbose


    # read structure
    fpdbs_valid = []
    for i, fpdb in enumerate(fpdbs):
        try:
            atom_pos, atom_mask, res_type, res_idx, chain_idx, bfactor = read_pdb(fpdb, return_bfactor=True)
            fpdbs_valid.append(fpdb)
        except Exception as e:
            print("Error occurs -> {}".format(e))
            print("WARNING cannot read any structure from {}".format(fpdb))
            print("WARNING ignore this structure")

    # parse docking options
    resolution = args.resolution
    mode = args.mode
    nt = args.nt
    thresh = args.thresh
    angle_step = args.angle_step
    fgrid = args.fgrid
    sgrid = args.sgrid

    assert 2.0 <= resolution <= 8.0, "2.0 <= resolution <= 8.0, but got {:.2f}".format(resolution)
    assert nt >= 1, "nt >= 1 but got {}".format(nt)
    assert thresh >= 0.0, "thresh >= 0.0 but got {:.6f}".format(thresh)
    assert angle_step >= 12.0, "angle_step >= 12.0 but got {:.2f}".format(angle_step)
    assert fgrid >= 1.0, "fgrid >= 1.0 but got {:.2f}".format(fgrid)
    assert sgrid >= 0.5, "sgrid >= 0.5 but got {:.2f}".format(sgrid)

    dock_args = {
        'resolution': resolution,
        'mode': mode,
        'nt': nt,
        'thresh': thresh,
        'angle_step': angle_step,
        'fgrid': fgrid,
        'sgrid': sgrid,
    }

    print("Docking args")
    for k, v in dock_args.items():
        print("{} -> {}".format(k, v))

    # Making temp dirs
    dock_chain_temp_dir = pjoin(out_dir, "chain")
    os.makedirs(dock_chain_temp_dir, exist_ok=True)

    ###########################################################################
    ###########################################################################
    ################ Fitting of individual chains #############################
    ###########################################################################
    ###########################################################################
    n_max_chains = 60
    print("Docking in chain level")
    print("Intend to dock {} chains".format(len(fpdbs_valid)))
    if len(fpdbs_valid) > n_max_chains:
        print("WARNING too many chains to be docked -> {} maximum is -> {}".format(len(fpdbs_valid), n_max_chains))
        print("WARNING exit now")
        te = time.time()
        #print("Time consuming = {:.4f}".format(te-ts), flush=True)
    elif len(fpdbs_valid) == 0:
        print("WARNING no chains can be found")
        print("WARNING exit now")
        te = time.time()
        #print("Time consuming = {:.4f}".format(te-ts), flush=True)
    else:
        # parse domains
        dock_map_dir = fmap
        dock_pdb_dir = []
        dock_pdb_domains = []
        print("Parsing smaller domains for chains")
    
        # parsing domains
        # use unidoc
        for fpdb in fpdbs_valid:
            #unidoc = run_unidoc(fpdb, lib_dir=lib_dir, verbose=verbose, domain_type='merged')
            unidoc = run_unidoc(fpdb, lib_dir=lib_dir, verbose=verbose, domain_type='unmerged')
            domains = parse_unidoc_result(unidoc)
            dock_pdb_dir.append(fpdb)
            dock_pdb_domains.append(domains)

            print(domains)

        # run dock and refine
        print("Running dock and refine 2/2")
        fpdbout = pjoin(out_dir, "fitted_chains.cif")
        flogout = pjoin(out_dir, "fitted_chains.log")
        assigned_chain_idx = run_dock_and_refine_pipeline(
            map_dir=dock_map_dir,
            chains_pdb_dir=dock_pdb_dir,
            chains_domains=dock_pdb_domains,
            out_dir=fpdbout,
            log_dir=flogout,
            lib_dir=lib_dir,
            verbose=verbose,
            temp_dir=dock_chain_temp_dir,
            **dock_args,
        )
        print("Write docked pdbs to {}".format(fpdbout))

        # split to domains and write to a new file
        # cannot use dock_pdb_domains here because some chains maybe un-docked
        # we should use domain parser instead
        # TODO use above domain result since a slight displacement leads to different domain parsing result
        fpdbout = pjoin(out_dir, "fitted_chains.cif")
        temp_dir = pjoin(out_dir, "chain_to_domain")
        os.makedirs(temp_dir, exist_ok=True)
        atom_pos, atom_mask, res_type, res_idx, chain_idx, bfactor = read_pdb(fpdbout, keep_valid=False, return_bfactor=True)
        n_chain = chain_idx.max() + 1

        doms_atom_pos = []
        doms_atom_mask = []
        doms_res_type = []
        doms_res_idx = []
        doms_bfactor = []
        doms_chain_idx = []

        n_total = 0
        n_total_res = 0
        for i in range(n_chain):
            n_chain_dom_res = 0
            chain_mask = chain_idx == i
            # re-write chain
            # res_idxs start from 0
            fout = pjoin(temp_dir, f"fitted_chain_{i}.pdb")
            chains_atom_pos_to_pdb(
                filename=fout,
                chains_atom_pos=[atom_pos[chain_mask]],
                chains_atom_mask=[atom_mask[chain_mask]],
                chains_res_types=[res_type[chain_mask]],
                chains_bfactors=[bfactor[chain_mask]],
                suffix='pdb',
            )
            print("Re-write fitted chain to {}".format(fout))

            # parse chain domain
            domains = run_unidoc(fout, lib_dir=lib_dir, temp_dir=temp_dir, verbose=verbose, domain_type='merged')
            domains = parse_unidoc_result(domains)
            d1d = convert_domains_to_1d_repr(domains)

            if len(d1d) != len(atom_pos[chain_mask]):
                print("Impossible")
                continue
            
            n_domain = d1d.max() + 1
            for k in range(n_domain):
                domain_mask = d1d == k
                doms_atom_pos.append(atom_pos[chain_mask][domain_mask])
                doms_atom_mask.append(atom_mask[chain_mask][domain_mask])
                doms_res_type.append(res_type[chain_mask][domain_mask])
                # res_idxs from original
                doms_res_idx.append(res_idx[chain_mask][domain_mask])
                doms_bfactor.append(bfactor[chain_mask][domain_mask])
                doms_chain_idx.append(n_total)
                n_total += 1

                n_chain_dom_res += len(atom_pos[chain_mask][domain_mask])

            n_total_res += n_chain_dom_res
 
        # write all domains
        fout = pjoin(out_dir, "fitted_chains_domains.cif")
        chains_atom_pos_to_pdb(
            filename=fout,
            chains_atom_pos=doms_atom_pos,
            chains_atom_mask=doms_atom_mask,
            chains_res_types=doms_res_type,
            # use new res idx
            chains_bfactors=doms_bfactor,
            suffix='cif',
        )
        print("Write docked pdbs to {}".format(fout))

    te = time.time()
    #print("Time consuming = {:.4f}".format(te-ts), flush=True)

if __name__ == '__main__':
    script_dir = abspath(os.path.dirname(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", "-p", nargs='+', help="Input templates of AF/ESMFold")
    parser.add_argument("--map", "-m", help="Input predicted main-chain map")
    parser.add_argument("--lib", "-l", help="Lib directory", default=os.path.join(script_dir, ".."))
    parser.add_argument("--verbose", "-v", action='store_true', help="Whether to pring log to stdout")
    parser.add_argument("--output", "-o", help="Output directory", default='./')
    # docking options
    parser.add_argument('--resolution', '--res', '-res', type=float, default=5.0, help="Map resolution")
    parser.add_argument('--mode', '-mode', type=str, default='fast', help="Docking mode, 'normal' or 'fast'")
    parser.add_argument('--nt', '-nt', type=int, default=4, help="Num of threads to accelerate")
    parser.add_argument('--thresh', '-thresh', type=float, default=10, help="Threshold of input map")
    parser.add_argument('--angle_step', '-angle_step', type=float, default=18.0, help="Angle step for FFT sampling")
    parser.add_argument('--fgrid', '-fgrid', type=float, default=5.0, help="FTMatch grid apix")
    parser.add_argument('--sgrid', '-sgrid', type=float, default=2.0, help="Simplex grid apix")
    args = parser.parse_args()
    main(args)
