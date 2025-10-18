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

    # estimate the number of predicted CAs
    pass

    # split template to large domains
    print("Parsing larger domains")
    # each large domain has at least xxx residues
    n_min_res_dom_ratio = 0.05
    n_min_res_dom = 50
    # each chain split to at least xxx domains
    n_min_dom = 1

    large_domains = []
    fdoms = []
    fpdbs_valid = []
    for i, fpdb in enumerate(fpdbs):
        try:
            # for each chain
            templ_dir = os.path.dirname(fpdb)

            atom_pos, atom_mask, res_type, res_idx, chain_idx, bfactor = read_pdb(fpdb, return_bfactor=True)
            unidoc = run_unidoc(fpdb, lib_dir=lib_dir, verbose=verbose, domain_type='merged')
            domains = parse_unidoc_result(unidoc)
            print("Before merging domains = {}".format(domains))
            # merge to large domains
            n_min_res_dom_cutoff = max(n_min_res_dom, int(n_min_res_dom_ratio * len(atom_pos)))
            print("Minimum domain res num = max({}, {}) = {}".format(n_min_res_dom, int(n_min_res_dom_ratio * len(atom_pos)), n_min_res_dom_cutoff))
            domains = merge_domains_simple(domains, n_min_res_dom_cutoff)
            print("After  merging domains = {}".format(domains))
            d1d = convert_domains_to_1d_repr(domains)

            # split to large domains
            for k in range(len(domains)):
                idx = d1d == k
                sel_atom_pos = atom_pos[idx]
                sel_atom_mask = atom_mask[idx]
                sel_res_type = res_type[idx]
                sel_res_idx = res_idx[idx]
                sel_bfactor = bfactor[idx]

                fdom = pjoin(templ_dir, f"chain_templ_{i}_dom_{k}.pdb")
                fdoms.append(fdom)
                print("Write dom {} to {}".format(k, fdom))
                chains_atom_pos_to_pdb(
                    filename=fdom,
                    chains_atom_pos=[sel_atom_pos],
                    chains_atom_mask=[sel_atom_mask],
                    chains_res_types=[sel_res_type],
                    chains_res_idxs=[sel_res_idx],
                    chains_bfactors=[sel_bfactor],
                    suffix='pdb'
                )

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
    dock_domain_temp_dir = pjoin(out_dir, "domain")
    dock_chain_temp_dir = pjoin(out_dir, "chain")
    dock_complex_temp_dir = pjoin(out_dir, "complex")

    os.makedirs(dock_domain_temp_dir, exist_ok=True)
    os.makedirs(dock_chain_temp_dir, exist_ok=True)
    os.makedirs(dock_complex_temp_dir, exist_ok=True)

    # renumber res idx
    fdoms_renum = []
    fdoms_res_idx = []
    for i, fn in enumerate(fdoms):
        atom_pos, atom_mask, res_type, res_idx, chain_idx, bfactor = read_pdb(fn, return_bfactor=True)
        filename = pjoin(dock_domain_temp_dir, f"renum_dom_{i}.pdb")
        chains_atom_pos_to_pdb(
            filename=filename,
            chains_atom_pos=[atom_pos],
            chains_atom_mask=[atom_mask],
            # res_idx are renumbered
            chains_res_types=[res_type],
            chains_bfactors=[bfactor],
            suffix='pdb',
        )
        fdoms_renum.append(filename)
        fdoms_res_idx.append(res_idx)
        print("Renumber res idx for {}".format(fn))
    
    fpdbs_valid_renum = []
    fpdbs_valid_res_idx = []
    for i, fn in enumerate(fpdbs_valid):
        atom_pos, atom_mask, res_type, res_idx, chain_idx, bfactor = read_pdb(fn, return_bfactor=True)
        filename = pjoin(dock_chain_temp_dir, f"renum_chain_{i}.pdb")
        chains_atom_pos_to_pdb(
            filename=filename,
            chains_atom_pos=[atom_pos],
            chains_atom_mask=[atom_mask],
            # res_idx are renumbered
            chains_res_types=[res_type],
            chains_bfactors=[bfactor],
            suffix='pdb',
        )
        fpdbs_valid_renum.append(filename)
        fpdbs_valid_res_idx.append(res_idx)
        print("Renumber res idx for {}".format(fn))


    # replace
    fdoms = fdoms_renum
    fpdbs_valid = fpdbs_valid_renum
    n_max_doms = 60
    n_max_chains = 40
    ###########################################################################
    ###########################################################################
    ################ Fitting of large domains #################################
    ###########################################################################
    ###########################################################################
    if args.domain:
        print("Docking in domain level")
        print("Intend to dock {} domains".format(len(fdoms)))
        if len(fdoms) > n_max_doms:
            print("WARNING too many domains to be docked -> {}".format(len(fdoms)))
            print("WARNING exit now")
            te = time.time()
            #print("Time consuming = {:.4f}".format(te-ts), flush=True)
        elif len(fdoms) == 0:
            print("WARNING no domains can be found")
            print("WARNING exit now")
            te = time.time()
            #print("Time consuming = {:.4f}".format(te-ts), flush=True)
        else:
            # parse domains
            dock_map_dir = fmap
            dock_pdb_dir = []
            dock_pdb_domains = []
    
            print("Parsing smaller domains for large domains")
            for fpdb in fdoms:
                unidoc = run_unidoc(fpdb, lib_dir=lib_dir, verbose=verbose, domain_type='unmerged')
                domains = parse_unidoc_result(unidoc)
                dock_pdb_dir.append(fpdb)
                dock_pdb_domains.append(domains)
    
            # run dock and refine
            print("Running dock and refine 1/2")
            fpdbout = pjoin(out_dir, "fitted_domains.cif")
            flogout = pjoin(out_dir, "fitted_domains.log")
            success = run_dock_and_refine_pipeline(
                map_dir=dock_map_dir,
                chains_pdb_dir=dock_pdb_dir,
                chains_domains=dock_pdb_domains,
                out_dir=fpdbout,
                log_dir=flogout,
                lib_dir=lib_dir,
                verbose=verbose,
                temp_dir=dock_domain_temp_dir,
                **dock_args,
            )
            print("Write docked pdbs to {}".format(fpdbout))

        # end domain fitting

    ###########################################################################
    ###########################################################################
    ################ Fitting of individual chains #############################
    ###########################################################################
    ###########################################################################
    if args.chain:
        print("Docking in chain level")
        print("Intend to dock {} chains".format(len(fpdbs_valid)))
        if len(fpdbs_valid) > n_max_chains:
            print("WARNING too many chains to be docked -> {}".format(len(fpdbs_valid)))
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
    
            for fpdb in fpdbs_valid:
                unidoc = run_unidoc(fpdb, lib_dir=lib_dir, verbose=verbose, domain_type='merged')
                domains = parse_unidoc_result(unidoc)
                dock_pdb_dir.append(fpdb)
                dock_pdb_domains.append(domains)
  
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

        # end chain fitting



    ########################################################################
    ########################################################################
    ################ Fitting of entire complex #############################
    ########################################################################
    ########################################################################
    if args.complex:
        print("Docking in complex level")
        print("Intend to dock 1 complex with {} chains".format(len(fpdbs_valid)))

        # write all chain into complex
        chains_atom_pos = []
        chains_atom_mask = []
        chains_res_type = []
        chains_res_idx = []
        chains_bfactor = []
        chains_chain_idx = []

        for kk, fpdb in enumerate(fpdbs_valid):
            atom_pos, atom_mask, res_type, _, _, bfactor = read_pdb(fpdb, keep_valid=True, return_bfactor=True)
            
            chains_atom_pos.append( atom_pos )
            chains_atom_mask.append( atom_mask )
            chains_res_type.append( res_type )
            chains_bfactor.append( bfactor )
            chains_chain_idx.append( np.asarray([kk] * len(atom_pos), dtype=np.int32) )

        cpx_atom_pos = np.concatenate( chains_atom_pos, axis=0 )
        cpx_atom_mask = np.concatenate( chains_atom_mask, axis=0 )
        cpx_res_type = np.concatenate( chains_res_type, axis=0 )
        cpx_bfactor = np.concatenate( chains_bfactor, axis=0 )
        cpx_chain_idx = np.concatenate( chains_chain_idx, axis=0 )

        fcpx = pjoin(dock_complex_temp_dir, "complex.pdb")
        chains_atom_pos_to_pdb(
            fcpx, 
            [cpx_atom_pos],
            [cpx_atom_mask],
            [cpx_res_type],
            None,
            chains_bfactors=[cpx_bfactor],
            suffix='pdb',
        )
        print("# Write complex to {}".format(fcpx))
 
        # each chain as a domain
        dock_map_dir = fmap
        dock_pdb_dir = [fcpx]
        dock_pdb_domains = []

        res_idx_temp = np.arange(0, len(cpx_atom_pos))
        for i in range(cpx_chain_idx.max() + 1):
            res_range = res_idx_temp[cpx_chain_idx == i]
            if len(res_range) >= 2:
                start, end = res_range[0], res_range[-1]
                dock_pdb_domains.append([[start, end]])

        dock_pdb_domains = [dock_pdb_domains]
        print(dock_pdb_domains)

        # run dock and refine
        print("Running dock and refine 2/2")
        fpdbout = pjoin(out_dir, "fitted_complex.cif")
        flogout = pjoin(out_dir, "fitted_complex.log")
        assigned_chain_idx = run_dock_and_refine_pipeline(
            map_dir=dock_map_dir,
            chains_pdb_dir=dock_pdb_dir,
            chains_domains=dock_pdb_domains,
            out_dir=fpdbout,
            log_dir=flogout,
            lib_dir=lib_dir,
            verbose=verbose,
            temp_dir=dock_complex_temp_dir,
            **dock_args,
        )
        print("Write docked pdbs to {}".format(fpdbout))

        # After complex docking
        # Split complex into chains, then fragments

        if len(assigned_chain_idx) != 1:
            print("# Cannot dock entire complex to map")
        else:
            # split to domains and write to a new file
            # cannot use dock_pdb_domains here because some chains maybe un-docked
            # we should use domain parser instead
            # TODO use above domain result since a slight displacement leads to different domain parsing result

            (
                cpx_atom_pos, 
                cpx_atom_mask,
                cpx_res_type,
                cpx_res_idx,
                _,
                cpx_bfactor, 
            ) = read_pdb(fpdbout, keep_valid=True, return_bfactor=True)

            chains_atom_pos = []
            chains_atom_mask = []
            chains_res_type = []
            chains_res_idx = []
            chains_bfactor = []
            n_chain = cpx_chain_idx.max() + 1

            for i in range(n_chain):
                mask = cpx_chain_idx == i
                chains_atom_pos.append( cpx_atom_pos[mask] )
                chains_atom_mask.append( cpx_atom_mask[mask] )
                chains_res_type.append( cpx_res_type[mask] )
                chains_res_idx.append( cpx_res_idx[mask] )
                chains_bfactor.append( cpx_bfactor[mask] )

            fpdbout = pjoin(out_dir, "fitted_complex_chains.cif")
            chains_atom_pos_to_pdb(
                fpdbout,
                chains_atom_pos,
                chains_atom_mask, 
                chains_res_type,
                chains_res_idx,
                chains_bfactors=chains_bfactor,
            )
            print("# Split docked complex into chains and saved to {}".format(fpdbout))

            # split chains into domains
            temp_dir = pjoin(out_dir, "complex_to_chain_to_domain")
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
            fout = pjoin(out_dir, "fitted_complex_chains_domains.cif")
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

        # end chain fitting




    te = time.time()
    #print("Time consuming = {:.4f}".format(te-ts), flush=True)

if __name__ == '__main__':
    script_dir = os.path.abspath(os.path.dirname(__file__))
    lib_dir = os.path.join(script_dir, "..")
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", "-p", nargs='+', help="Input templates of AF/ESMFold")
    parser.add_argument("--map", "-m", help="Input predicted main-chain map")
    parser.add_argument("--lib", "-l", help="Lib directory", default=lib_dir)
    parser.add_argument("--verbose", "-v", action='store_true', help="Whether to print log to stdout")
    parser.add_argument("--output", "-o", help="Output directory", default='./')
    # options
    parser.add_argument("--domain", help="Docking in domain level", action='store_true')
    parser.add_argument("--chain",  help="Docking in chain level", action='store_true')
    parser.add_argument("--complex", help="Docking in complex level", action='store_true')
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
