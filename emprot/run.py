"""
EMProt main program
"""
import os
import sys
import math
import time
import glob
import shutil
import argparse
import datetime
import warnings
warnings.filterwarnings("ignore")
from emprot.utils.misc_utils import abspath, pjoin
from emprot.io.fileio import copy_file

def main(args):
    time0 = datetime.datetime.utcnow()
    print("# Job start at {}".format(time0))

    # Set up directory
    script_dir = abspath(os.path.dirname(__file__))
    output_dir = abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = abspath(pjoin(output_dir, "temp"))
    os.makedirs(temp_dir, exist_ok=True)
    print("# Output dir set to {}".format(output_dir))
    print("# Temp   dir set to {}".format(temp_dir))


    # 0. preprocess inputs
    if args.skip_preprocess:
        print("# Skip preprocess")
    else:
        start = time.time()
        from emprot.pipeline import preprocess
        preprocess_args = argparse.Namespace()
        preprocess_args.seq = args.seq
        preprocess_args.map = args.map
        preprocess_args.chain = args.chain
        preprocess_args.complex = args.complex

        preprocess_args.output = temp_dir
        preprocess.main(preprocess_args)
        end = time.time()
        print("# Time consupmtion {:.4f} seconds. Done preprocess".format(end - start))

    ##################################
    ### 1. pred mc and aa from map ###
    ##################################
    ncac_temp_dir = pjoin(temp_dir, "ncac")
    os.makedirs(ncac_temp_dir, exist_ok=True)

    if args.skip_dl_mc_aa:
        print("# Skip dl mc aa")
    else:
        start = time.time()
        from emprot.pipeline import pred_mc_aa_gen
        dl_mc_aa_args = argparse.Namespace()
        dl_mc_aa_args.input = pjoin(temp_dir, "format_map.mrc")
        dl_mc_aa_args.output = ncac_temp_dir
   
        dl_mc_aa_args.contour = args.contour
        dl_mc_aa_args.batchsize = args.dl_mc_aa_batchsize
        dl_mc_aa_args.device = args.device
        dl_mc_aa_args.model = pjoin(script_dir, "weights")
        dl_mc_aa_args.stride = args.dl_mc_aa_stride
        dl_mc_aa_args.usecpu = args.device in ['cpu', "CPU", None]
        if dl_mc_aa_args.usecpu:
            print("# WARNING you are running on CPU, which is very slow")
        dl_mc_aa_args.predict_mc = True
        dl_mc_aa_args.predict_aa = True
        pred_mc_aa_gen.main(dl_mc_aa_args)
        end = time.time()
        print("# Time consumption {:.4f} seconds. Done dl mc aa".format(end - start))


        # pmap to ncac
        start = time.time()
        from emprot.pipeline import pmap_to_ca
        pmap_to_ca_args = argparse.Namespace()
        pmap_to_ca_args.output = ncac_temp_dir
        pmap_to_ca_args.lib = script_dir
        pmap_to_ca_args.camap = pjoin(ncac_temp_dir, "ca.mrc")

        # Mean-shift controls
        pmap_to_ca_args.nt = args.nt
        pmap_to_ca_args.thresh = 10.0
        pmap_to_ca_args.res = 6.0
        pmap_to_ca_args.filter = 0.0
        pmap_to_ca_args.dmerge = 1.5

        if args.fast:
            # Fast mode for ultra large proteins e.g. > 50000 residues
            # Use g2p to generate initial points
            from emprot.pipeline import g2p
            g2p_args = argparse.Namespace()
            g2p_args.ratio = 0.10

            g2p_args.map = pjoin(ncac_temp_dir, "ca.mrc")
            g2p_args.output = pjoin(ncac_temp_dir, "g2p_ca.pdb")
            g2p.main(g2p_args)

            pmap_to_ca_args.ca = pjoin(ncac_temp_dir, "g2p_ca.pdb")
        else:
            pmap_to_ca_args.n  = None
            pmap_to_ca_args.ca = None
            pmap_to_ca_args.c  = None

        pmap_to_ca.main(pmap_to_ca_args)

        end = time.time()
        print("# Time consumption {:.4f} seconds. Done pmap to ca".format(end - start))


    ##########################
    ### 2. denovo modeling ###
    ##########################
    if args.skip_denovo:
        print("# Skip denovo")
    else:
        start = time.time()
        # threading, and sequence alignment
        from emprot.pipeline import denovo_pipeline_recycle as denovo_pipeline
        denovo_pipeline_args = argparse.Namespace()

        denovo_pipeline_args.map = pjoin(temp_dir, "format_map.mrc")
        denovo_pipeline_args.protein = pjoin(ncac_temp_dir, "raw_ca.pdb")
        denovo_pipeline_args.model_dir = pjoin(script_dir, "weights", "model_all_atom")
        denovo_pipeline_args.output_dir = pjoin(temp_dir, "denovo")
        denovo_pipeline_args.device = args.device
        denovo_pipeline_args.crop_length = 200
        denovo_pipeline_args.repeat_per_residue = 1
        denovo_pipeline_args.batch_size = 1
        denovo_pipeline_args.fp16 = False
        denovo_pipeline_args.voxel_size = 1.0
        denovo_pipeline_args.refine = False
        denovo_pipeline_args.aamap = pjoin(ncac_temp_dir, "aa_logits.npz")
        denovo_pipeline_args.sequence = pjoin(temp_dir, "format_seq.fasta")
        denovo_pipeline_args.recycle = 3

        denovo_pipeline.main(denovo_pipeline_args)
        end = time.time()

        print("# Time consumption {:.4f} seconds. Done denovo".format(end - start))


    ##########################
    ### No seq information ###
    ##########################
    if args.seq is None:
        print("# No input sequence, some following steps will be skipped")
        args.skip_fix = True
        args.skip_imp = True
        args.skip_dock = True
        args.skip_assemble = True
        args.skip_reorder = True
        args.skip_pmodel = True
        args.skip_final_refine = True

    ##########################################
    ### Use template information #############
    ### First check if have template input ###
    ##########################################
    templs = glob.glob(pjoin(temp_dir, "templs", "templ_*.pdb"))
    if (args.chain is None and args.complex is None) or len(templs) == 0:
        # If have old result, also skippd
        print("# No input template, some following steps will be skipped")
        args.skip_fix = True
        args.skip_imp = True
        args.skip_dock = True
        #args.skip_assemble = True
        args.skip_final_refine = True


    #####################
    ### 3. denovo fix ###
    #####################
    if args.skip_fix:
        print("# Skip fix")
    else:
        start = time.time()
        from emprot.pipeline import denovo_fix_pipeline
        denovo_fix_args = argparse.Namespace()
        denovo_fix_args.seq = pjoin(temp_dir, "format_seq.fasta")

        #denovo_fix_args.chain = glob.glob(pjoin(temp_dir, "denovo", "denovo_chains", "*.pdb"))
        denovo_fix_args.chain = glob.glob(pjoin(temp_dir, "denovo", "recycle_2", "denovo_chains", "*.pdb"))

        denovo_fix_args.template = glob.glob(pjoin(temp_dir, "templs", "templ_*.pdb"))

        denovo_fix_args.lib = script_dir
        fix_temp_dir = pjoin(temp_dir, "fix")
        os.makedirs(fix_temp_dir, exist_ok=True)
        denovo_fix_args.output = fix_temp_dir
        denovo_fix_args.verbose = False
        denovo_fix_pipeline.main(denovo_fix_args)
        end = time.time()
        print("# Time consumption {:.4f} seconds. Done fix".format(end - start))


    #####################
    ### 4. denovo imp ###
    #####################
    if args.skip_imp:
        print("# Skip imp")
    else:
        start = time.time()
        from emprot.pipeline import denovo_imp_pipeline
        denovo_imp_args = argparse.Namespace()
        denovo_imp_args.seq = pjoin(temp_dir, "format_seq.fasta")

        #denovo_imp_args.chain = glob.glob(pjoin(temp_dir, "denovo", "denovo_chains", "*.pdb"))
        denovo_imp_args.chain = glob.glob(pjoin(temp_dir, "denovo", "recycle_2", "denovo_chains", "*.pdb"))

        denovo_imp_args.template = glob.glob(pjoin(temp_dir, "templs", "templ_*.pdb"))
        denovo_imp_args.lib = script_dir
        imp_temp_dir = pjoin(temp_dir, "imp")
        os.makedirs(imp_temp_dir, exist_ok=True)
        denovo_imp_args.output = imp_temp_dir
        denovo_imp_args.verbose = False
        denovo_imp_pipeline.main(denovo_imp_args)
        end = time.time()
        print("# Time consumption {:.4f} seconds. Done imp".format(end - start))


    templ_type = "none"
    templ_type_dir = pjoin(temp_dir, "templs", "type.txt")
    if os.path.exists(templ_type_dir):
        with open(templ_type_dir, 'r') as f:
            templ_type = f.read().strip()
    print("# Template type = {}".format(templ_type))

    ###############
    ### 5. dock ###
    ###############
    if args.skip_dock:
        print("# Skip dock")
    else:
        start = time.time()
        from emprot.pipeline import dock_pipeline
        dock_args = argparse.Namespace()
        dock_args.pdb = glob.glob(pjoin(temp_dir, "templs", "templ_*.pdb"))
        dock_args.map = pjoin(ncac_temp_dir, "mc.mrc")
        dock_args.lib = script_dir
        dock_args.verbose = True

        dock_temp_dir = pjoin(temp_dir, "dock")
        dock_args.output = dock_temp_dir
        os.makedirs(dock_temp_dir, exist_ok=True)

        dock_args.resolution = 5.0
        dock_args.mode = 'fast'
        dock_args.nt = args.nt
        dock_args.thresh = 10.0
        dock_args.angle_step = 18.0
        dock_args.fgrid = 5.0
        dock_args.sgrid = 2.0

        # dock in chain and domain level
        dock_args.chain = True
        dock_args.domain = True

        if templ_type == "complex":
            dock_args.complex = True
        else:
            dock_args.complex = False

        dock_pipeline.main(dock_args)

        end = time.time()
        print("# Time consumption {:.4f} seconds. Done dock".format(end - start))


    #########################
    ### 6. final assemble ###
    #########################
    if args.skip_assemble:
        print("# Skip final assemble")
    else:
        start = time.time()
        from emprot.pipeline import assemble
        assemble_args = argparse.Namespace()
        assemble_args.seq = pjoin(temp_dir, "format_seq.fasta")


        # If have template structure
        if not (templ_type == 'none'):
            assemble_args.pdb = [
                pjoin(temp_dir, "imp",  "imp_chains_trimmed.cif"),
                pjoin(temp_dir, "fix",  "fix_chains_templs.cif"),
                pjoin(temp_dir, "dock", "fitted_domains.cif"),
                pjoin(temp_dir, "dock", "fitted_chains_domains.cif"),
                # also supports other fragments...
            ]

            # If has complex template
            #if templ_type == "complex":
            #    assemble_args.pdb += [
            #        pjoin(temp_dir, "dock", "fitted_complex_chains_domains.cif"), 
            #    ]
        else:
            assemble_args.pdb = [
                #pjoin(temp_dir, "denovo", "denovo_aa.cif"), 
                pjoin(temp_dir, "denovo", "recycle_2", "output.cif"), 
                # only have denovo built fragments
            ]

        assemble_args.map = pjoin(ncac_temp_dir, "ca.mrc")
        assemble_args.verbose = False
        assemble_args.lib = script_dir
        assemble_args.no_split = False

        assemble_temp_dir = pjoin(temp_dir, "assemble")
        os.makedirs(assemble_temp_dir, exist_ok=True)
        assemble_args.out = assemble_temp_dir

        assemble.main(assemble_args)

        end = time.time()
        print("# Time consumption {:.4f} seconds. Done assemble".format(end - start))

    ############################
    ### 7. reorder fragments ###
    ############################
    if args.skip_reorder:
        print("# Skip reorder")
    else:
        start = time.time()
        from emprot.pipeline import reorderx_group
        reorder_args = argparse.Namespace()

        reorder_args.pdb = pjoin(temp_dir, "assemble", "assemble.cif")
        reorder_args.seq = pjoin(temp_dir, "format_seq.fasta")
        reorder_args.chain = glob.glob(pjoin(temp_dir, "templs", "templ_*.pdb"))
        reorder_args.out = pjoin(temp_dir, "reorderx")

        reorder_args.alpha = 0.5
        reorder_args.time_limit = 10
        reorder_args.num_workers = 2
        reorder_args.log_search = False

        reorderx_group.main(reorder_args)
        end = time.time()
        print("# Time consumption {:.4f} seconds. Done reorder".format(end -  start))

    ############################
    ### 8. pick better model ###
    ############################
    #if args.skip_pmodel:
    #    print("# Skip pick model")
    #else:
    #    start = time.time()
    #    from emprot.pipeline import pick_model
    #    pmodel_args = argparse.Namespace()

    #    pmodel_args.assemble = pjoin(temp_dir, "reorderx", "reorder.cif")
    #    pmodel_args.dock = pjoin(temp_dir, "dock", "fitted_chains.cif")
    #    pmodel_args.map = pjoin(temp_dir, "format_map.mrc")
    #    pmodel_args.resolution = args.resolution
    #    pmodel_args.seq = pjoin(temp_dir, "format_seq.fasta")
    #    pmodel_args.output = pjoin(temp_dir, "pmodel")
    #    pmodel_args.bfactor = 0.0

    #    pick_model.main(pmodel_args)

    #    end = time.time()
    #    print("# Time consumption {:.4f} seconds. Done pick model".format(end -  start))



    ################################
    ### 97. all-atom refine again ###
    ################################
    #if args.skip_final_refine:
    #    print("# Skip final refine")
    #else:
    #    start = time.time()
    #    final_output_dir = pjoin(temp_dir, "final")
    #    os.makedirs(final_output_dir, exist_ok=True)

    #    from emprot.pipeline import denovo_pipeline
    #    denovo_pipeline_args = argparse.Namespace()
    #    denovo_pipeline_args.map = pjoin(temp_dir, "format_map.mrc")
    #    denovo_pipeline_args.protein = pjoin(temp_dir, "pmodel", "pmodel.cif")

    #    denovo_pipeline_args.model_dir = pjoin(script_dir, "weights", "model_all_atom")
    #    denovo_pipeline_args.output_dir = pjoin(temp_dir, "assemble")

    #    denovo_pipeline_args.device = args.device
    #    denovo_pipeline_args.crop_length = 200
    #    denovo_pipeline_args.repeat_per_residue = 1
    #    denovo_pipeline_args.batch_size = 1
    #    denovo_pipeline_args.fp16 = False
    #    denovo_pipeline_args.voxel_size = 1.0
    #    denovo_pipeline_args.refine = True
    #   
    #    denovo_pipeline.main(denovo_pipeline_args)

    #    end = time.time()
    #    print("# Time consumption {:.4f} seconds. Done final refine".format(end - start))


    #########################################
    ### 98. copy the final file to output ###
    #########################################
    possible_outputs = [
        #pjoin(temp_dir, "pmodel", "pmodel.cif"),
        pjoin(temp_dir, "reorderx", "reorder.cif"),
        pjoin(temp_dir, "assemble", "assemble.cif"),
        #pjoin(temp_dir, "denovo", "recycle_2", "denovo_aa.cif"),
        pjoin(temp_dir, "denovo", "recycle_2", "output.cif"),
    ]

    has_output = False
    for po in possible_outputs:
        if os.path.exists(po):
            print("# Found model {}".format(po))
            shutil.copyfile(
                po,
                pjoin(output_dir, "output.cif"), 
            )
            has_output = True
            break


    # Copy the other models
    denovo_model_dir = pjoin(temp_dir, "denovo", "recycle_2", "output.cif")
    if os.path.exists(denovo_model_dir):
        fo = pjoin(output_dir, "output_denovo.cif")
        shutil.copyfile(
            denovo_model_dir,
            fo,
        )
        print("# Found de novo model {}".format(denovo_model_dir))
        print("# Copy de novo model to {}".format(fo))


    fit_model_dir = pjoin(temp_dir, "dock", "fitted_chains.cif")
    if os.path.exists(fit_model_dir):
        fo = pjoin(output_dir, "output_fit.cif")
        shutil.copyfile(
            fit_model_dir,
            fo, 
        )
        print("# Found fitting model {}".format(fit_model_dir))
        print("# Copy fitting model to {}".format(fo))



    #######################
    ### Clean temp file ###
    #######################
    if args.keep_temp_files:
        pass
    else:
        print("# Clean large temp files")
        try:
            os.remove(pjoin(temp_dir, "ncac", "aa_logits.npz"))
        except:
            pass

    ###################
    ### Final flags ###
    ###################
    if has_output:
        ending = "#####################\n" + \
                 "# Modeling finished #\n" + \
                 "#####################\n" + \
                 "# Check the output model at {}\n".format(output_dir)
    else:
        ending = "#########################" + \
                 "# Failed to build model #" + \
                 "#########################"

    print(ending)
    time1 = datetime.datetime.utcnow()
    print("# Job end   at {}".format(time1))


def add_args(parser):
    # Inputs
    basic_group = parser.add_argument_group("Basic options")
    basic_group.add_argument("--map", "-m", help="Input cryo-EM map in MRC2014 format", required=True)
    basic_group.add_argument("--seq", "-s", help="Input sequence(s) in FASTA format")
    basic_group.add_argument("--resolution", "-r", required=False, type=float, help="Resolution of input map")
    basic_group.add_argument("--output", "-o", help="Output directory", required=True)
    basic_group.add_argument("--chain", nargs='+', help="Input AF2 single-chain prediction, each file contains only one chain")
    basic_group.add_argument("--complex", help="Input AF3 complex prediction, this file should contain all sequences")


    # Other options
    other_group = parser.add_argument_group("Other options")
    other_group.add_argument("--fast", action='store_true', 
        help="If your map contains > 30000 residues, please use `fast` mode, else it will take `very` long time to finish"
    )
    other_group.add_argument(
        "--device", "--gpu", "-g",
        type=str, default="0", 
        help="Which GPU to use, specify \"i\" to use GPU i, default = 0"
    )
    other_group.add_argument("--nt", "-nt", type=int, default=4, 
        help="Num of threads to run on meanshift and docking, default = 4"
    )
    other_group.add_argument("--contour", "-c", type=float, default=1e-6, 
        #help="Map contour level, default = 1e-6", 
        help=argparse.SUPPRESS, 
    )

    # Keep temp files
    other_group.add_argument("--keep_temp_files", action='store_true', help="Keep temp files")

    # DL options
    # Not shown by default
    dl_group = parser.add_argument_group("Deep-learning options", 
        #help="Do not modify these options `UNLESS` you know what you are doing", 
        #help=argparse.SUPPRESS,
    )
    dl_group.add_argument("--dl_mc_aa_batchsize", type=int, default=40, 
        #help="MC and AA prediction batchsize",
        help=argparse.SUPPRESS,
    )
    dl_group.add_argument("--dl_mc_aa_stride", type=int, default=16, 
        #help="MC and AA prediction stride",
        help=argparse.SUPPRESS,
    )


    # Skipping controls
    skip_group = parser.add_argument_group("Skipping options")
    skip_group.add_argument("--skip_preprocess",   action='store_true', help=argparse.SUPPRESS)
    skip_group.add_argument("--skip_dl_mc_aa",     action='store_true', help=argparse.SUPPRESS)
    skip_group.add_argument("--skip_denovo",       action='store_true', help=argparse.SUPPRESS)
    skip_group.add_argument("--skip_fix",          action='store_true', help=argparse.SUPPRESS)
    skip_group.add_argument("--skip_imp",          action='store_true', help=argparse.SUPPRESS)
    skip_group.add_argument("--skip_dock",         action='store_true', help=argparse.SUPPRESS)
    skip_group.add_argument("--skip_assemble",     action='store_true', help=argparse.SUPPRESS)
    skip_group.add_argument("--skip_reorder",      action='store_true', help=argparse.SUPPRESS)
    skip_group.add_argument("--skip_pmodel",       action='store_true', help=argparse.SUPPRESS)
    skip_group.add_argument("--skip_final_refine", action='store_true', help=argparse.SUPPRESS)

    return parser


if __name__ == '__main__':
    detailed_help = """
\033[32m
Basic usage:
de novo modeling   : run.py --map input.mrc --seq seq.fasta -o output_dir
de novo + template : run.py --map input.mrc --seq seq.fasta -o output_dir --template chain0.pdb chain1.pdb ...

Fast mode:
de novo modeling   : run.py --map input.mrc --seq seq.fasta -o output_dir --fast
\033[0m
\033[31mIf you launch >1 modeling jobs at a time
The output_dir `MUST` be set differently\033[0m
"""

    class HelpFormatter(argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        description=detailed_help,
        formatter_class=HelpFormatter,
    )

    args = add_args(parser).parse_args()
    main(args)
