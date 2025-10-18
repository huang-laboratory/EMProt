'''
This script is adatped from
$PHENIX/modules/phenix/phenix/programs/map_model_cc.py
Use cctbx-base package out of Phenix program
'''
from __future__ import division, print_function
import os

import mmtbx.maps.map_model_cc
from iotbx.map_model_manager import map_model_manager
import mmtbx.maps.mtriage
from libtbx.utils import Sorry, null_out
import mmtbx.utils

from iotbx.data_manager.model import ModelDataManager
from iotbx.data_manager.real_map import RealMapDataManager

from collections import namedtuple
import numpy as np

Compute = namedtuple(
    "compute", 
    [
        "cc_box",
        "cc_image",
        "cc_mask",
        "cc_peaks",
        "cc_per_chain",
        "cc_per_residue",
        "cc_per_residue_group",
        "cc_volume",
        "fsc",
    ]
)
Params = namedtuple(
    "params", 
    [
        "atom_radius",
        "ignore_symmetry_conflicts", 
        "keep_map_calc",
        "resolution",
        "scattering_table",
        "wrapping",
        "compute",
    ]
)


def compute_cc(
    model_file,
    map_file,
    resolution,
    bfactor=None,
):
    # set up data
    model = ModelDataManager().get_model(filename=model_file)

    if bfactor is not None:
        for atom in model.get_hierarchy().atoms():
            atom.set_b(np.float64(bfactor))

    real_map = RealMapDataManager().get_real_map(filename=map_file)

    compute = Compute(
        cc_box=True,
        cc_image=False,
        cc_mask=True,
        cc_peaks=True,
        cc_per_chain=True,
        cc_per_residue=True,
        cc_per_residue_group=False,
        cc_volume=True,
        fsc=True,
    )

    params = Params(
        atom_radius=None,
        ignore_symmetry_conflicts=False,
        keep_map_calc=False,
        resolution=np.float64(resolution),
        scattering_table="electron",
        #wrapping=True,
        wrapping=False,
        compute=compute,
    )

    # set up base
    base = map_model_manager(
        map_manager=real_map,
        model=model,
        #wrapping=True,
        wrapping=False,
        ignore_symmetry_conflicts=False,
        log=null_out(),
    )
    base.box_all_maps_around_model_and_shift_origin()

    # compute cc
    task = mmtbx.maps.map_model_cc.map_model_cc(
        map_data=base.map_data(),
        pdb_hierarchy=base.model().get_hierarchy(),
        crystal_symmetry=base.model().crystal_symmetry(),
        params=params, 
    )
    task.validate()
    task.run()
    return task.get_results()

def add_args(parser):
    parser.add_argument("--pdb", "-p", required=True, help="Input structure")
    parser.add_argument("--map", "-m", required=True, help="Input map")
    parser.add_argument("--resolution", "-r", required=True, help="Map resolution")
    parser.add_argument("--bfactor", "-b", default=None)
    return parser

def main(args):
    # ignore bfactor in pdb file

    if args.bfactor is not None:
        print("Using specified bfactor {}".format(args.bfactor))

    print("Model from {}".format(args.pdb))
    print("Map from {}".format(args.map))

    result = compute_cc(
        args.pdb,
        args.map,
        args.resolution,
        args.bfactor, 
    )
    print("CC_mask   : {:.4f}".format(result.cc_mask))
    print("CC_volume : {:.4f}".format(result.cc_volume))
    print("CC_peaks  : {:.4f}".format(result.cc_peaks))
    print("CC_box    : {:.4f}".format(result.cc_box))
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args = add_args(parser).parse_args()
    main(args)
