# a simple python wrapper to run model docking and refinement
import os
import sys
import tempfile
import subprocess
import numpy as np
from collections import OrderedDict
from typing import List

from emprot.io.fileio import getlines, writelines
from emprot.io.pdbio import read_pdb, chains_atom_pos_to_pdb

from emprot.utils.misc_utils import abspath, pjoin
from emprot.utils.domain import domains_to_sword_result
from emprot.utils.geo import apply

def check_exists(path):
    return os.path.exists(path)

def renum_pdb_forward(fpdb):
    raise NotImplementedError

def renum_pdb_backward(fpdb):
    raise NotImplementedError

def prepare_input(
        chains_pdb_dir, 
        chains_domains,
        out_dir,
        lib_dir="./",
        temp_dir=None,
        log_dir=None,
        verbose=True,
        qs=3,
        ms=150,
        q=1,
        m=30,
        k=3.0,
        **kwargs,
    ):
    assert len(chains_pdb_dir) == len(chains_domains)

    # create a temp directory to dump SWORD result
    exists = []
    all_lines = []
    all_pipe_out = []
    with tempfile.TemporaryDirectory() as __temp_dir:
        if temp_dir is None:
            temp_dir = __temp_dir
        if log_dir is None:
            log_dir = pjoin(temp_dir, "prepare.log")
        print("# Log written to {}".format(log_dir))

        # parse domain
        for kk in range(len(chains_pdb_dir)):
            # convert to sword result
            sword_result = domains_to_sword_result(chains_domains[kk])
            # dump to file
            sword_temp_dir = pjoin(temp_dir, f"xxx_chain_{kk}_sword.out")

            writelines(sword_temp_dir, [sword_result])
            # assign domain
            # domasg input.pdb swd.out output.pdb
            out_temp_dir = pjoin(temp_dir, f"xxx_chain_{kk}.pdb")

            cmd = lib_dir + "/bin/domasgx" + " {} {} {} -qs {} -ms {} -q {} -m {} -k {}".format(chains_pdb_dir[kk], sword_temp_dir, out_temp_dir, qs, ms, q, m, k)
            if verbose:
                print(f"# Running command {cmd}")

            # run and record pipe out
            result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            all_pipe_out.extend( result.stdout.decode('utf-8').split('\n') )

            if result.returncode != 0:
                print("WARNING this command has run failed")
                print("WARNING unexpected result may be observed")

            # check exists
            exist = check_exists(out_temp_dir)
            exists.append(exist)

            lines = getlines(out_temp_dir)
            for line in lines:
                if line.startswith("ATOM") or line.startswith("REMARK"):
                    all_lines.append(line)

            all_lines.append("TER")

    # cat all chains
    writelines(out_dir, all_lines)
    if verbose:
        print("Combine all marked chains into {}".format(out_dir))

    # log anyway
    writelines(log_dir, all_pipe_out)
            
    all_exists = check_exists(out_dir)
    return len(exists) > 0 and all(exists) and all_exists



# Do not use for modified docking program
def postprocess_output(
        pdb_dir, 
        out_dir, 
        lib_dir="./", 
        temp_dir=None,
        log_dir=None, 
        verbose=True,
        **kwargs,
    ):
    with tempfile.TemporaryDirectory() as __temp_dir:
        if temp_dir is None:
            temp_dir = __temp_dir
        if log_dir is None:
            log_dir = pjoin(temp_dir, "postprocess.log")
        print("# Log written to {}".format(log_dir))

        # renumpdb output.pdb outputx.pdb
        bin_dir = lib_dir + "/bin/renumpdbx"
        cmd = bin_dir + " {} {}".format(pdb_dir, out_dir)

        # run
        if verbose:
            print(f"# Running command {cmd}")
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode != 0:
            print("WARNING this command has run failed")
            print("WARNING unexpected result may be observed")
            writelines(log_dir, result.stderr.decode('utf-8').split('\n') + ["success"])
        else:
            writelines(log_dir, result.stdout.decode('utf-8').split('\n') + ["success"])

    return check_exists(out_dir)



def run_dock_and_refine(
        map_dir,
        pdb_dir,
        out_dir,
        lib_dir="./",
        temp_dir=None,
        log_dir=None,
        verbose=True,
        resolution=5.0,
        thresh=15.0,
        nt=4,
        angle_step=18.0,
        fgrid=5.0,
        sgrid=2.0,
        mode='normal',

        flex_refine=False,
        init_trans_dir=None,
        **kwargs,
    ):
    # args [MC.mrc AB.pdb 5.0 out.pdb -nt 8 -thresh 15 -fgrid 5 -sgrid 2]
    if mode == 'fast':
        # compiled on centos 6 ifort 15.0.1 with -O3
        if not flex_refine:
            bin_dir = lib_dir + "/bin/dockx"
        else:
            bin_dir = lib_dir + "/bin/flexrefinex"
    else:
        # compiled on centos 6 ifort 15.0.1 with -O2
        raise NotImplementedError
        if not flex_refine:
            bin_dir = lib_dir + "/bin/dock"
        else:
            bin_dir = lib_dir + "/bin/flexrefine"


    assigned_chain_idx = set()
    with tempfile.TemporaryDirectory() as __temp_dir:
        if temp_dir is None:
            temp_dir = __temp_dir
        if log_dir is None:
            log_dir = pjoin(temp_dir, "run.log")
        print("# Log written to {}".format(log_dir))

        transform_dir = pjoin(temp_dir, "transform.pdb")

        # command
        if flex_refine and init_trans_dir is not None:
            cmd = bin_dir + " {} {} {} {} -nt {} -thresh {} -angle_step {} -fgrid {} -sgrid {} -trans {}".format(map_dir, pdb_dir, resolution, transform_dir, nt, thresh, angle_step, fgrid, sgrid, init_trans_dir)
        else:
            cmd = bin_dir + " {} {} {} {} -nt {} -thresh {} -angle_step {} -fgrid {} -sgrid {}".format(map_dir, pdb_dir, resolution, transform_dir, nt, thresh, angle_step, fgrid, sgrid)


        if verbose:
            print(f"# Running command {cmd}")
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode != 0:
            print("WARNING this command has run failed")
            print("WARNING unexpected result may be observed")
            writelines(log_dir, result.stderr.decode('utf-8').split('\n'))
        else:
            writelines(log_dir, result.stdout.decode('utf-8').split('\n'))


        # postprocess of atoms
        seg_Ts = OrderedDict()
        lines = getlines(transform_dir)

        # get the transformation matrix
        for line in lines:
            if line.startswith("REMARK chain"):
                fields = line.strip().split()

                chain_idx = int(fields[2])
                seg_idx = int(fields[4])
                label = "{}_{}".format(chain_idx, seg_idx)

                center = fields[6:9]
                if center[0] == '-':
                    seg_Ts[label] = None
                    continue

                center = [float(x) for x in center]
                center = np.asarray(center, dtype=np.float32) # (3, )

                R = [float(x) for x in fields[10:19]] # (9, )
                t = [float(x) for x in fields[20:23]] # (3, )
                R = np.asarray(R, dtype=np.float32).reshape(3, 3)
                t = np.asarray(t, dtype=np.float32)

                seg_Ts[label] = (center, R, t)

                assigned_chain_idx.add(chain_idx - 1)

        # get atom records
        chain_idxs = []
        seg_idxs = []
        atom_lines = []
        chain_idx = 1
        lines = getlines(pdb_dir)
        for line in lines:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    seg_idx = int(line[70:74])
                except Exception as e:
                    seg_idx = 1
                label = "{}_{}".format(chain_idx, seg_idx)

                chain_idxs.append(chain_idx)
                seg_idxs.append(seg_idx)
                atom_lines.append(line)

            if line.startswith("TER"):
                chain_idx += 1

        atom_lines = np.asarray(atom_lines) # object array
        chain_idxs = np.asarray(chain_idxs, dtype=np.int32)
        seg_idxs   = np.asarray(seg_idxs, dtype=np.int32)

        # transform atoms
        chains_atom_lines = OrderedDict()
        for k, v in seg_Ts.items():
            chain_idx, seg_idx = [int(x) for x in k.split("_")]
            if v is None:
                print("Chain {} is not assigned".format(chain_idx))
                continue

            chain_mask = chain_idxs == chain_idx
            seg_mask = seg_idxs == seg_idx
            chain_seg_mask = np.logical_and(chain_mask, seg_mask)
            
            sel_atom_lines = atom_lines[chain_seg_mask]
            coords = []
            for line in sel_atom_lines:
                coords.append([float(x) for x in [line[i:i+8] for i in [30, 38, 46]]])
            coords = np.asarray(coords, dtype=np.float32)

            center, R, t = v
            transformed_coords = apply(coords - center, R, t) + center
            transformed_atom_lines = []
            for i, line in enumerate(sel_atom_lines):
                new_line = line[:30] + "{:8.3f}{:8.3f}{:8.3f}".format(
                    transformed_coords[i][0],
                    transformed_coords[i][1],
                    transformed_coords[i][2],
                ) + line[54:]
                transformed_atom_lines.append(new_line)

            chains_atom_lines.setdefault(chain_idx, [])
            chains_atom_lines[chain_idx].extend(transformed_atom_lines)
           
 
        # renum res idx by stable sorting
        chains_atom_pos = []
        chains_atom_mask = []
        chains_res_type = []
        chains_res_idx = []
        chains_bfactor = []
        for i, (k, v) in enumerate(chains_atom_lines.items()):
            v = np.asarray(v)
            res_idxs = np.asarray([int(line[22:26]) for line in v], dtype=np.int32)
            sorted_idxs = np.argsort(res_idxs, kind='stable')
            sorted_v = v[sorted_idxs]

            chain_temp_dir = pjoin(temp_dir, f"fitted_chain_{i}.pdb")

            # 2025-05-19
            # set all residues to be chain 'A'
            new_sorted_v = []
            for line in sorted_v.tolist():
                new_line = line[:21] + "A" + line[22:]
                new_sorted_v.append( new_line )

            writelines(chain_temp_dir, new_sorted_v + ["TER\n"])
            print("Write fitted chain {} to {}".format(i, chain_temp_dir))

            atom_pos, atom_mask, res_type, res_idx, chain_idx, bfactor = read_pdb(chain_temp_dir, keep_valid=False, return_bfactor=True)

            chains_atom_pos.append(atom_pos)
            chains_atom_mask.append(atom_mask)
            chains_res_type.append(res_type)
            chains_res_idx.append(res_idx)
            chains_bfactor.append(bfactor)            

        # todo convert to mmcif
        chains_atom_pos_to_pdb(
            filename=out_dir,
            chains_atom_pos=chains_atom_pos,
            chains_atom_mask=chains_atom_mask,
            chains_res_types=chains_res_type,
            chains_res_idxs=chains_res_idx,
            chains_bfactors=chains_bfactor,
            suffix='cif' if out_dir.endswith('.cif') else 'pdb',
        )
        print("Write all fitted chains to {}".format(out_dir))

    # check if we have run successfully
    return assigned_chain_idx




def run_dock_and_refine_pipeline(
        map_dir, 
        chains_pdb_dir, 
        chains_domains, 
        out_dir, 
        lib_dir, 
        log_dir=None,
        temp_dir=None, 
        flex_refine=False,
        **kwargs,
    ):
    
    with tempfile.TemporaryDirectory() as __temp_dir:
        # If specifies a temp dir
        if temp_dir is None:
           temp_dir = __temp_dir

        # if flex refine, modify params
        if not flex_refine:
            qs, ms, q, m, k = 3, 150, 1, 30, 3.0
        else:
            qs, ms, q, m, k = 0,   0, 0,  0, 3.0

        success0 = prepare_input(
            chains_pdb_dir=chains_pdb_dir, 
            chains_domains=chains_domains, 
            out_dir=pjoin(temp_dir, 'xxx_all.pdb'), 
            lib_dir=lib_dir,
            log_dir=pjoin(temp_dir, "preprocess.log"),
            temp_dir=temp_dir,
            qs=qs,
            ms=ms,
            q=q,
            m=m,
            k=k,
            **kwargs,
        )
        assigned_chain_idx = run_dock_and_refine(
            map_dir=map_dir, 
            pdb_dir=pjoin(temp_dir, 'xxx_all.pdb'), 
            out_dir=out_dir, 
            lib_dir=lib_dir,
            log_dir=pjoin(temp_dir, "run.log"),
            temp_dir=temp_dir,
            flex_refine=flex_refine,
            **kwargs,
        )
        if log_dir is None:
            log_dir = pjoin(temp_dir, "fitted_xxxx.log")

        all_pipe_out = []
        all_pipe_out.extend(getlines(pjoin(temp_dir, "preprocess.log")))
        all_pipe_out.extend(getlines(pjoin(temp_dir, "run.log")))
        writelines(log_dir, all_pipe_out)

    return assigned_chain_idx











# only run flex refine
def run_flex_refine_pipeline(
        map_dir, 
        chains_pdb_dir, 
        chains_domains, 
        chains_init_trans_dir,
        out_dir, 
        lib_dir, 
        log_dir=None,
        temp_dir=None, 
        **kwargs,
    ):
    
    with tempfile.TemporaryDirectory() as __temp_dir:
        # If specifies a temp dir
        if temp_dir is None:
           temp_dir = __temp_dir

        # if flex refine, modify params
        qs, ms, q, m, k = 0,   0, 0,  0, 3.0

        success0 = prepare_input(
            chains_pdb_dir=chains_pdb_dir, 
            chains_domains=chains_domains, 
            out_dir=pjoin(temp_dir, 'xxx_all.pdb'), 
            lib_dir=lib_dir,
            log_dir=pjoin(temp_dir, "preprocess.log"),
            temp_dir=temp_dir,
            qs=qs,
            ms=ms,
            q=q,
            m=m,
            k=k,
            **kwargs,
        )

        # write init trans
        lines = []
        for fp in chains_init_trans_dir:
            lines.extend(getlines(fp))
        init_trans_dir = pjoin(temp_dir, "xxx_init_trans.log")
        writelines(init_trans_dir, lines)
        

        assigned_chain_idx = run_dock_and_refine(
            map_dir=map_dir, 
            pdb_dir=pjoin(temp_dir, 'xxx_all.pdb'), 
            out_dir=out_dir, 
            lib_dir=lib_dir,
            log_dir=pjoin(temp_dir, "run.log"),
            temp_dir=temp_dir,
            flex_refine=True,
            init_trans_dir=init_trans_dir,
            **kwargs,
        )
        if log_dir is None:
            log_dir = pjoin(temp_dir, "fitted_xxxx.log")

        all_pipe_out = []
        all_pipe_out.extend(getlines(pjoin(temp_dir, "preprocess.log")))
        all_pipe_out.extend(getlines(pjoin(temp_dir, "run.log")))
        writelines(log_dir, all_pipe_out)

    return assigned_chain_idx



def run_dock_score(
        map_dir, pdb_dir, out_dir, lib_dir='./',
        resolution=5.0,
        thresh=15.0,
        nt=4,
        angle_step=18.0,
        fgrid=5.0,
        sgrid=2.0,
        verbose=True,
        **kwargs
    ):
    raise NotImplementedError

