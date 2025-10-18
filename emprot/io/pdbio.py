import os
import Bio
import numpy as np
from typing import List
from copy import deepcopy
from collections import defaultdict

from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.mmcifio import MMCIFIO, mmcif_order
from Bio.PDB import PDBParser, MMCIFParser

from emprot.utils.residue_constants import (
    restype_3_to_index,
    restype_1_to_index,
    index_to_restype_1,
    index_to_restype_3,
    restype_3to1,
    restype_1to3,

    restype3_to_atoms,
)



def read_structure_simple(filename, type='pdb',targets=None):
    if type == 'cif' or filename[-4:] in ['.cif']:
        return read_cif_simple(filename, targets)
    elif type == 'pdb' or filename[-4:] in ['.pdb', '.pdb1', '.pdb0']:
        return read_pdb_simple(filename, targets)
    else:
        raise "Error file must be .pdv or .cif"

def read_cif_simple(filename, targets=None):
    "data_xxx"
    "#"
    "loop_"
    "_atom_site.xxxx"
    "ATOM xxxx"

    mmcif_keys = ["_atom_site.group_PDB",
                  "_atom_site.id",
                  "_atom_site.type_symbol",
                  "_atom_site.label_atom_id",
                  "_atom_site.label_alt_id",
                  "_atom_site.label_comp_id",
                  "_atom_site.label_asym_id",
                  "_atom_site.label_entity_id",
                  "_atom_site.label_seq_id",
                  "_atom_site.pdbx_PDB_ins_code",
                  "_atom_site.Cartn_x",
                  "_atom_site.Cartn_y",
                  "_atom_site.Cartn_z",
                  "_atom_site.occupancy",
                  "_atom_site.B_iso_or_equiv",
                  "_atom_site.pdbx_formal_charge",
                  "_atom_site.auth_seq_id",
                  "_atom_site.auth_comp_id",
                  "_atom_site.auth_asym_id",
                  "_atom_site.auth_atom_id",
                  "_atom_site.pdbx_PDB_model_num",
    ]


    coords = []
    types = []

    with open(filename, 'r') as f:
        lines = f.readlines()

    loop = False
    keys = []

    atom_pos_key_inds = []
    res_type_key_ind = None
    atom_name_key_ind = None
    atom_type_key_ind = None
    for line in lines:
        # Get keys
        if line.startswith("loop_"):
            loop = True
            continue
        if line.startswith("#"):
            loop = False
            continue

        if loop and line.strip() in mmcif_keys:
            keys.append(line.strip())

        # Read records according to keys
        fields = line.strip().split()
        if line.startswith("ATOM") and len(fields) == len(keys):

            # Only init once
            if not atom_pos_key_inds:
                atom_pos_key_inds += [keys.index("_atom_site.Cartn_x"), keys.index("_atom_site.Cartn_y"), keys.index("_atom_site.Cartn_z")]
                res_type_key_ind = keys.index("_atom_site.label_comp_id")
                atom_name_key_ind = keys.index("_atom_site.label_atom_id")
                atom_type_key_ind = keys.index("_atom_site.type_symbol")

            # Always ignore 'H'
            atom_type = fields[atom_type_key_ind].strip()
            if atom_type == "H":
                continue
            
            atom_name = fields[atom_name_key_ind].strip()
            if targets is not None:
                if atom_name not in targets:
                    continue
            
            x_ind, y_ind, z_ind = atom_pos_key_inds
            x = float(fields[x_ind])
            y = float(fields[y_ind])
            z = float(fields[z_ind])
            coords.append([x, y, z])
            types.append(fields[res_type_key_ind].strip())

    return np.asarray(coords), types


def read_pdb_simple(filename, targets=None, res_type=True, bfactor=False):
    with open(filename, 'r') as f:
        lines = f.readlines()
    coords = []
    aatypes = []
    bfactors = []
    for line in lines:
        if line.startswith("ATOM"):
            atom_name = line[12:16].strip()
            if targets is not None:
                if atom_name not in targets:
                    continue
            coord = [float(x) for x in [line[i:i+8] for i in [30, 38, 46]]]
            coords.append(coord)
            aatype = line[17:20].strip()
            aatypes.append(aatype)

            if len(line) >= 66:
                _bfactor = float(line[60:66])
            else:
                _bfactor = 0.0
            bfactors.append(_bfactor)

    coords = np.asarray(coords, dtype=np.float32)
    bfactors = np.asarray(bfactors, dtype=np.float32)

    ret = (coords, )
    if res_type:
        ret += (aatypes, )

    if bfactor:
        ret += (bfactors, )

    return ret



def read_pdb_chains(filename, targets=None):
    with open(filename, 'r') as f:
        lines = f.readlines()
    coords = []
    aatypes = []
    chains_coords = []
    chains_aatypes = []
    chains = []

    n_chain = 0
    for line in lines:
        if line.startswith("ATOM"):
            atom_name = line[12:16].strip()
            if targets is not None:
                if atom_name not in targets:
                    continue
            coord = [float(x) for x in [line[i:i+8] for i in [30, 38, 46]]]
            coords.append(coord)
            aatype = line[17:20].strip()
            aatypes.append(aatype)
            chains.append(n_chain)

        # At TER
        if line.startswith("TER") or line.startswith("END"):
            n_chain += 1
    coords = np.asarray(coords, dtype=np.float32)
    chains = np.asarray(chains, dtype=np.int32)
    return coords, aatypes, chains

            #if len(coords) > 0:
            #    chains_coords.append(np.asarray(coords))
            #    chains_aatypes.append(aatypes)
            #    coords = []
            #    aatypes = []
    # Extras
    #if len(coords) > 0:
    #    chains_coords.append(coords)
    #    chains_aatypes.append(aatypes)
    #
    #return chains_coords, chains_aatypes



def read_residue_ncac_pdb_simple(filename, targets=None):
    with open(filename, 'r') as f:
        lines = f.readlines()
    n  = []
    ca = []
    c  = []
    for line in lines:
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        if atom_name not in ['N', 'CA', 'C']:
            continue
        coord = [float(x) for x in [line[x:x+8] for x in [30, 38, 46]]]

        if atom_name == "N":
            n.append(coord)
        if atom_name == "CA":
            ca.append(coord)
        if atom_name == "C":
            c.append(coord)

    n  = np.asarray(n, dtype=np.float32)
    ca = np.asarray(ca, dtype=np.float32)
    c  = np.asarray(c, dtype=np.float32)

    if not n.shape == ca.shape == c.shape:
        raise Exception("Invalid residues")

    coords = np.stack([n, ca, c], axis=1) # (N, 3, 3)
    return np.asarray(coords, dtype=np.float32)



# Read pdb for protein
def read_pdb(filename, ignore_hetatm=True, keep_valid=True, return_bfactor=False, return_occupancy=False):
    # Read file
    if filename.endswith(".pdb"):
        parser = PDBParser(QUIET=True)
    elif filename.endswith(".cif"):
        parser = MMCIFParser(QUIET=True)
    else:
        print(f"Error only support pdb/cif file")
    structure = parser.get_structure('pdb', filename)

    # Only use model 0
    model = structure[0]

    # Extract residue information
    atom_pos = []
    atom_mask = []
    res_type = []
    res_idx = []
    chain_idx = []
    atom_bfactor = []
    atom_occupancy = []

    # TODO Include d/rna

    for i, chain in enumerate(model):
        prev_residue_number = None
        for residue_index, residue in enumerate(chain):
            hetflag, residue_number, icode = residue.get_id()
            # 0325 Skip inserted residues
            if icode != " ":
                continue
            if prev_residue_number is None or residue_number != prev_residue_number:
                # ALA GLY ...
                resname3 = residue.get_resname().strip()
                # If resname is not standard residues
                try:
                    resname1 = restype_3to1[resname3]
                except:
                    resname1 = "X"
                    resname3 = "XXX"

                hetfield, resseq, icode = residue.get_id()

                # If hetatom
                if ignore_hetatm and hetfield != " ":
                    continue

                if resname1 != "X":
                    res_type.append(index_to_restype_1.index(resname1))
                else:
                    res_type.append(len(index_to_restype_1))

                res_idx.append(residue_number)
                prev_residue_number = residue_number

            coords = []
            mask = []
            bfactor = []
            # Get atom types for current residue
            try:
                atom_names = restype3_to_atoms[resname3] # (14, )
            except:
                atom_names = ["N", "CA", "C", "O"] + [""]*10

            for atom_index in atom_names:
                try:
                    atom = residue[atom_index]
                    coords.append(atom.get_coord())
                    #
                    bfactor.append(atom.get_bfactor())
                    mask.append(1)
                except KeyError:
                    coords.append([float("nan") for _ in range(3)])
                    #
                    bfactor.append(-1)
                    mask.append(0)
            atom_pos.append(coords)
            atom_mask.append(mask)
            chain_idx.append(i)
            #
            atom_bfactor.append(bfactor)

    # Convert to NumPy arrays
    #print(len(atom_pos))
    #print(atom_pos[-1])
    atom_pos = np.array(atom_pos).astype(np.float32) # (N, 14, 3)
    atom_mask = np.array(atom_mask).astype(bool) # (N, 14)
    res_type = np.array(res_type).astype(np.int32) # (N, )
    res_idx = np.array(res_idx).astype(np.int32) # (N, )
    chain_idx = np.array(chain_idx).astype(np.int32) # (N, )
    atom_bfactor = np.array(atom_bfactor).astype(np.float32) # (N, 14)

    # Keep valid amino-acids
    if keep_valid:
        idxs = np.all(atom_mask[:, :3], axis=-1)
        if not len(idxs) > 0:
            raise "# Error cannot find any amino-acid in the pdb"
        atom_pos = atom_pos[idxs]
        atom_mask = atom_mask[idxs]
        res_type = res_type[idxs]
        res_idx = res_idx[idxs]
        chain_idx = chain_idx[idxs]
        #
        atom_bfactor = atom_bfactor[idxs]

    if return_bfactor:
        return atom_pos, atom_mask, res_type, res_idx, chain_idx, atom_bfactor
    else:
        return atom_pos, atom_mask, res_type, res_idx, chain_idx


def split_atoms_to_chains(atom_pos, chain_idx):
    n = np.max(chain_idx) + 1
    chains = []
    for i in range(n):
        idx = chain_idx == i
        chains.append(atom_pos[idx])
    return chains

###############################################################################
################################ Write utils ##################################
###############################################################################

def write_points_as_pdb(filename, res_pos, res_types, bfactors=None, ter=False, chain='A', atom=" CA "):
    assert len(res_types) >= len(res_pos)
    f = open(filename, 'w')
    if bfactors is None:
        bfactors = np.ones(len(res_pos))
    for i in range(len(res_pos)):
        n=i+1
        f.write("ATOM  {:5d} {:4s} {:>3s}{:>2s}{:4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}\n".format(n, atom, res_types[i], chain, n, res_pos[i][0], res_pos[i][1], res_pos[i][2], bfactors[i], bfactors[i]))
        if ter:
            f.write("TER\n")
    if not ter:
        f.write("TER\n")



def chains_to_pdb(
    filename, 
    chains: List[np.ndarray],
    res_name="ALA",
    atom_name=" CA ",
):
    f = open(filename, 'w')
    L = len(chains)
    n = 0
    for i in range(L):
        chain = chains[i]
        chain_id = str(i) if i < 100 else 99
        for k in range(len(chain)):
            f.write("ATOM  {:5d} {:4s} {:>3s}{:>2s}{:4d}    {:8.3f}{:8.3f}{:8.3f}\n".format(n, atom_name, res_name, chain_id, n, chain[k][0], chain[k][1], chain[k][2]))
            n += 1
        f.write("TER\n")
    f.close()


"""
chain_names = []
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
# One-letter chain id
for i in range(len(letters)):
    chain_names.append(letters[i])
# Two-letter chain id
for i in range(len(letters)):
    for k in range(len(letters)):
        chain_names.append(letters[i]+letters[k])
# In total we have 62 + 62 * 62 ~= 4000 chain ids
# Three or more-letter chain ids
for l in range(len(letters)):
    for m in range(len(letters)):
        for n in range(len(letters)):
            chain_names.append(letters[l]+letters[m]+letters[n])
# 62 * 62 * 62 = 238328 chains
"""


with open( os.path.join(os.path.dirname(__file__), "chain_names.txt" ), "r") as f:
    chain_names = f.readline()
    chain_names = chain_names.strip().split("|")


class CIFXIO(MMCIFIO):
    def _save_dict(self, out_file):
        label_seq_id = deepcopy(self.dic["_atom_site.auth_seq_id"])
        auth_seq_id = deepcopy(self.dic["_atom_site.auth_seq_id"])
        self.dic["_atom_site.label_seq_id"] = label_seq_id
        self.dic["_atom_site.auth_seq_id"] = auth_seq_id

        # Adding missing "pdbx_formal_charge", "auth_comp_id", "auth_atom_id" to complete a record
        N = len(self.dic["_atom_site.group_PDB"])
        self.dic["_atom_site.pdbx_formal_charge"] = ["?"]*N
        self.dic["_atom_site.auth_comp_id"] = deepcopy(self.dic["_atom_site.label_comp_id"])
        self.dic["_atom_site.auth_asym_id"] = deepcopy(self.dic["_atom_site.label_asym_id"])
        self.dic["_atom_site.auth_atom_id"] = deepcopy(self.dic["_atom_site.label_atom_id"])

        # Handle an extra space at the end of _atom_site.xxx
        _atom_site = mmcif_order["_atom_site"]
        _atom_site = [x.strip() + " " for x in _atom_site]
        mmcif_order["_atom_site"] = _atom_site

        new_dic = defaultdict()
        for k, v in self.dic.items():
            if k[:11] == "_atom_site.":
                new_k = k.strip() + " "
            else:
                new_k = k
            new_dic[new_k] = v
        self.dic = new_dic

        return super()._save_dict(out_file)




# Write a chain
def chain_atom_pos_to_pdb():
    pass











# Write multiple chains to a pdb file
def chains_atom_pos_to_pdb(
    filename : str,
    chains_atom_pos : List[np.ndarray],
    chains_atom_mask : List[np.ndarray],
    chains_res_types=None,
    chains_res_idxs=None,
    chains_chain_idxs=None,
    chains_occupancy=None,
    chains_bfactors=None,
    suffix='cif',
    remarks=None,
):
    # For different chains
    assert len(chains_atom_pos) == len(chains_atom_mask)
    if chains_occupancy is None:
        chains_occupancy = []
        for k in range(len(chains_atom_pos)):
            chains_occupancy.append( np.full(len(chains_atom_pos[k]), 1.0, dtype=np.float32) )

    if chains_bfactors is None:
        chains_bfactors = []
        for k in range(len(chains_atom_pos)):
            chains_bfactors.append( np.ones_like(chains_atom_mask[k], dtype=np.float32) * 100.0 )
   
    if chains_res_types is None:
        chains_res_types = []
        for k in range(len(chains_atom_pos)):
            chains_res_types.append( np.full(len(chains_atom_pos[k]), 0, dtype=np.int32) )
 
    if chains_res_idxs is None:
        chains_res_idxs = []
        for k in range(len(chains_atom_pos)):
            chains_res_idxs.append( np.arange(len(chains_atom_pos[k]), dtype=np.int32) )

    if chains_chain_idxs is None:
        chains_chain_idxs = []
        for k in range(len(chains_atom_pos)):
            chains_chain_idxs.append( k )

    struct = StructureBuilder()
    struct.init_structure("1")
    struct.init_seg("1")
    struct.init_model("1")

    std_res_types = list(restype_1_to_index.keys())[:20]
    n_atom = 0
    for k in range(len(chains_atom_pos)):
        # For each chain
        atom_pos = chains_atom_pos[k]
        atom_mask = chains_atom_mask[k]

        res_types = chains_res_types[k]
        res_idxs = chains_res_idxs[k]
        chain_idxs = chains_chain_idxs[k]
        bfactors = chains_bfactors[k]

        # Init a new chain
        chain_idx = chains_chain_idxs[k]
        struct.init_chain(chain_names[chain_idx])

        # For each residue
        #bfactor = bfactors[k]
        for i in range(len(atom_pos)):
            # For each atom
            #res_name_1 = res_types[i]
            res_type = res_types[i]
            res_name_1 = index_to_restype_1[res_type]

            res_name_1 = "A" if res_name_1 not in std_res_types else res_name_1

            res_name_3 = restype_1to3[res_name_1]

            atom_names = restype3_to_atoms[res_name_3] # (n, ) n <= 14
            #atom_names = ["CA"]
            n_atom = len(atom_pos[i])
            if len(atom_names) > n_atom:
                atom_names = atom_names[:n_atom]

            field_name = " "

            # Init a new residue
            #print(k, i, res_name_3, field_name, res_idxs[i])
            struct.init_residue(resname=res_name_3, field=field_name, resseq=res_idxs[i], icode=" ")

            #print(
            #    len(atom_names), 
            #    len(atom_pos), 
            #    len(bfactors), 
            #    len(atom_mask),
            #)

            #$for atom_name, pos, mask in zip(
            for atom_name, pos, bfactor, mask in zip(
                atom_names, atom_pos[i], bfactors[i], atom_mask[i]
                #atom_names, atom_pos[i], bfactors, atom_mask[i]
                #atom_names, atom_pos[i], atom_mask[i]
            ):
                if atom_name is None or \
                   atom_name == "" or \
                   mask < 1 or \
                   np.any(np.isnan( pos )):
                    continue

                struct.set_line_counter(n_atom+1)
                struct.init_atom(
                    name=atom_name,
                    coord=pos,
                    b_factor=bfactor,
                    occupancy=1.0,
                    altloc=" ",
                    fullname=atom_name,
                    element=atom_name[0],
                )
                n_atom += 1

    struct = struct.get_structure()
    if suffix in ['cif', '.cif', 'CIF', '.CIF', 'mmcif', 'MMCIF']:
        io = CIFXIO()
        io.set_structure(struct)
        io.save(filename)
    else:
        io = PDBIO()
        io.set_structure(struct)
        io.save(filename, write_end=False)


def chains_atom_pos_to_cif(
    filename : str,
    chains_atom_pos : List[np.ndarray],
    chains_atom_mask : List[np.ndarray],
    chains_res_types : List[str],
    chains_occupancy=None,
    chains_bfactors=None,
):
    return chains_atom_pos_to_pdb(
        filename,
        chains_atom_pos,
        chains_atom_mask,
        chains_res_types,
        chains_occupancy,
        chains_bfactors,
    )


def atom3_to_atom14(atom3_pos):
    pass

def ca_to_atom3(ca):
    l = ca.shape[0]
    atom3_pos = np.zeros((l, 3, 3), dtype=np.float32)
    atom3_mask = np.zeros((l, 3), dtype=np.int32)
    for i in range(l):
        atom3_pos[i][1] = ca[i]
        atom3_mask[i][1] = 1
    return atom3_pos, atom3_mask

def write_atoms_as_pdb(filename, atom_pos, res_type=None, bfactor=None, ter=True, final_ter=False):
    restypes_1 = [
        "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
        "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
    ]
    restypes_3 = [
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    ]

    # (L, 3)
    if atom_pos.ndim == 2 and atom_pos.shape[-1] == 3:
        atom_names = [" CA "]
        atom_pos = atom_pos[:, None, :]
        # (L, 1, 3)
    elif atom_pos.ndim == 3 and atom_pos.shape[-1] == 3 and atom_pos.shape[1] == 1:
        atom_names = [" CA "]
        # (L, 1, 3)
    elif atom_pos.ndim == 3 and atom_pos.shape[-1] == 3 and atom_pos.shape[1] >= 3:
        atom_names = [" N  ", " CA ", " C  "]
        atom_pos = atom_pos[:, :3, :]
        # (L, 3, 3)
    else:
        raise f"# Error shape - {atom_pos.shape}"

    if np.any(np.isnan(atom_pos)):
        raise f"# Error atom pos have NaNs"

    if res_type is None:
        res_type = np.zeros(len(atom_pos), dtype=np.int32)
    assert len(atom_pos) <= len(res_type)

    if bfactor is None:
        bfactor = np.ones(len(atom_pos), dtype=np.float32) * 100.0
    assert len(atom_pos) <= len(bfactor)

    n_res = 1
    n_atom = 1
    f = open(filename, 'w')
    for i in range(len(atom_pos)):
        residue = atom_pos[i]
        residue_type = restypes_3[res_type[i]]
        #ATOM      2  CA  MET A 214     203.072 155.727 155.024  1.00129.29           C
        for k in range(len(residue)):
            atom_name = atom_names[k]
            f.write(
                "ATOM  {:>5d} {:4s} {:3s}{:>2s}{:>4d}    {:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}\n".format(
                    n_atom if n_atom <= 99999 else 99999,
                    atom_name,
                    residue_type,
                    "A",
                    n_res if n_res <= 9999 else 9999,
                    residue[k][0],
                    residue[k][1],
                    residue[k][2],
                    1.0,
                    bfactor[i],
                )
            )
            n_atom += 1
        n_res += 1
        if ter:
            f.write("TER\n")
    if final_ter:
        f.write("TER\n")
    f.close()


# Convert to chains
def convert_to_chains(chain_idxs, *inputs):
    rets = ()
    n_chain = np.max(chain_idxs) + 1
    #rets = (list(range(n_chain)), )
    for x in inputs:
        l = list()
        for i in range(n_chain):
            idxs = chain_idxs == i
            l.append(x[idxs])
        rets = rets + (l, )
    return rets
