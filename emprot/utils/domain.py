import os
import re
import tempfile
import subprocess
import numpy as np
from collections import OrderedDict

from emprot.utils.misc_utils import pjoin
from emprot.io.fileio import writelines

######################
# Some utils for SWORD
######################

def run_sword(pdb_dir, lib_dir='./'):
    raise NotImplementedError

def parse_sword_result(lines):
    raise NotImplementedError

def domains_to_sword_result(domains=None):
    # sword result looks like 
    """
    PDB: 1JX4A
    ASSIGNMENT   
    #D|Min|                                                  BOUNDARIES|   AVERAGE κ|   QUALITY|
    3 |73 |                                       1-163 164-236 237-335|    3.656125|       ***|
    ALTERNATIVES
    #D|Min|                                                  BOUNDARIES|   AVERAGE κ|   QUALITY|
    5 |28 |                         1-74 75-135 136-163 164-236 237-335|    3.438997|         *|
    5 |28 |                 1-74 75-135;202-236 136-163 164-201 237-335|    3.414087|         *|
    """
    if domains is None:
        line = ""
    else:
        n_domain = len(domains)
        min_res_num = 100000
        average_k = 3.8
        quality = '*' * 5
    
        boundaries = ""
        for k, domain in enumerate(domains):
            boundary = ""
            dom_res_num = 0
            for sub_domain in domain:
                start, end = sub_domain

                dom_res_num += (end - start + 1)
                boundary += "{}-{};".format(start, end)
            # get min dom res num
            min_res_num = min(min_res_num, dom_res_num)

            # delete last ';'
            boundary = boundary[:-1]
            boundaries += boundary + " "
        # delete last ' '
        boundaries = boundaries[:-1]
        
        line = "{:<1d} |{:<2d} | {:>59s}|{:>12.6f}|{:>10s}|".format(n_domain, min_res_num, boundaries, average_k, quality)
 
    lines = "PDB: XXXX\n" + \
            "ASSIGNMENT\n" + \
            "#D|Min|                                                  BOUNDARIES|   AVERAGE κ|   QUALITY|\n" + \
            line + "\n" + \
            "ALTERNATIVES\n" + \
            "#D|Min|                                                  BOUNDARIES|   AVERAGE κ|   QUALITY|\n" + \
            line

    return lines


def detect_large_loops(secstr, n=30, tokens=['0', '1', '2']):
    pattern = r'{}'.format(tokens[2]) + '{' + str(n) + ',}'
    matches = list(re.finditer(pattern, secstr))
    return [
        (match.start(), match.group(0)) for match in matches
    ]

#######################
# Some utils for stride
#######################
def run_stride(pdb_dir, lib_dir='./', temp_dir=None, verbose=True):
    stride_bin_dir = pjoin(lib_dir, "/bin/stride")
    with tempfile.TemporaryDirectory() as __temp_dir:
        if temp_dir is None:
            temp_dir = __temp_dir
        # parse ss from structure
        cmd = stride_bin_dir + " {} ".format(pdb_dir)
        if verbose:
            print(f"# Running command {cmd}")
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

        if result.returncode == 0:
            lines = result.stdout.decode('utf-8').split("\n")
        else:
            lines = []
    return lines


def extract_secstr_1d(lines):
    # ASG  ALA A  589    1    C          Coil    360.00    155.00     112.9      ~~~~
    # ASG  PRO A  595    7    E        Strand    -53.73    136.41       4.1      ~~~~
    # ASG  SER A  605   17    T          Turn    -63.03    -13.27      60.1      ~~~~
    # ASG  SER A  636   48    G      310Helix    -58.74    -32.06      88.5      ~~~~
    # ASG  ASP A  686   98    H    AlphaHelix    -50.58    -32.93     142.9      ~~~~
    sec = []
    for line in lines:
        if line.startswith("ASG"):
            fields = line.strip().split()
            secstr1 = fields[5]
            secstr3 = fields[6]
            if secstr1 in ['H', 'G', 'I']:
                sec.append("0")
            elif secstr1 in ['C', 'T']:
                sec.append("2")
            else:
                sec.append("1")
    return "".join(sec)


#######################
# Some utils for UniDoc
#######################

def run_unidoc(pdb_dir, chain='A', lib_dir="./", temp_dir=None, verbose=True, domain_type='merged'):
    unidoc_bin_dir = pjoin(lib_dir, "bin/unidoc_frag")
    stride_bin_dir = pjoin(lib_dir, "bin/stride")
    with tempfile.TemporaryDirectory() as __temp_dir:
        if temp_dir is None:
            temp_dir = __temp_dir
        # parse ss from structure
        cmd = stride_bin_dir + " {} ".format(pdb_dir)
        fstride = pjoin(temp_dir, "stride.out")
        if verbose:
            print("# Running command {}".format(cmd))
            print("# Intend to write stride output to {}".format(fstride))
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        writelines(
            fstride,
            [result.stdout.decode('utf-8'), result.stderr.decode('utf-8')],
        )

        if result.returncode != 0:
            print("# WARNING cannot parse domain for {}".format(pdb_dir))
            lines = [None, None]
        else:
            # run unidoc with pdb and sword.out
            cmd = unidoc_bin_dir + " {} {} {}".format(pdb_dir, chain, fstride)
            if verbose:
                print(f"# Running command {cmd}")
            result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

            if result.returncode == 0:
                lines = result.stdout.decode('utf-8').split('\n')[:2]
                lines = [line.strip() for line in lines]
                if len(lines) != 2:
                    lines = [None, None]
            else:
                lines = [None, None]

    #print(lines)
    unmerged = lines[0]
    merged = lines[1]
    if domain_type == 'merged':
        return merged
    else:
        return unmerged


def parse_unidoc_result(line):
    # unidoc result looks like
    # 846~986/619~845,1329~1382/1106~1328/31~218/1~30,219~281/987~1105/282~335/336~493/494~618
    # 1~189/190~252/253~306/307~464/465~589/590~816/817~915
    # (xxx-xxx,xxx-xxxx,xxx-xxxx/) * n - '/'
    # ...

    line = line.strip()
    res_idxs = []

    result = []
    domains = line.split('/')
    for k, domain in enumerate(domains):
        # each domain has pattern xxx~xxx,xxx-xxx,...,xxx-xxx
        sub_domains = domain.split(',')
        __domain = []
        for sub_domain in sub_domains:
            # each sub_domain has pattern xxx~xxx
            start, end = [int(x) for x in sub_domain.split('~')]
            __domain.append([start, end])
        result.append(__domain)

    return result

def domains_to_unidoc_result(domains):
    line = ""
    for k, domain in enumerate(domains):
        domain_line = ""
        for sub_domain in domain:
            start, end = sub_domain
            domain_line += str(start) + "~" + str(end) # xxx~xxx
            domain_line += "," # xxx-xxx,
        # delete last ','
        domain_line = domain_line[:-1]
        line += domain_line # xxx~xxx,xxx-xxx
        line += "/" # xxx~xxx,xxx-xxx/
    # delete last '/'
    line = line[:-1]
    return line


###########################
# Some utils for conversion
###########################

def convert_domains_to_1d_repr(domains):
    # warning
    # assumes all res idx starts from 0
    max_res_idx = -1
    for domain in domains:
        for sub_domain in domain:
            start, end = sub_domain
            max_res_idx = max(max_res_idx, end)


    d = np.full( (max_res_idx + 1, ), -1, dtype=np.int32)
    for k, domain in enumerate(domains):
        for sub_domain in domain:
            start, end = sub_domain
            for idx in range(start, end + 1):
                d[idx] = k
    return d


def convert_1d_repr_to_domains(d):
    domains = []
    d_max = d.max() + 1
    for i in range(d_max):
        (idxs, ) = np.where(d == i)
        domains.append([[int(idxs[0]), int(idxs[-1])]])
    return domains



###########################
# Some utils for annotation
###########################

def annotate_pdb_with_domains(lines, domains):
    # simply replace the bfactor columns with domain idxs
    domain = convert_domains_to_1d_repr(domains)

    new_lines = []
    for line in lines:
        if line.startswith("ATOM"):
            # query the domain idx
            res_idx = int(line[22:26])
            domain_idx = domain[res_idx]

            l = line[:60] + "{:>6.2f}".format(domain_idx) + line[66:]
            new_lines.append(l)
        else:
            continue
    return new_lines

def annotate_pdb_cif_with_domains(lines, domains):
    raise NotImplementedError

def split_pdb_with_domains(line, domains):
    # split a pdb file
    domain = convert_domains_to_1d_repr(domains)

    new_lines_domains = [None] * len(domains)
    for line in lines:
        if line.startswith("ATOM"):
            # query the domain idx
            res_idx = int(line[22:26])
            domain_idx = domain[res_idx]

            l = line[:60] + "{:>6.2f}".format(domain_idx) + line[66:]
            new_lines_domains[domain_idx].append(l)
        else:
            continue
    return new_lines_domains


##########################
# utils for domain merging
##########################

def merge_intervals(dom):
    new_dom = []
    dom.sort(key=lambda x: x[0])
    for current in dom:
        if not new_dom:
            new_dom.append(current)
        else:
            last = new_dom[-1]
            if last[1] + 1 >= current[0]:
                new_dom[-1] = [last[0], max(last[1], current[1])]
            else:
                new_dom.append(current)
    return new_dom

def merge_domains_simple(domains, n_min_res=50):
    # specify a minimum n_min_res such that domains have fewer residues
    doms = domains.copy()
    n_iter = 0
    max_iter = 20
    while n_iter < max_iter:
        need_merge = False
        for i in range(len(doms)):
            l = 0
            ters = []
            for d in doms[i]:
                l += abs(d[1] - d[0])
                ters.append(d[0])
                ters.append(d[1])
            if l < n_min_res:
                need_merge = True
                neighbors = []
                for k in range(len(doms)):
                    if k == i:
                        continue
                    k_is_neighbor = False
                    for d in doms[k]:
                        if d[0] - 1 in ters or \
                            d[1] + 1 in ters:
                            k_is_neighbor = True
    
                    if k_is_neighbor:
                        neighbors.append(k)
                        break
                neighbors.sort(key=lambda x:len(doms[x]))
                # merge i and neighbors[0]
                if neighbors:
                    doms[i].extend(doms[neighbors[0]].copy())
                    doms[neighbors[0]] = []
        if not need_merge:
            break
        n_iter += 1
    doms = [dom for dom in doms if dom]
    new_doms = []
    for dom in doms:
        try:
            new_dom = merge_intervals(dom)
        except:
            new_dom = dom
        new_doms.append(new_dom)
    return new_doms

def merge_domains_dmap(dmap, domains, n_min_res=50):
    # merge domains using distance map
    pass

if __name__ == '__main__':
    lib_dir = os.path.dirname(os.path.abspath(__file__))
    run_unidoc(
        "demo/8A00.pdb",
        lib_dir=lib_dir,
        temp_dir="demo",
    )
