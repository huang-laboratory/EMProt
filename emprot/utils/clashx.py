import numpy as np
from numba import njit
from typing import Tuple, Optional


@njit
def _get_clash_kernel(
    crda: np.ndarray,
    densa: np.ndarray,
    crdb: np.ndarray,
    densb: np.ndarray,
    resol: float = 5.0,
    clash_dist: float = 1.5,
    compute_b_scores: bool = True, 
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Numba-accelerated kernel function for clash calculation."""
    PI = np.pi
    na = crda.shape[0]
    nb = crdb.shape[0]

    rsigma2 = (PI / (2.4 + 0.8 * resol)) ** 2
    bw2 = 6.0 / rsigma2
    bw = np.sqrt(bw2)
    sqrt3bw = np.sqrt(3.0) * bw

    scoresa = np.zeros(na)
    mind2a = np.full(na, bw2)
    
    if compute_b_scores:
        scoresb = np.zeros(nb)
        mind2b = np.full(nb, bw2)
    else:
        scoresb = None

    for i in range(na):
        for j in range(nb):
            dens = densa[i] * densb[j]
            ftmp = np.abs(crda[i] - crdb[j])
            
            # Early termination if too far apart
            if np.max(ftmp) > bw or np.sum(ftmp) > sqrt3bw:
                continue
                
            d2 = np.sum(ftmp**2)
            d2s = max(np.sqrt(d2) - clash_dist, 0.0) ** 2
            
            if d2 < bw2:
                prob = np.exp(-rsigma2 * d2s)
                
                if d2 < mind2a[i]:
                    scoresa[i] = prob * dens
                    mind2a[i] = d2
                
                if compute_b_scores and d2 < mind2b[j]:
                    scoresb[j] = prob * dens
                    mind2b[j] = d2

    return scoresa, scoresb

def get_clash(
    crda: np.ndarray,
    densa: np.ndarray,
    crdb: np.ndarray,
    densb: np.ndarray,
    resol: float = 5.0,
    clash_dist: float = 1.0,
    return_b_scores: bool = True, 
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Calculate clash scores between two sets of coordinates with Numba acceleration.
    
    Args:
        crda: (n, 3) array of coordinates for set A
        densa: (n,) array of densities for set A
        crdb: (m, 3) array of coordinates for set B
        densb: (m,) array of densities for set B
        resol: Resolution parameter
        clash_dist: Clash distance threshold
        return_b_scores: Whether to calculate scores for set B
        
    Returns:
        Tuple containing:
        - scoresa: (n,) array of clash scores for set A
        - scoresb: (m,) array of clash scores for set B (if return_b_scores=True)
                   None otherwise
    """
    # Input validation
    assert crda.shape[1] == 3, "crda must be (n, 3) array"
    assert crdb.shape[1] == 3, "crdb must be (m, 3) array"
    assert len(crda) == len(densa), "crda and densa must have same length"
    assert len(crdb) == len(densb), "crdb and densb must have same length"
    
    scoresa, scoresb = _get_clash_kernel(
        crda, densa, crdb, densb, resol, clash_dist, return_b_scores
    )
    
    return (scoresa, scoresb) if return_b_scores else (scoresa, None)
    

def main(args):
    fa = args.a
    fb = args.b
    from emprot.io.pdbio import read_pdb

    a_pos, _, _, _, _ = read_pdb(fa)
    b_pos, _, _, _, _ = read_pdb(fb)

   
    clash_a, clash_b = get_clash(
        a_pos[..., 1, :],
        np.ones( len(a_pos) ),
        b_pos[..., 1, :],
        np.ones( len(b_pos) ),
    )

    clash_a /= len(clash_a)

    print(clash_a.sum(), len(clash_a))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-a")
    parser.add_argument("-b")
    args = parser.parse_args()
    main(args)
