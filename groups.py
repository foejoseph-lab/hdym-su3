"""
groups.py -- real structure constants f^abc for the gauge groups this project compares, in one convention:
    [T^a, T^b] = i f^abc T^c,   F^a_ij = f^abc A_i^b A_j^c,   V = 1/4 F^a_ij F^a_ij   (units g = 1)

  su2 : f = eps (3 generators)                                   -- hdym's group
  su3 : Gell-Mann, T = lambda/2 (8 generators)                   -- as su3_core.py
  so4 : rotation generators J_k and "boosts" K_k of so(4) (6 generators, order J1 J2 J3 K1 K2 K3),
        [J_i, J_j] = eps J_k,  [J_i, K_j] = eps K_k,  [K_i, K_j] = eps J_k.
        In the orthonormal basis  L+- = (J +- K)/sqrt2  it is two commuting su(2) with f = sqrt2 * eps each,
        i.e. so(4) = su(2) + su(2), each with coupling g = sqrt2 (checked in check_so4.py).
"""
import numpy as np


def _from_triples(n, triples):
    f = np.zeros((n, n, n))
    for (a, b, c), v in triples.items():
        a, b, c = a - 1, b - 1, c - 1
        for (i, j, k), s in (((a, b, c), 1), ((b, c, a), 1), ((c, a, b), 1),
                             ((b, a, c), -1), ((a, c, b), -1), ((c, b, a), -1)):
            f[i, j, k] = s * v
    return f


def f_su2():
    return _from_triples(3, {(1, 2, 3): 1.0})


def f_su3():
    return _from_triples(8, {(1, 2, 3): 1.0, (1, 4, 7): 0.5, (1, 5, 6): -0.5, (2, 4, 6): 0.5, (2, 5, 7): 0.5,
                             (3, 4, 5): 0.5, (3, 6, 7): -0.5, (4, 5, 8): np.sqrt(3) / 2,
                             (6, 7, 8): np.sqrt(3) / 2})


def f_so4():
    # J = 1,2,3 ; K = 4,5,6
    return _from_triples(6, {(1, 2, 3): 1.0,                      # [J,J] = J
                             (1, 5, 6): 1.0, (2, 6, 4): 1.0, (3, 4, 5): 1.0})  # [J,K] = K and [K,K] = J


GROUPS = {'su2': f_su2, 'su3': f_su3, 'so4': f_so4}


def structure_constants(group):
    return GROUPS[group]()


def so4_split_basis():
    """Orthonormal change of basis R (6x6), rows = new generators L+_1..3, L-_1..3 in terms of (J, K)."""
    s = 1 / np.sqrt(2)
    I = np.eye(3)
    return np.block([[s * I, s * I], [s * I, -s * I]])
