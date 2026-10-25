"""
su3_core.py -- homogeneous SU(3) Yang-Mills potential, units g = 1.

Same conventions as ym/hdym_derive.py:  L = -1/4 F^a F^a,  F^a_ij = f^abc A_i^b A_j^c
for spatially homogeneous fields, so

    V(A) = 1/4 sum_{i,j,a} (F^a_ij)^2 ,     A has shape (..., 3, 8)  [space i, colour a]

With f = Gell-Mann structure constants (T^a = lambda^a / 2, [T^a,T^b] = i f^abc T^c).

Isospin embedding: A_i^a = delta_i^a q_i for a in {1,2,3}, zero otherwise.
Since f^{abc} = eps^{abc} on {1,2,3}, this reproduces the SU(2) model
V = 1/2 (x^2 y^2 + y^2 z^2 + z^2 x^2) of ym/hdym_core.py exactly.
"""
import numpy as np

# ------------------------------------------------------------ structure constants
_F_NONZERO = {
    (1, 2, 3): 1.0,
    (1, 4, 7): 0.5, (1, 5, 6): -0.5,
    (2, 4, 6): 0.5, (2, 5, 7): 0.5,
    (3, 4, 5): 0.5, (3, 6, 7): -0.5,
    (4, 5, 8): np.sqrt(3) / 2, (6, 7, 8): np.sqrt(3) / 2,
}


def structure_constants():
    f = np.zeros((8, 8, 8))
    for (a, b, c), v in _F_NONZERO.items():
        a, b, c = a - 1, b - 1, c - 1
        for (i, j, k), s in (((a, b, c), 1), ((b, c, a), 1), ((c, a, b), 1),
                             ((b, a, c), -1), ((a, c, b), -1), ((c, b, a), -1)):
            f[i, j, k] = s * v
    return f


F = structure_constants()

# colour index groups (0-based)
SU2 = [0, 1, 2]          # lambda_1..3  (isospin adjoint)
DOUBLET = [3, 4, 5, 6]   # lambda_4..7  (two real components of one complex doublet)
SINGLET = [7]            # lambda_8     (hypercharge)


# ------------------------------------------------------------ generic potential
def field_strength(A):
    """F[..., i, j, a] = f^abc A_i^b A_j^c."""
    return np.einsum('abc,...ib,...jc->...ija', F, A, A)


def V(A):
    Fs = field_strength(A)
    return 0.25 * np.sum(Fs * Fs, axis=(-3, -2, -1))


def gradV(A):
    """dV/dA_i^b = sum_j f^abc F_ij^a A_j^c."""
    Fs = field_strength(A)
    return np.einsum('abc,...ija,...jc->...ib', F, Fs, A)


def hessV_dot(A, v):
    """Directional derivative of gradV along v (exact Hessian-vector product)."""
    Fs = field_strength(A)
    dF = (np.einsum('ade,...id,...je->...ija', F, v, A)
          + np.einsum('ade,...id,...je->...ija', F, A, v))
    return (np.einsum('abc,...ija,...jc->...ib', F, dF, A)
            + np.einsum('abc,...ija,...jc->...ib', F, Fs, v))


def hessV(A):
    """Full 24x24 Hessian for a single configuration A of shape (3, 8)."""
    H = np.zeros((24, 24))
    for k in range(24):
        e = np.zeros(24)
        e[k] = 1.0
        H[:, k] = hessV_dot(A, e.reshape(3, 8)).ravel()
    return 0.5 * (H + H.T)


def embed(q):
    """Isospin embedding of the SU(2) colour-diagonal point q (..., 3)."""
    A = np.zeros(q.shape[:-1] + (3, 8))
    for i in range(3):
        A[..., i, i] = q[..., i]
    return A


def block_indices(colours):
    """Flat indices (i*8 + a) of the 24-vector for the given colour set, space-major."""
    return [i * 8 + a for i in range(3) for a in colours]
