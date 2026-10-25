"""doublet.py -- the 3x3 doublet block M(q): the 12x12 lambda_4..7 Hessian block on the isospin
embedding is four identical copies of M (copy 1 = (A1^4, A2^5, -A3^6); see derive_doublet.py)."""
import numpy as np


def M_dot(q, v):
    x, y, z = q[:, 0], q[:, 1], q[:, 2]
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    return np.stack([0.25 * (y*y + z*z) * vx + 0.75 * x*y * vy + 0.75 * x*z * vz,
                     0.75 * x*y * vx + 0.25 * (z*z + x*x) * vy + 0.75 * y*z * vz,
                     0.75 * x*z * vx + 0.75 * y*z * vy + 0.25 * (x*x + y*y) * vz], axis=1)


def M_mat(q):
    e = np.eye(3)
    return np.stack([M_dot(q, np.broadcast_to(e[j], q.shape)) for j in range(3)], axis=2)
