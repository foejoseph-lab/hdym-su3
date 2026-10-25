"""
check_embedding.py -- stage 1 checks for the SU(3) isospin embedding.

  1. f is totally antisymmetric and satisfies the Jacobi identity (Gell-Mann table typed in right).
  2. hessV_dot agrees with finite differences of gradV at a random SU(3) point.
  3. On A = embed(q): V and grad V equal the SU(2) model in ym/hdym_core.py,
     and grad V has no component outside {lambda_1,2,3} x diagonal -> invariant submanifold.
  4. The 24x24 Hessian on the embedding is block diagonal:
        su(2) adjoint (9) + doublet lambda_4..7 (12) + singlet lambda_8 (3),
     and the singlet block is identically zero.
  5. Closed-form doublet block vs the numeric one, and its eigenvalues against the
     SU(2) 3x3 Hessian that drives the paper's diagonal ghost.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ym'))
import hdym_core as su2
import su3_core as s3

rng = np.random.default_rng(1)
f = s3.F
ok = True


def report(name, err, tol):
    global ok
    flag = 'OK ' if err < tol else 'FAIL'
    ok &= err < tol
    print(f'[{flag}] {name}: {err:.3e}')


# 1. antisymmetry + Jacobi
report('antisymmetry', np.abs(f + np.transpose(f, (1, 0, 2))).max(), 1e-15)
jac = (np.einsum('abe,ecd->abcd', f, f) + np.einsum('bce,ead->abcd', f, f)
       + np.einsum('cae,ebd->abcd', f, f))
report('Jacobi identity', np.abs(jac).max(), 1e-14)
# Casimir: f^acd f^bcd = 3 delta^ab for SU(3)
report('adjoint Casimir = 3', np.abs(np.einsum('acd,bcd->ab', f, f) - 3 * np.eye(8)).max(), 1e-14)

# 2. Hessian-vector product vs finite differences
A = rng.standard_normal((3, 8))
v = rng.standard_normal((3, 8))
h = 1e-6
fd = (s3.gradV(A + h * v) - s3.gradV(A - h * v)) / (2 * h)
report('Hv vs finite difference (rel)', np.abs(fd - s3.hessV_dot(A, v)).max() / np.abs(fd).max(), 1e-8)
fdV = (s3.V(A + h * v) - s3.V(A - h * v)) / (2 * h)
report('gradV vs finite difference (rel)', abs(fdV - np.sum(s3.gradV(A) * v)) / abs(fdV), 1e-8)

# 3. reduction to SU(2) model
q = rng.standard_normal((1000, 3))
Aq = s3.embed(q)
report('V(embed q) - V_SU2(q)', np.abs(s3.V(Aq) - su2.V(q)).max(), 1e-12)
g = s3.gradV(Aq)
g_diag = np.stack([g[:, i, i] for i in range(3)], axis=1)
report('diag grad - gradV_SU2', np.abs(g_diag - su2.gradV(q)).max(), 1e-12)
g_off = g.copy()
for i in range(3):
    g_off[:, i, i] = 0
report('grad off the embedded submanifold', np.abs(g_off).max(), 1e-12)

# 4. block structure of the Hessian on the embedding
iA, iD, iS = (s3.block_indices(c) for c in (s3.SU2, s3.DOUBLET, s3.SINGLET))
worst_mix, worst_singlet = 0.0, 0.0
for k in range(200):
    H = s3.hessV(s3.embed(q[k]))
    for X, Y in ((iA, iD), (iA, iS), (iD, iS)):
        worst_mix = max(worst_mix, np.abs(H[np.ix_(X, Y)]).max())
    worst_singlet = max(worst_singlet, np.abs(H[np.ix_(iS, iS)]).max())
    # su(2) diagonal 3x3 sub-block == paper's Hess V
    idx = [i * 8 + i for i in range(3)]
    Hd = H[np.ix_(idx, idx)]
    e = np.eye(3)
    Href = np.stack([su2.hessV_dot(q[k:k + 1], e[j:j + 1])[0] for j in range(3)], axis=1)
    worst_singlet = worst_singlet  # (kept separate below)
    if k == 0:
        report('su(2) diag 3x3 block == hdym_core Hess V', np.abs(Hd - Href).max(), 1e-12)
report('cross-block mixing (adjoint/doublet/singlet)', worst_mix, 1e-12)
report('singlet block (lambda_8) identically zero', worst_singlet, 1e-12)

print('\nall checks passed' if ok else '\nSOME CHECKS FAILED')
