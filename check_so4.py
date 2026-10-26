"""
check_so4.py -- SO(4) as a "should match" check of the group-general code (ym_group.py).

so(4) = su(2) + su(2), each with coupling g = sqrt2 in the orthonormal basis L+- = (J +- K)/sqrt2 (groups.py).
Hence homogeneous SO(4) Yang-Mills is two decoupled SU(2) systems, and with g = sqrt2 scaled away
(B = g A, same time):
    an SO(4) state with all its energy E in one half  ==  a generic SU(2) state at energy g^2 E = 2E,
for the background AND for both tangents (YM: d'' = -Hess V d; ghost: w'' = -(1 + Hess V) w; the ghost
mass 1 is not rescaled because Hess_A V_half(A) = Hess V_su2(B) for the quartic). So the finite-time
log-growth of every tangent must agree trajectory by trajectory, up to integration error.

Checks:
  1. f antisymmetric + Jacobi for su2, su3, so4; so4 in the split basis is block-diag(sqrt2 eps, sqrt2 eps).
  2. ym_group --group su3 reproduces su3_full.py exactly (same seed -> same exponents).
  3. ym_group --group su2: on the colour-diagonal sector V and grad V equal hdym_core.py (the paper's model).
  4. Generic SO(4) flow: the energy of each half is conserved separately (exact decoupling).
  5. The should-match: half-loaded SO(4) at E vs generic SU(2) at 2E, per trajectory, both tangents.
"""
import sys, os
import numpy as np
import time
_t0 = time.time()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import groups
import ym_group as yg
import su3_full
import hdym_core

ok = True


def rep(name, err, tol):
    global ok
    good = bool(err < tol)
    ok &= good
    print(f"[{'OK ' if good else 'FAIL'}] {name}: {err:.2e}")


rng = np.random.default_rng(42)
eps = groups.f_su2()
R = groups.so4_split_basis()

# 1. algebra
for g in ('su2', 'su3', 'so4'):
    f = groups.structure_constants(g)
    jac = (np.einsum('abe,ecd->abcd', f, f) + np.einsum('bce,ead->abcd', f, f) + np.einsum('cae,ebd->abcd', f, f))
    rep(f'{g}: antisymmetry + Jacobi', max(np.abs(f + np.transpose(f, (1, 0, 2))).max(), np.abs(jac).max()), 1e-14)
f4 = groups.f_so4()
f4s = np.einsum('ai,bj,ck,ijk->abc', R, R, R, f4)            # structure constants in the split basis
target = np.zeros((6, 6, 6))
target[:3, :3, :3] = np.sqrt(2) * eps
target[3:, 3:, 3:] = np.sqrt(2) * eps
rep('so4 = su2(+) + su2(-), each with g = sqrt2', np.abs(f4s - target).max(), 1e-14)

# 2. ym_group su3 == su3_full
kw = dict(E=0.7, N=40, seed=5, dt=0.01, tmax=20.0, t_burn=5.0, renorm=20, odt=0.25, backend='numpy', t_block=20.0)
r_old = su3_full.run_chunk(**kw)
r_new = yg.run_chunk(**kw, group='su3')
rep('ym_group su3 == su3_full (lam_ym, lam_ghost)',
    max(np.abs(r_old['lam_ym'] - r_new['lam_ym']).max(), np.abs(r_old['lam_ghost'] - r_new['lam_ghost']).max()), 1e-12)

# 3. su2 diagonal sector == hdym_core
m2 = yg.Model(np, 'su2')
q = rng.standard_normal((200, 3))
A = np.zeros((200, 3, 3))
for i in range(3):
    A[:, i, i] = q[:, i]
Fs = m2.F(A)
g2 = m2.grad(A, Fs)
rep('su2 diagonal sector: V == hdym_core.V', np.abs(m2.V(A, Fs) - hdym_core.V(q)).max(), 1e-12)
rep('su2 diagonal sector: grad == hdym_core.gradV',
    np.abs(np.stack([g2[:, i, i] for i in range(3)], 1) - hdym_core.gradV(q)).max(), 1e-12)


# helpers for SO(4) <-> split basis (colour index is last; A_JK = A_split @ R)
def to_split(X):
    return X @ R.T


def to_JK(X):
    return X @ R


m4 = yg.Model(np, 'so4')
m2g = yg.Model(np, 'su2')


def half_energy(A, P):
    """Energy of each half in the split basis; each half is SU(2) with f = sqrt2 eps."""
    As, Ps = to_split(A), to_split(P)
    out = []
    for sl in (slice(0, 3), slice(3, 6)):
        B = np.sqrt(2) * As[..., sl]                           # B = g A: SU(2) with g = 1
        out.append(0.5 * np.sum(Ps[..., sl] ** 2, axis=(1, 2)) + m2g.V(B) / 2.0)   # V_half(A) = V_su2(B)/g^2
    return out


# 4. decoupling
A, P = yg.initial_state(64, 1.0, 9, 'so4')
Ep0, Em0 = half_energy(A, P)
rep('so4 split energies add up to H', np.abs(Ep0 + Em0 - (0.5 * np.sum(P * P, axis=(1, 2)) + m4.V(A))).max(), 1e-12)
none = [[np.zeros_like(A), np.zeros_like(A)], [np.zeros_like(A), np.zeros_like(A)]]
for _ in range(4000):                                          # t = 20
    A, P = yg._advance(m4, A, P, none, 0.005, 0.25)
Ep1, Em1 = half_energy(A, P)
# A coupling would move energy between the halves, so each half's drift would exceed the total's.
# Uncoupled, each half drifts only by integration error, i.e. no more than the total does
# (at dt = 0.005, 0.0025, 0.00125: half 1.47e-7, 9.2e-9, 5.7e-10 = total, falling 16x per halving).
half_drift = max(np.abs(Ep1 - Ep0).max(), np.abs(Em1 - Em0).max())
total_drift = np.abs(Ep1 + Em1 - Ep0 - Em0).max()
rep('each half conserves its own energy: half drift / total drift (t = 20)', half_drift / total_drift, 1.5)
share = np.minimum(Ep0, Em0) / (Ep0 + Em0)
rep('the test is not trivial: every trajectory shares energy, 1 / (min share of the smaller half)', 1 / share.min(), 1e3)

# 5. should-match: half-loaded SO(4) at E  vs  SU(2) at 2E
E, N, dt, T = 0.5, 64, 0.0025, 30.0
B, Q = yg.initial_state(N, 2 * E, 21, 'su2')                    # SU(2), g = 1, energy 2E
d0 = rng.standard_normal((N, 3, 3)); dv0 = rng.standard_normal((N, 3, 3))
w0 = rng.standard_normal((N, 3, 3)); wv0 = rng.standard_normal((N, 3, 3))
s2 = [B.copy(), Q.copy(), [[d0.copy(), dv0.copy()], [w0.copy(), wv0.copy()]]]


def embed_plus(X):
    """SU(2) (g = 1) array -> SO(4) (J,K) array with only the + half, scaled by 1/g."""
    Xs = np.zeros(X.shape[:-1] + (6,))
    Xs[..., :3] = X / np.sqrt(2)
    return to_JK(Xs)


A4, P4 = embed_plus(B), embed_plus(Q)
rep('half-loaded SO(4) energy == E', np.abs(0.5 * np.sum(P4 * P4, axis=(1, 2)) + m4.V(A4) - E).max(), 1e-12)
s4 = [A4, P4, [[embed_plus(d0), embed_plus(dv0)], [embed_plus(w0), embed_plus(wv0)]]]


def lognorm(t):
    return np.log(np.sqrt(np.sum(t[0] ** 2 + t[1] ** 2, axis=(1, 2))))


l2_0 = [lognorm(t) for t in s2[2]]
l4_0 = [lognorm(t) for t in s4[2]]
for _ in range(int(T / dt)):
    s2[0], s2[1] = yg._advance(m2g, s2[0], s2[1], s2[2], dt, 0.05)
    s4[0], s4[1] = yg._advance(m4, s4[0], s4[1], s4[2], dt, 0.05)
for k, name in enumerate(('YM tangent', 'ghost tangent')):
    g2_ = lognorm(s2[2][k]) - l2_0[k]
    g4_ = lognorm(s4[2][k]) - l4_0[k]
    rep(f'should-match, {name}: |log-growth SO(4) - SU(2)| per trajectory (growth up to {g2_.max():.1f})',
        np.abs(g4_ - g2_).max(), 1e-6)
rep('should-match, background: |A_SO(4) - embed(A_SU(2))|', np.abs(s4[0] - embed_plus(s2[0])).max(), 1e-8)

print('\nall checks passed' if ok else '\nSOME CHECKS FAILED')
print(f'# END check_so4 status={"OK" if ok else "FAIL"} wall={time.time() - _t0:.0f}s', flush=True)
sys.exit(0 if ok else 1)
