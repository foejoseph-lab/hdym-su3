"""
su2_blocks.py -- SU(2) at zero angular momentum, the naive ghost split by sector, with an independent integrator.

Why: ym_run.py follows ONE tangent through all components. In finite time that tangent rides whichever sector is
growing fastest on each trajectory, so its estimate is the expected MAXIMUM of the sectors' noisy finite-time
exponents, biased upward. hdym's GPU kernel (ym/hdym_rung1_gpu.py) follows ONE off-diagonal block. This script
measures both on the same trajectories.

How: for --group su2 --L0 initial data (G = 0 and L = 0), A(t) = U diag(q(t)) V^T with U, V fixed (checked in
ym_run --selftest), so each trajectory is rotated once into its diagonal frame and integrated as hdym's 3-d system
q'' = -grad V(q), V = (x^2 y^2 + y^2 z^2 + z^2 x^2)/2, with DOP853 (rtol 1e-10; independent of ym_run's Yoshida-4).
There the naive ghost w'' = -(1 + Hess V) w splits exactly (checked to 9e-16 in this session) into
    vec:     the 3x3 diagonal block (hdym's vector ghost)
    blocks:  three 2x2 off-diagonal blocks, u'' = -(1 + [[q_k^2, s q_i q_j], [s q_i q_j, q_k^2]]) u
             (hdym's c_j = 0 Hill equations 1 + z^2 -+ xy, signs s read off ym_run's Hessian)
each with its own tangent, renormalised every 20 time units.

Initial data: exactly ym_run's worker 0 of `ym_run.py --group su2 --L0 --E 0.1 --N 1100 --workers 22 --seed 11`
(N = 50, seed 11), background burn-in 100 as in ym_run. Reported per checkpoint: per-block ensemble lam_T, the
one-block estimate (mean over the three blocks, +- standard error over trajectories), the max-of-three-blocks
estimate (what a single full tangent measures), their ratio, and the vector ghost.

    python su2_blocks.py --E 0.1 --tmax 18000 > results/su2_blocks.txt
Single core, about 30 minutes at E = 0.1. Added 2026-10-27.
"""
import argparse, os, sys, time
import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ym_run
from groups import structure_constants

ap = argparse.ArgumentParser()
ap.add_argument('--E', type=float, default=0.1)
ap.add_argument('--N', type=int, default=50)
ap.add_argument('--seed', type=int, default=11)
ap.add_argument('--tmax', type=float, default=18000.0)
ap.add_argument('--t-burn', type=float, default=100.0)
ap.add_argument('--rtol', type=float, default=1e-10)
ap.add_argument('--report', type=float, default=1000.0)
ap.add_argument('--out', default=os.path.join('results', 'su2_blocks'))
a = ap.parse_args()
T0 = time.time()
print(f'# START su2_blocks E={a.E:g} N={a.N} seed={a.seed} tmax={a.tmax:g} t_burn={a.t_burn:g} rtol={a.rtol:g}', flush=True)
f2 = structure_constants('su2')
m2 = ym_run.Model(f2)
A, P = ym_run.initial_state(f2, a.N, a.E, a.seed, L0=True)
U, q0, Vt = np.linalg.svd(A)
Pd = np.transpose(U, (0, 2, 1)) @ P @ np.transpose(Vt, (0, 2, 1))
print(f'off-diagonal momentum in each trajectory\'s diagonal frame (0 for L = 0, G = 0): {np.abs(Pd - Pd * np.eye(3)).max():.1e}')
p0 = np.einsum('nii->ni', Pd)

Ad = np.zeros((1, 3, 3)); Ad[0] = np.diag([1.3, 0.7, -0.4])
H9 = np.zeros((9, 9))
for k in range(9):
    e = np.zeros((1, 3, 3)); e.flat[k] = 1
    H9[:, k] = m2.hvp(Ad, m2.F(Ad), e).ravel()
BLOCKS = [((0, 1), 2), ((0, 2), 1), ((1, 2), 0)]
SG = [float(np.sign(H9[3 * i + j, 3 * j + i])) for (i, j), k in BLOCKS]
N = a.N


def rhs(t, Y):
    Y = Y.reshape(N, -1)
    q, p = Y[:, 0:3], Y[:, 3:6]
    x, y, z = q.T
    g = np.stack([x * (y * y + z * z), y * (x * x + z * z), z * (x * x + y * y)], 1)
    Hd = np.zeros((N, 3, 3))
    Hd[:, 0, 0] = y * y + z * z; Hd[:, 1, 1] = x * x + z * z; Hd[:, 2, 2] = x * x + y * y
    Hd[:, 0, 1] = Hd[:, 1, 0] = 2 * x * y; Hd[:, 0, 2] = Hd[:, 2, 0] = 2 * x * z; Hd[:, 1, 2] = Hd[:, 2, 1] = 2 * y * z
    w, wd = Y[:, 6:9], Y[:, 9:12]
    out = [p, -g, wd, -(w + np.einsum('nij,nj->ni', Hd, w))]
    for b, ((i, j), k) in enumerate(BLOCKS):
        u, ud = Y[:, 12 + 4 * b:14 + 4 * b], Y[:, 14 + 4 * b:16 + 4 * b]
        d = q[:, k] ** 2; c = SG[b] * q[:, i] * q[:, j]
        out += [ud, -np.stack([u[:, 0] + d * u[:, 0] + c * u[:, 1], u[:, 1] + c * u[:, 0] + d * u[:, 1]], 1)]
    return np.concatenate(out, 1).ravel()


def energy(Y):
    q, p = Y[:, 0:3], Y[:, 3:6]
    return 0.5 * (p ** 2).sum(1) + 0.5 * ((q[:, 0] * q[:, 1]) ** 2 + (q[:, 1] * q[:, 2]) ** 2 + (q[:, 0] * q[:, 2]) ** 2)


rng = np.random.default_rng(a.seed + 1)
def unit(n):
    v = rng.standard_normal((N, n)); return v / np.linalg.norm(v, axis=1, keepdims=True)


Y = np.concatenate([q0, p0, unit(6)] + [unit(4) for _ in BLOCKS], 1)
Y = solve_ivp(rhs, (0, a.t_burn), Y.ravel(), rtol=a.rtol, atol=a.rtol * 1e-2, method='DOP853').y[:, -1].reshape(N, -1)
Y[:, 6:12] = unit(6)
for b in range(3):
    Y[:, 12 + 4 * b:16 + 4 * b] = unit(4)
E0 = energy(Y)
SL = [slice(6, 12)] + [slice(12 + 4 * b, 16 + 4 * b) for b in range(3)]
logs = np.zeros((N, 4))
os.makedirs(a.out, exist_ok=True)
t, dt, nextrep = 0.0, 20.0, a.report
print(f"{'t':>7} {'block12':>10} {'block13':>10} {'block23':>10} {'one block':>10} {'+-':>8} {'max of 3':>10} "
      f"{'ratio':>6} {'vec':>10} {'max|dE|/E':>9} {'wall':>6}", flush=True)
while t < a.tmax - 1e-9:
    Y = solve_ivp(rhs, (t, t + dt), Y.ravel(), rtol=a.rtol, atol=a.rtol * 1e-2, method='DOP853').y[:, -1].reshape(N, -1)
    t += dt
    for col, s_ in enumerate(SL):
        nrm = np.linalg.norm(Y[:, s_], axis=1)
        logs[:, col] += np.log(nrm)
        Y[:, s_] /= nrm[:, None]
    if t >= nextrep - 1e-9:
        b = logs[:, 1:4] / t
        one, se, mx = b.mean(), b.mean(1).std() / np.sqrt(N), b.max(1).mean()
        print(f"{t:7.0f} {b[:, 0].mean():10.4e} {b[:, 1].mean():10.4e} {b[:, 2].mean():10.4e} {one:10.4e} {se:8.1e} "
              f"{mx:10.4e} {mx / one:6.3f} {logs[:, 0].mean() / t:10.4e} {np.abs(energy(Y) - E0).max() / a.E:9.1e} "
              f"{time.time() - T0:6.0f}", flush=True)
        np.savez(os.path.join(a.out, f'logs_E{a.E:g}.npz'), t=t, logs=logs, cols=np.array(['vec', 'b12', 'b13', 'b23']))
        nextrep += a.report
print(f'# END su2_blocks status=OK wall={time.time() - T0:.0f}s', flush=True)
