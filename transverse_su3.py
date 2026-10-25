"""
transverse_su3.py -- transverse Lyapunov exponents of the SU(2) colour-diagonal sector embedded in SU(3)
(isospin embedding), all on the SAME pure-YM trajectories p(t):

  ym_in    : ordinary YM chaos inside the sector        d'' = -Hess V(p) d
  ghost_su2: SU(2) ghost (paper's 'vec' ghost)            w'' = -(1 + Hess V(p)) w
  ym_dbl   : new SU(3) YM direction, doublet lambda_4..7  d'' = -M(p) d      (gauge mode d = p is bounded)
  ghost_dbl: ghost in the doublet direction               w'' = -(1 + M(p)) w

Linearising H = 1/2|P|^2 - 1/2|W|^2 + V(q) + w.gradV(q) - 1/2|w|^2 about w = W = 0 decouples the
p- and w-perturbations, so each line above is exact at linear order. The ghost lines are the naive
(un-gauge-reduced) operator, the same level as the paper's diagonal-sector ghost.

Integrator: Yoshida-4 for (p, P) and all tangent vectors together, fixed dt; per-component
renormalisation every `renorm` steps; exponent = mean over trajectories of accumulated log growth / time.
"""
import sys, os, time, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ym'))
import hdym_core as su2
from doublet import M_dot

C, D = su2.YOSHIDA_C, su2.YOSHIDA_D


def run(E, N, dt, tmax, t_burn, renorm, seed):
    rng = np.random.default_rng(seed)
    p, P = su2.ym_initial(N, E, rng)
    # burn in pure YM so the ensemble is on the energy shell's invariant measure
    for _ in range(int(t_burn / dt)):
        for c, d in zip(C, D):
            p = p + c * dt * P
            if d:
                P = P - d * dt * su2.gradV(p)
    ops = {
        'ym_in':     lambda p, v: su2.hessV_dot(p, v),
        'ghost_su2': lambda p, v: v + su2.hessV_dot(p, v),
        'ym_dbl':    lambda p, v: M_dot(p, v),
        'ghost_dbl': lambda p, v: v + M_dot(p, v),
    }
    tan = {}
    for k in ops:
        x = rng.standard_normal((N, 3)); v = rng.standard_normal((N, 3))
        n = np.sqrt(np.sum(x * x + v * v, 1))[:, None]
        tan[k] = [x / n, v / n]
    logs = {k: np.zeros(N) for k in ops}
    H0 = 0.5 * np.sum(P * P, 1) + su2.V(p)
    nsteps = int(tmax / dt)
    for s in range(1, nsteps + 1):
        for c, d in zip(C, D):
            p = p + c * dt * P
            for k in ops:
                tan[k][0] = tan[k][0] + c * dt * tan[k][1]
            if d:
                P = P - d * dt * su2.gradV(p)
                for k, op in ops.items():
                    tan[k][1] = tan[k][1] - d * dt * op(p, tan[k][0])
        if s % renorm == 0:
            for k in ops:
                x, v = tan[k]
                n = np.sqrt(np.sum(x * x + v * v, 1))
                logs[k] += np.log(n)
                tan[k] = [x / n[:, None], v / n[:, None]]
    T = nsteps * dt
    dE = np.abs(0.5 * np.sum(P * P, 1) + su2.V(p) - H0).max() / E
    out = {}
    for k in ops:
        lam = logs[k] / T
        out[k] = (lam.mean(), lam.std() / np.sqrt(N))
    return out, dE


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--E', type=float, nargs='+', default=[1.0])
    ap.add_argument('--N', type=int, default=400)
    ap.add_argument('--dt', type=float, default=0.01)
    ap.add_argument('--tmax', type=float, default=2000.0)
    ap.add_argument('--t-burn', type=float, default=100.0)
    ap.add_argument('--renorm', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    print(f'# N={a.N} dt={a.dt} tmax={a.tmax} t_burn={a.t_burn} seed={a.seed}', flush=True)
    print(f'# {"E":>6} ' + ' '.join(f'{k:>22}' for k in ('ym_in', 'ghost_su2', 'ym_dbl', 'ghost_dbl')) + '   max|dE|/E  wall', flush=True)
    for E in a.E:
        t0 = time.time()
        res, dE = run(E, a.N, a.dt, a.tmax, a.t_burn, a.renorm, a.seed)
        print(f'  {E:6.3f} ' + ' '.join(f'{m:11.5f} ± {s:8.5f}' for m, s in res.values())
              + f'   {dE:.1e}  {time.time() - t0:.0f}s', flush=True)
