"""
doublet_physical.py -- is the pure-YM doublet escape (ym_dbl in transverse_su3.py) physical?

In one 3x3 doublet copy the linearised YM flow is d'' = -M(p) d, with
  * gauge mode       (d, d') = (p, p')      exact solution, since M p = grad V(p);
  * Gauss charge     G = p.d' - p'.d         conserved (M symmetric).
Physical perturbations have G = 0 and are taken modulo the gauge mode.

Three estimators on the same trajectories:
  raw   : random (d, d'), plain norm                      (= ym_dbl of transverse_su3.py)
  gauss : start on G = 0                                   (G monitored)
  phys  : start on G = 0 and remove the gauge-mode component at each renormalisation
          (projection orthogonal to (p, p') in the phase-space Euclidean metric)
If 'phys' equals 'raw', the escape lives in the physical sector.
"""
import sys, os, argparse, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ym'))
import hdym_core as su2
from doublet import M_dot

C, D = su2.YOSHIDA_C, su2.YOSHIDA_D


def gauss(p, P, d, v):
    return np.sum(p * v - P * d, 1)


def remove_gauge(p, P, d, v):
    nn = np.sum(p * p + P * P, 1)
    c = np.sum(p * d + P * v, 1) / nn
    return d - c[:, None] * p, v - c[:, None] * P


def main(E, N, dt, tmax, t_burn, renorm, seed, odt=0.25):
    rng = np.random.default_rng(seed)
    p, P = su2.ym_initial(N, E, rng)
    for _ in range(int(t_burn / dt)):
        p, P = su2.ym_step_substepped(p, P, dt, odt)
    tan = {}
    for k in ('raw', 'gauss', 'phys'):
        d = rng.standard_normal((N, 3)); v = rng.standard_normal((N, 3))
        if k != 'raw':   # put on G = 0 by shifting v along p
            v = v - (gauss(p, P, d, v) / np.sum(p * p, 1))[:, None] * p
        if k == 'phys':
            d, v = remove_gauge(p, P, d, v)
        n = np.sqrt(np.sum(d * d + v * v, 1))[:, None]
        tan[k] = [d / n, v / n]
    logs = {k: np.zeros(N) for k in tan}
    Gmax = 0.0

    def step(p, P, tn, h):
        for c, dd in zip(C, D):
            p = p + c * h * P
            for k in tn:
                tn[k][0] = tn[k][0] + c * h * tn[k][1]
            if dd:
                P = P - dd * h * su2.gradV(p)
                for k in tn:
                    tn[k][1] = tn[k][1] - dd * h * M_dot(p, tn[k][0])
        return p, P

    nsteps = int(tmax / dt)
    for s in range(1, nsteps + 1):
        omega = np.maximum(np.max(np.abs(p), axis=1), 1.0)
        nsub = np.maximum(1, np.ceil(omega * dt / odt)).astype(int)
        for n in np.unique(nsub):
            idx = np.nonzero(nsub == n)[0]
            a, b = p[idx], P[idx]
            tn = {k: [tan[k][0][idx], tan[k][1][idx]] for k in tan}
            for _ in range(n):
                a, b = step(a, b, tn, dt / n)
            p[idx], P[idx] = a, b
            for k in tan:
                tan[k][0][idx], tan[k][1][idx] = tn[k]
        if s % renorm == 0:
            for k in tan:
                d, v = tan[k]
                if k != 'raw':
                    n0 = np.sqrt(np.sum(d * d + v * v, 1))
                    Gmax = max(Gmax, np.abs(gauss(p, P, d, v) / n0).max())
                if k == 'phys':
                    d, v = remove_gauge(p, P, d, v)
                n = np.sqrt(np.sum(d * d + v * v, 1))
                logs[k] += np.log(n)
                tan[k] = [d / n[:, None], v / n[:, None]]
    T = nsteps * dt
    print(f'# E={E} N={N} dt={dt} tmax={tmax} seed={seed}')
    for k in tan:
        lam = logs[k] / T
        print(f'  {k:6s} {lam.mean():.5f} ± {lam.std() / np.sqrt(N):.5f}   (/E^1/4 = {lam.mean() / E**0.25:.5f})')
    print(f'  max |G|/|tangent| on constrained vectors: {Gmax:.1e}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--E', type=float, default=1.0)
    ap.add_argument('--N', type=int, default=300)
    ap.add_argument('--dt', type=float, default=0.005)
    ap.add_argument('--tmax', type=float, default=1000.0)
    ap.add_argument('--t-burn', type=float, default=100.0)
    ap.add_argument('--renorm', type=int, default=20)
    ap.add_argument('--seed', type=int, default=3)
    a = ap.parse_args()
    t0 = time.time()
    main(a.E, a.N, a.dt, a.tmax, a.t_burn, a.renorm, a.seed)
    print(f'  wall {time.time() - t0:.0f}s')
