"""
L_correlation.py -- how each trajectory's exponents depend on its spatial angular momentum L (conserved).

Initial data are deterministic in the seeds, so |L| of every trajectory of a finished run can be recomputed
without rerunning it: worker k drew its trajectories with seed + 1000 k (su3_full.py and ym_run.py alike), and
ym_run.initial_state reproduces su3_full.initial_state bit for bit. The per-trajectory exponents are read from
the run's lam_E<E>.npz (lam_ghost / lam_ym: su3_full's estimator, and ym_run's lam_T).

Prints, per energy: Spearman rank correlation of ell = |L| / E^{3/4} (scale-invariant) with the ghost and the
YM exponent, and the mean ghost exponent in |L| percentile bins relative to the ensemble mean.

    python L_correlation.py --dir results/full_scan --E 1 0.5 0.3 0.2 0.1 --N 4400 --workers 22
    python L_correlation.py --dir results/run_low --E 0.3 0.1 0.05 --N 1100 --workers 22
"""
import argparse, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ym_run
from groups import structure_constants


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', required=True)
    ap.add_argument('--E', type=float, nargs='+', required=True)
    ap.add_argument('--N', type=int, required=True)
    ap.add_argument('--workers', type=int, required=True)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--group', default='su3')
    a = ap.parse_args()
    t0 = time.time()
    print(f'# START L_correlation dir={a.dir} N={a.N} workers={a.workers} seed={a.seed} group={a.group}', flush=True)
    f = structure_constants(a.group)
    sizes = [a.N // a.workers + (1 if k < a.N % a.workers else 0) for k in range(a.workers)]
    edges = [0, 10, 25, 50, 75, 90, 100]
    print(f"{'E':>6} {'n':>5} {'rho(ell,ghost)':>15} {'rho(ell,YM)':>12}   ghost/mean in |L| percentile bins "
          f"{' '.join(f'{edges[i]}-{edges[i + 1]}' for i in range(6))}")
    for E in a.E:
        d = np.load(os.path.join(a.dir, f'lam_E{E:g}.npz'))
        g, y = d['lam_ghost'], d['lam_ym']
        if 'Lmag' in d.files and d['Lmag'].max() < 1e-8:
            print(f'{E:6g} {g.size:5d}   an --L0 run: |L| = 0 on every trajectory (max {d["Lmag"].max():.1e}), '
                  f'nothing to correlate', flush=True)
            continue
        Ls = []
        for k, n in enumerate(sizes):
            if n:
                A, P = ym_run.initial_state(f, n, E, a.seed + 1000 * k)
                Ls.append(np.linalg.norm(ym_run.angmom(A, P), axis=1))
        ell = np.concatenate(Ls) / E ** 0.75
        if ell.size != g.size:
            raise SystemExit(f'size mismatch at E={E:g}: {ell.size} recomputed vs {g.size} in the npz')
        if 'Lmag' in d.files:          # ym_run runs from the --L0 patch on store |L| at the end: cross-check
            print(f'   stored |L| vs recomputed: max diff {np.abs(d["Lmag"] / E ** 0.75 - ell).max():.1e}')
        if 'L' in d.files:             # and, from 26 Oct, the signed vector
            Lv = np.concatenate([np.asarray(ym_run.angmom(*ym_run.initial_state(f, n, E, a.seed + 1000 * k)))
                                 for k, n in enumerate(sizes) if n])
            print(f'   stored L vector vs recomputed: max diff {np.abs(d["L"] - Lv).max():.1e}')
        q = np.percentile(ell, edges)
        bins = [g[(ell >= q[i]) & (ell <= q[i + 1])].mean() / g.mean() for i in range(6)]
        print(f'{E:6g} {g.size:5d} {spearman(ell, g):+15.3f} {spearman(ell, y):+12.3f}   '
              + '  '.join(f'{b:5.2f}' for b in bins), flush=True)
    print(f'# END L_correlation status=OK wall={time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
