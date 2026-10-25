"""
ym_group.py -- generic homogeneous Yang-Mills for any group in groups.py (su2, su3, so4) and its Lee-Wick ghost.

The same numerics as su3_full.py with the group as a parameter (--group): A[n, i, a] with a = 1..dim G,
initial data on the Gauss shell G^a = f^abc A_i^b P_i^c = 0 scaled to energy E exactly, Yoshida-4 with
per-trajectory sub-stepping, and two 3*dimG tangents:
    ym    : d'' = -Hess V(A) d            -> lambda_YM
    ghost : w'' = -(1 + Hess V(A)) w      -> lambda_ghost   (naive operator, as su3_full.py)
su3_full.py is kept unchanged (it produced the committed SU(3) logs); --group su3 here reproduces it exactly
(check_so4.py checks that).

    python ym_group.py --group so4 --E 1.0 --N 2200 --tmax 300 --workers 22 --out results/so4_smoke
"""
import argparse, csv, os, sys, time
import numpy as np

from groups import structure_constants


_C = 2.0 ** (1.0 / 3.0)
_W1 = 1.0 / (2.0 - _C)
_W0 = -_C / (2.0 - _C)
YC = (_W1 / 2, (_W0 + _W1) / 2, (_W0 + _W1) / 2, _W1 / 2)
YD = (_W1, _W0, _W1, 0.0)


class Model:
    """All array maths on backend xp (numpy or cupy)."""

    def __init__(self, xp, group):
        self.xp = xp
        f = structure_constants(group)
        self.nc = f.shape[0]
        self.f = xp.asarray(f)

    # Two contractions, written as tensordot + batched matmul (fast on CPU BLAS and on the GPU):
    #   bracket(X, Y)[n,i,j,a] = f^abc X_i^b Y_j^c
    #   pull(Fs, Y)[n,i,b]      = f^abc sum_j Fs_ij^a Y_j^c
    def bracket(self, X, Y):
        xp = self.xp
        n = X.shape[0]
        T = xp.tensordot(X, self.f, axes=([2], [1]))                 # (n, i, a, c)
        nc = self.nc
        out = xp.matmul(T.reshape(n, 3 * nc, nc), xp.transpose(Y, (0, 2, 1)))   # (n, i*a, j)
        return xp.transpose(out.reshape(n, 3, nc, 3), (0, 1, 3, 2))   # (n, i, j, a)

    def pull(self, Fs, Y):
        xp = self.xp
        n = Fs.shape[0]
        nc = self.nc
        S = xp.matmul(xp.transpose(Fs, (0, 1, 3, 2)).reshape(n, 3 * nc, 3), Y)  # (n, i*a, c)
        return xp.tensordot(S.reshape(n, 3, nc, nc), self.f, axes=([2, 3], [0, 2]))  # (n, i, b)

    def F(self, A):
        return self.bracket(A, A)

    def V(self, A, Fs=None):
        Fs = self.F(A) if Fs is None else Fs
        return 0.25 * self.xp.sum(Fs * Fs, axis=(1, 2, 3))

    def grad(self, A, Fs):
        return self.pull(Fs, A)

    def hvp(self, A, Fs, v):
        dF = self.bracket(v, A) + self.bracket(A, v)
        return self.pull(dF, A) + self.pull(Fs, v)

    def gauss(self, A, P):
        return self.xp.einsum('abc,nib,nic->na', self.f, A, P)


# ------------------------------------------------------------------ initial data (host, numpy)
def initial_state(N, E, seed, group):
    """Random (A, P) projected onto the Gauss shell G = 0, then scaled to energy exactly E with the exact
    scaling A -> l A, P -> l^2 P, E -> l^4 E (which keeps G = 0)."""
    rng = np.random.default_rng(seed)
    f = structure_constants(group)
    nc = f.shape[0]
    A = rng.standard_normal((N, 3, nc))
    P = rng.standard_normal((N, 3, nc))
    L = np.einsum('abc,nib->naic', f, A).reshape(N, nc, 3 * nc)     # G = L P
    Pf = P.reshape(N, 3 * nc, 1)
    LLt = L @ np.transpose(L, (0, 2, 1))
    corr = np.transpose(L, (0, 2, 1)) @ np.linalg.solve(LLt, L @ Pf)
    P = (Pf - corr).reshape(N, 3, nc)
    m = Model(np, group)
    E0 = 0.5 * np.sum(P * P, axis=(1, 2)) + m.V(A)
    lam = (E / E0) ** 0.25
    return A * lam[:, None, None], P * (lam ** 2)[:, None, None]


# ------------------------------------------------------------------ evolution
def _yoshida(m, A, P, tan, h):
    for c, d in zip(YC, YD):
        A = A + c * h * P
        for t in tan:
            t[0] = t[0] + c * h * t[1]
        if d:
            Fs = m.F(A)
            P = P - d * h * m.grad(A, Fs)
            Hd = m.hvp(A, Fs, tan[0][0])
            tan[0][1] = tan[0][1] - d * h * Hd
            Hw = m.hvp(A, Fs, tan[1][0])
            tan[1][1] = tan[1][1] - d * h * (tan[1][0] + Hw)
    return A, P


def _advance(m, A, P, tan, dt, odt):
    xp = m.xp
    omega = xp.maximum(xp.max(xp.sqrt(xp.sum(A * A, axis=2)), axis=1), 1.0)
    nsub = xp.maximum(1, xp.ceil(omega * dt / odt)).astype(xp.int32)
    nmax = int(nsub.max())
    if nmax == 1:
        return _yoshida(m, A, P, tan, dt)
    for n in sorted(set(int(k) for k in (xp.unique(nsub).get() if hasattr(xp, 'cuda') else np.unique(nsub)))):
        idx = xp.nonzero(nsub == n)[0]
        a, b = A[idx], P[idx]
        tn = [[t[0][idx], t[1][idx]] for t in tan]
        for _ in range(n):
            a, b = _yoshida(m, a, b, tn, dt / n)
        A[idx], P[idx] = a, b
        for t, s in zip(tan, tn):
            t[0][idx], t[1][idx] = s
    return A, P


def run_chunk(E, N, seed, dt, tmax, t_burn, renorm, odt, backend, t_block, quiet=True, tag='', group='su3',
              init=None):
    """Run one ensemble; returns host arrays: lam_ym, lam_ghost (per trajectory), history, diagnostics."""
    if backend == 'cupy':
        import cupy as xp
    else:
        xp = np
    m = Model(xp, group)
    A0, P0 = initial_state(N, E, seed, group) if init is None else init
    A, P = xp.asarray(A0), xp.asarray(P0)
    rng = np.random.default_rng(seed + 1)
    nothing = [[xp.zeros_like(A), xp.zeros_like(A)], [xp.zeros_like(A), xp.zeros_like(A)]]
    for _ in range(int(round(t_burn / dt))):
        A, P = _advance(m, A, P, nothing, dt, odt)
    tan = []
    for _ in range(2):
        x = rng.standard_normal((N, 3, m.nc)); v = rng.standard_normal((N, 3, m.nc))
        nrm = np.sqrt(np.sum(x * x + v * v, axis=(1, 2)))[:, None, None]
        tan.append([xp.asarray(x / nrm), xp.asarray(v / nrm)])
    H0 = 0.5 * xp.sum(P * P, axis=(1, 2)) + m.V(A)
    G0 = xp.max(xp.abs(m.gauss(A, P)), axis=1)
    logs = [xp.zeros(N), xp.zeros(N)]
    nsteps = int(round(tmax / dt))
    block = max(renorm, int(round(t_block / dt)) // renorm * renorm)
    hist = []
    t0 = time.time()
    for s in range(1, nsteps + 1):
        A, P = _advance(m, A, P, tan, dt, odt)
        if s % renorm == 0:
            for k in range(2):
                x, v = tan[k]
                nrm = xp.sqrt(xp.sum(x * x + v * v, axis=(1, 2)))
                logs[k] += xp.log(nrm)
                tan[k] = [x / nrm[:, None, None], v / nrm[:, None, None]]
        if s % block == 0 or s == nsteps:
            t = s * dt
            good = xp.isfinite(logs[0]) & xp.isfinite(logs[1])
            row = [t] + [float(xp.mean(logs[k][good]) / t) for k in range(2)]
            hist.append(row)
            if not quiet:
                print(f'  {tag}E={E:g} t={t:9.1f}  lam_ym={row[1]:.5f}  lam_ghost={row[2]:.6f}  '
                      f'wall={time.time() - t0:.0f}s', flush=True)
    T = nsteps * dt
    H1 = 0.5 * xp.sum(P * P, axis=(1, 2)) + m.V(A)
    G1 = xp.max(xp.abs(m.gauss(A, P)), axis=1)
    to_host = (lambda z: z.get()) if backend == 'cupy' else (lambda z: z)
    return dict(lam_ym=to_host(logs[0] / T), lam_ghost=to_host(logs[1] / T), hist=np.array(hist),
                dE=to_host(xp.abs(H1 - H0) / E), dG=to_host(xp.abs(G1 - G0)), T=T)


def _chunk_worker(args):
    return run_chunk(*args)


def run_energy(E, a):
    t0 = time.time()
    if a.backend == 'numpy' and a.workers > 1:
        import multiprocessing as mp
        for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
            os.environ[k] = '1'                         # children inherit: one core per worker
        sizes = [a.N // a.workers + (1 if k < a.N % a.workers else 0) for k in range(a.workers)]
        jobs = [(E, n, a.seed + 1000 * k, a.dt, a.tmax, a.t_burn, a.renorm, a.omega_dt_max, 'numpy', a.t_block,
                 a.quiet or k > 0, f'[w{k}] ', a.group) for k, n in enumerate(sizes) if n > 0]
        with mp.get_context('spawn').Pool(len(jobs)) as pool:
            parts = pool.map(_chunk_worker, jobs)
        res = {k: np.concatenate([p[k] for p in parts]) for k in ('lam_ym', 'lam_ghost', 'dE', 'dG')}
        res['hist'] = parts[0]['hist']                  # worker 0's running means (indicative)
    else:
        res = run_chunk(E, a.N, a.seed, a.dt, a.tmax, a.t_burn, a.renorm, a.omega_dt_max, a.backend,
                        a.t_block, a.quiet, '', a.group)
    good = np.isfinite(res['lam_ym']) & np.isfinite(res['lam_ghost']) & np.isfinite(res['dE'])
    n = int(good.sum())
    out = dict(group=a.group, E=E, N=a.N, dropped=a.N - n, tmax=a.tmax, dt=a.dt,
               lam_ym=res['lam_ym'][good].mean(), lam_ym_sem=res['lam_ym'][good].std() / np.sqrt(n),
               lam_ghost=res['lam_ghost'][good].mean(), lam_ghost_sem=res['lam_ghost'][good].std() / np.sqrt(n),
               max_dE=res['dE'][good].max(), max_dG=res['dG'][good].max(), wall_s=time.time() - t0)
    os.makedirs(a.out, exist_ok=True)
    np.savez(os.path.join(a.out, f'{a.group}_lam_E{E:g}.npz'), lam_ym=res['lam_ym'], lam_ghost=res['lam_ghost'],
             dE=res['dE'], dG=res['dG'], hist=res['hist'])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--group', choices=['su2', 'su3', 'so4'], default='su3')
    ap.add_argument('--E', type=float, nargs='+')
    ap.add_argument('--N', type=int, default=2000)
    ap.add_argument('--backend', choices=['numpy', 'cupy'], default='numpy')
    ap.add_argument('--workers', type=int, default=1, help='numpy only: processes (one core each)')
    ap.add_argument('--dt', type=float, default=0.01)
    ap.add_argument('--tmax', type=float, default=1000.0)
    ap.add_argument('--t-burn', type=float, default=100.0)
    ap.add_argument('--t-block', type=float, default=250.0, help='progress / history interval')
    ap.add_argument('--renorm', type=int, default=20)
    ap.add_argument('--omega-dt-max', type=float, default=0.25)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--out', default=os.path.join('results', 'group'))
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()
    if not a.E:
        ap.error('--E is required')
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f'{a.group}.csv')
    new = not os.path.exists(path)
    print(f'# ym_group: group={a.group} backend={a.backend} workers={a.workers} N={a.N} dt={a.dt} tmax={a.tmax} '
          f't_burn={a.t_burn} seed={a.seed} out={a.out}', flush=True)
    with open(path, 'a', newline='') as fh:
        w = None
        for E in a.E:
            row = run_energy(E, a)
            if w is None:
                w = csv.DictWriter(fh, fieldnames=list(row))
                if new:
                    w.writeheader()
            w.writerow(row); fh.flush()
            print(f"{a.group} E={E:g}  lam_ym={row['lam_ym']:.5f} +- {row['lam_ym_sem']:.5f}  "
                  f"lam_ghost={row['lam_ghost']:.6f} +- {row['lam_ghost_sem']:.6f}  "
                  f"dropped={row['dropped']}  max|dE|/E={row['max_dE']:.1e}  max|dG|={row['max_dG']:.1e}  "
                  f"wall={row['wall_s']:.0f}s", flush=True)


if __name__ == '__main__':
    main()
