"""
su3_full.py -- generic homogeneous SU(3) Yang-Mills and its Lee-Wick ghost (steps 1 and 2 of README.md "Open").

Background: full 24-component homogeneous SU(3) YM, A[n, i, a] (n trajectory, i = 1..3 space, a = 1..8 colour),
    H = 1/2 |P|^2 + V(A),  V = 1/4 F^a_ij F^a_ij,  F^a_ij = f^abc A_i^b A_j^c       (units g = M = 1, as hdym)
with initial data on the Gauss shell  G^a = f^abc A_i^b P_i^c = 0  (conserved; monitored).

Two tangent vectors ride along each trajectory, both 24-d:
    ym    : d'' = -Hess V(A) d            -> lambda_YM, the Lyapunov exponent of generic SU(3) YM (scale-free: ~ E^{1/4})
    ghost : w'' = -(1 + Hess V(A)) w      -> lambda_ghost, the linear ghost about generic SU(3) YM (the SU(3) analogue
                                             of the paper's transverse exponent; naive operator, no Lee-Wick Gauss-law
                                             reduction, the level of the paper's 'vec' ghost)
Linearising H = 1/2|P|^2 - 1/2|W|^2 + V(q) + w.gradV(q) - 1/2|w|^2 about w = W = 0 decouples the two exactly.

Integrator: Yoshida-4 for (A, P) and both tangents together; per-trajectory sub-stepping
    nsub = ceil(omega dt / omega_dt_max),  omega = max(1, max_i |A_i|)  (|A_i| = colour norm)
the same rule as hdym_core.ym_step_substepped (omega >= 1 resolves the ghost's unit mass).

Backends (--backend):
    numpy : CPU. With --workers K the ensemble is split into K independent processes (one core each).
    cupy  : GPU, the same array code on the device.

Output (--out DIR): DIR/su3_full.csv (one row per energy), DIR/lam_E<E>.npz (per-trajectory finite-time
exponents and the running-mean history for convergence checks), and progress lines on stdout.

Examples (see README.md for the run plan):
    python su3_full.py --selftest
    python su3_full.py --E 1.0 --N 2000 --tmax 500 --backend numpy --workers 20 --out results/full_smoke
    python su3_full.py --E 1.0 0.5 0.3 --N 100000 --tmax 4000 --backend cupy --out results/full_gpu
"""
import argparse, csv, os, sys, time
import numpy as np

# ------------------------------------------------------------------ structure constants (as su3_core.py)
_F_NONZERO = {(1, 2, 3): 1.0, (1, 4, 7): 0.5, (1, 5, 6): -0.5, (2, 4, 6): 0.5, (2, 5, 7): 0.5,
              (3, 4, 5): 0.5, (3, 6, 7): -0.5, (4, 5, 8): np.sqrt(3) / 2, (6, 7, 8): np.sqrt(3) / 2}


def structure_constants():
    f = np.zeros((8, 8, 8))
    for (a, b, c), v in _F_NONZERO.items():
        a, b, c = a - 1, b - 1, c - 1
        for (i, j, k), s in (((a, b, c), 1), ((b, c, a), 1), ((c, a, b), 1),
                             ((b, a, c), -1), ((a, c, b), -1), ((c, b, a), -1)):
            f[i, j, k] = s * v
    return f


_C = 2.0 ** (1.0 / 3.0)
_W1 = 1.0 / (2.0 - _C)
_W0 = -_C / (2.0 - _C)
YC = (_W1 / 2, (_W0 + _W1) / 2, (_W0 + _W1) / 2, _W1 / 2)
YD = (_W1, _W0, _W1, 0.0)


class Model:
    """All array maths on backend xp (numpy or cupy)."""

    def __init__(self, xp):
        self.xp = xp
        self.f = xp.asarray(structure_constants())

    # Two contractions, written as tensordot + batched matmul (fast on CPU BLAS and on the GPU):
    #   bracket(X, Y)[n,i,j,a] = f^abc X_i^b Y_j^c
    #   pull(Fs, Y)[n,i,b]      = f^abc sum_j Fs_ij^a Y_j^c
    def bracket(self, X, Y):
        xp = self.xp
        n = X.shape[0]
        T = xp.tensordot(X, self.f, axes=([2], [1]))                 # (n, i, a, c)
        out = xp.matmul(T.reshape(n, 24, 8), xp.transpose(Y, (0, 2, 1)))   # (n, i*a, j)
        return xp.transpose(out.reshape(n, 3, 8, 3), (0, 1, 3, 2))   # (n, i, j, a)

    def pull(self, Fs, Y):
        xp = self.xp
        n = Fs.shape[0]
        S = xp.matmul(xp.transpose(Fs, (0, 1, 3, 2)).reshape(n, 24, 3), Y)  # (n, i*a, c)
        return xp.tensordot(S.reshape(n, 3, 8, 8), self.f, axes=([2, 3], [0, 2]))  # (n, i, b)

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
def initial_state(N, E, seed):
    """Random (A, P) projected onto the Gauss shell G = 0, then scaled to energy exactly E with the exact
    scaling A -> l A, P -> l^2 P, E -> l^4 E (which keeps G = 0)."""
    rng = np.random.default_rng(seed)
    f = structure_constants()
    A = rng.standard_normal((N, 3, 8))
    P = rng.standard_normal((N, 3, 8))
    L = np.einsum('abc,nib->naic', f, A).reshape(N, 8, 24)          # G = L P
    Pf = P.reshape(N, 24, 1)
    LLt = L @ np.transpose(L, (0, 2, 1))
    corr = np.transpose(L, (0, 2, 1)) @ np.linalg.solve(LLt, L @ Pf)
    P = (Pf - corr).reshape(N, 3, 8)
    m = Model(np)
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


def run_chunk(E, N, seed, dt, tmax, t_burn, renorm, odt, backend, t_block, quiet=True, tag=''):
    """Run one ensemble; returns host arrays: lam_ym, lam_ghost (per trajectory), history, diagnostics."""
    if backend == 'cupy':
        import cupy as xp
    else:
        xp = np
    m = Model(xp)
    A0, P0 = initial_state(N, E, seed)
    A, P = xp.asarray(A0), xp.asarray(P0)
    rng = np.random.default_rng(seed + 1)
    nothing = [[xp.zeros_like(A), xp.zeros_like(A)], [xp.zeros_like(A), xp.zeros_like(A)]]
    for _ in range(int(round(t_burn / dt))):
        A, P = _advance(m, A, P, nothing, dt, odt)
    tan = []
    for _ in range(2):
        x = rng.standard_normal((N, 3, 8)); v = rng.standard_normal((N, 3, 8))
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
                 a.quiet or k > 0, f'[w{k}] ') for k, n in enumerate(sizes) if n > 0]
        with mp.get_context('spawn').Pool(len(jobs)) as pool:
            parts = pool.map(_chunk_worker, jobs)
        res = {k: np.concatenate([p[k] for p in parts]) for k in ('lam_ym', 'lam_ghost', 'dE', 'dG')}
        res['hist'] = parts[0]['hist']                  # worker 0's running means (indicative)
    else:
        res = run_chunk(E, a.N, a.seed, a.dt, a.tmax, a.t_burn, a.renorm, a.omega_dt_max, a.backend,
                        a.t_block, a.quiet)
    good = np.isfinite(res['lam_ym']) & np.isfinite(res['lam_ghost']) & np.isfinite(res['dE'])
    n = int(good.sum())
    out = dict(E=E, N=a.N, dropped=a.N - n, tmax=a.tmax, dt=a.dt,
               lam_ym=res['lam_ym'][good].mean(), lam_ym_sem=res['lam_ym'][good].std() / np.sqrt(n),
               lam_ghost=res['lam_ghost'][good].mean(), lam_ghost_sem=res['lam_ghost'][good].std() / np.sqrt(n),
               max_dE=res['dE'][good].max(), max_dG=res['dG'][good].max(), wall_s=time.time() - t0)
    os.makedirs(a.out, exist_ok=True)
    np.savez(os.path.join(a.out, f'lam_E{E:g}.npz'), lam_ym=res['lam_ym'], lam_ghost=res['lam_ghost'],
             dE=res['dE'], dG=res['dG'], hist=res['hist'])
    return out


# ------------------------------------------------------------------ self-test (no GPU needed)
def selftest():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import su3_core as s3
    ok = True

    def rep(name, err, tol):
        nonlocal ok
        ok &= bool(err < tol)
        print(f"[{'OK ' if err < tol else 'FAIL'}] {name}: {err:.2e}")

    m = Model(np)
    rng = np.random.default_rng(0)
    A = rng.standard_normal((50, 3, 8)); v = rng.standard_normal((50, 3, 8))
    Fs = m.F(A)
    rep('V == su3_core.V', np.abs(m.V(A, Fs) - s3.V(A)).max(), 1e-12)
    rep('grad == su3_core.gradV', np.abs(m.grad(A, Fs) - s3.gradV(A)).max(), 1e-12)
    rep('hvp == su3_core.hessV_dot', np.abs(m.hvp(A, Fs, v) - s3.hessV_dot(A, v)).max(), 1e-12)
    A0, P0 = initial_state(200, 0.7, 1)
    rep('initial energy == E', np.abs(0.5 * np.sum(P0 * P0, axis=(1, 2)) + m.V(A0) - 0.7).max(), 1e-12)
    rep('initial Gauss G = 0', np.abs(m.gauss(A0, P0)).max(), 1e-12)
    r = run_chunk(0.7, 64, 3, 0.005, 20.0, 0.0, 20, 0.25, 'numpy', 10.0)
    rep('energy drift over t = 20 (rel)', r['dE'].max(), 1e-4)
    rep('Gauss drift over t = 20', r['dG'].max(), 1e-10)
    # embedded SU(2) sector: G = 0 identically, and the sector is invariant under the 24-d flow
    q = rng.standard_normal((8, 3)); Pq = rng.standard_normal((8, 3))
    A = s3.embed(q); P = s3.embed(Pq)
    nothing = [[np.zeros_like(A), np.zeros_like(A)], [np.zeros_like(A), np.zeros_like(A)]]
    for _ in range(400):
        A, P = _advance(m, A, P, nothing, 0.005, 0.25)
    off = A.copy()
    for i in range(3):
        off[:, i, i] = 0
    rep('SU(2) sector invariant under the 24-d flow', np.abs(off).max(), 1e-12)
    print('\nall checks passed' if ok else '\nSOME CHECKS FAILED')
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--selftest', action='store_true', help='run the checks and exit')
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
    ap.add_argument('--out', default=os.path.join('results', 'full'))
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not a.E:
        ap.error('--E is required (or --selftest)')
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, 'su3_full.csv')
    new = not os.path.exists(path)
    print(f'# su3_full: backend={a.backend} workers={a.workers} N={a.N} dt={a.dt} tmax={a.tmax} '
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
            print(f"E={E:g}  lam_ym={row['lam_ym']:.5f} +- {row['lam_ym_sem']:.5f}  "
                  f"lam_ghost={row['lam_ghost']:.6f} +- {row['lam_ghost_sem']:.6f}  "
                  f"dropped={row['dropped']}  max|dE|/E={row['max_dE']:.1e}  max|dG|={row['max_dG']:.1e}  "
                  f"wall={row['wall_s']:.0f}s", flush=True)


if __name__ == '__main__':
    main()
