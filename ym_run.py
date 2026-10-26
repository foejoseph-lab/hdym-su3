"""
ym_run.py -- the long-run engine: su3_full.py's physics with checkpoint/resume, an offset-free estimator,
a convergence stop, and START/END marker lines in the log.

Physics: identical to su3_full.py (which stays the committed record of the first scan). Generic homogeneous
YM for the chosen group (--group su3: 24 components; su2: 9, generic SU(2); so4: 18), structure constants from
groups.py, on the Gauss shell, with two tangents per trajectory:
    ym    : d'' = -Hess V(A) d            -> lambda_YM
    ghost : w'' = -(1 + Hess V(A)) w      -> lambda_ghost  (naive operator, no Lee-Wick Gauss-law reduction)
Same initial data, Yoshida-4 integrator, sub-stepping rule, renormalisation and per-worker seeds as
su3_full.py. --L0 also projects the initial momenta onto zero spatial angular momentum L_i = eps_ijk A_j^a P_k^a
(conserved). Each trajectory's exponents depend strongly on |L| (results/L_correlation.txt), so ensembles with
and without --L0 are different ensembles; for su2, L = 0 with G = 0 is exactly hdym's colour-diagonal sector up
to a fixed rotation, which makes --L0 the like-for-like setting for comparisons with hdym. For --group su3 the per-trajectory cumulative logs are bit-identical to su3_full.run_chunk
(checked by --selftest), however the run is cut into segments and resumed.

How a run proceeds (per energy):
  * The time axis is cut into segments of --t-seg. Each segment every worker loads its checkpoint
    (DIR/ckpt/E<E>_w<k>.npz), advances, and writes it back atomically. A kill at any moment loses at most
    one segment; rerun the same command with --resume and it continues from the last checkpoint.
  * After each segment the master records every trajectory's cumulative log L_i(t) and prints
        lam_T   = <L(t)> / t                       (su3_full's estimator; carries a start-up offset b/t)
        lam_inc = <L(t) - L(t/2)> / (t/2)          (second-half increment: offset-free) and its SEM
        b       = intercept of a straight-line fit of <L> vs t over the second half
  * Convergence stop (ghost, unless --no-stop): lam_inc * t >= --min-lamT  and  SEM(lam_inc)/lam_inc <=
    --target-rel, at two consecutive segment ends. Otherwise the run stops at --tmax.
  * --tmax can be raised on a later --resume to lengthen a finished energy; a converged energy is skipped
    on --resume unless --no-stop is given.

Outputs (--out DIR):
    DIR/ym_run.csv          one row per finished energy (appended; run id in each row)
    DIR/lam_E<E>.npz        per-trajectory lam_T, lam_inc, drifts, and the full L_i(t) history
    DIR/ckpt/               checkpoints (git-ignored: working state, not results)
    stdout                  '# START ym_run run=...' first, one line per segment, '# END ym_run run=...
                            status=OK|FAIL|INTERRUPTED wall=...s' last. No END line means the process died.

Examples:
    python ym_run.py --selftest
    python ym_run.py --group su3 --E 0.3 0.1 --N 1100 --workers 22 --tmax 200000 --out results/run_low
    python ym_run.py --group su3 --E 0.3 0.1 --N 1100 --workers 22 --tmax 200000 --out results/run_low --resume
"""
import argparse, csv, json, os, sys, time, traceback
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from groups import structure_constants, GROUPS          # su2, su3, so4 in one convention


_C = 2.0 ** (1.0 / 3.0)
_W1 = 1.0 / (2.0 - _C)
_W0 = -_C / (2.0 - _C)
YC = (_W1 / 2, (_W0 + _W1) / 2, (_W0 + _W1) / 2, _W1 / 2)
YD = (_W1, _W0, _W1, 0.0)


class Model:
    """su3_full.Model with the colour dimension d taken from f (same operations, same order)."""

    def __init__(self, f):
        self.f = f
        self.d = f.shape[0]

    def bracket(self, X, Y):
        n, d = X.shape[0], self.d
        T = np.tensordot(X, self.f, axes=([2], [1]))
        out = np.matmul(T.reshape(n, 3 * d, d), np.transpose(Y, (0, 2, 1)))
        return np.transpose(out.reshape(n, 3, d, 3), (0, 1, 3, 2))

    def pull(self, Fs, Y):
        n, d = Fs.shape[0], self.d
        S = np.matmul(np.transpose(Fs, (0, 1, 3, 2)).reshape(n, 3 * d, 3), Y)
        return np.tensordot(S.reshape(n, 3, d, d), self.f, axes=([2, 3], [0, 2]))

    def F(self, A):
        return self.bracket(A, A)

    def V(self, A, Fs=None):
        Fs = self.F(A) if Fs is None else Fs
        return 0.25 * np.sum(Fs * Fs, axis=(1, 2, 3))

    def grad(self, A, Fs):
        return self.pull(Fs, A)

    def hvp(self, A, Fs, v):
        dF = self.bracket(v, A) + self.bracket(A, v)
        return self.pull(dF, A) + self.pull(Fs, v)

    def gauss(self, A, P):
        return np.einsum('abc,nib,nic->na', self.f, A, P)


EPS3 = np.zeros((3, 3, 3))
EPS3[0, 1, 2] = EPS3[1, 2, 0] = EPS3[2, 0, 1] = 1.0
EPS3[0, 2, 1] = EPS3[2, 1, 0] = EPS3[1, 0, 2] = -1.0


def angmom(A, P):
    """Spatial angular momentum L_i = eps_ijk A_j^a P_k^a (conserved: V is rotation invariant), shape (N, 3)."""
    return np.einsum('ijk,nja,nka->ni', EPS3, A, P)


def initial_state(f, N, E, seed, L0=False):
    """As su3_full.initial_state: random (A, P) projected onto G = 0, scaled exactly to energy E.
    With L0, P is projected onto G = 0 and L = 0 together (both linear in P; the scaling keeps both zero).
    Without it the arithmetic is exactly su3_full's (bit-identical, checked by --selftest)."""
    d = f.shape[0]
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((N, 3, d))
    P = rng.standard_normal((N, 3, d))
    L = np.einsum('abc,nib->naic', f, A).reshape(N, d, 3 * d)
    if L0:                                         # add the three rows dL_i/dP_k^a = eps_ijk A_j^a
        L = np.concatenate([L, np.einsum('ijk,nja->nika', EPS3, A).reshape(N, 3, 3 * d)], axis=1)
    Pf = P.reshape(N, 3 * d, 1)
    LLt = L @ np.transpose(L, (0, 2, 1))
    corr = np.transpose(L, (0, 2, 1)) @ np.linalg.solve(LLt, L @ Pf)
    P = (Pf - corr).reshape(N, 3, d)
    m = Model(f)
    E0 = 0.5 * np.sum(P * P, axis=(1, 2)) + m.V(A)
    lam = (E / E0) ** 0.25
    return A * lam[:, None, None], P * (lam ** 2)[:, None, None]


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
    omega = np.maximum(np.max(np.sqrt(np.sum(A * A, axis=2)), axis=1), 1.0)
    nsub = np.maximum(1, np.ceil(omega * dt / odt)).astype(np.int32)
    if int(nsub.max()) == 1:
        return _yoshida(m, A, P, tan, dt)
    for n in sorted(set(int(k) for k in np.unique(nsub))):
        idx = np.nonzero(nsub == n)[0]
        a, b = A[idx], P[idx]
        tn = [[t[0][idx], t[1][idx]] for t in tan]
        for _ in range(n):
            a, b = _yoshida(m, a, b, tn, dt / n)
        A[idx], P[idx] = a, b
        for t, s in zip(tan, tn):
            t[0][idx], t[1][idx] = s
    return A, P


# ------------------------------------------------------------------ checkpoint I/O
def _atomic_savez(path, **arrs):
    """Write to a temp file, then os.replace (atomic). Retries cover Windows briefly locking the target."""
    tmp = path + '.tmp.npz'
    np.savez(tmp, **arrs)
    for k in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.5 * (k + 1))
    os.replace(tmp, path)


def _load(path):
    """Read every array of an .npz into memory and close the file. A bare np.load keeps the file open,
    and Windows then refuses to replace it (PermissionError in os.replace)."""
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _held_open(path):
    """True if this process still holds `path` open (Linux: /proc; elsewhere: try to move it)."""
    target = os.path.realpath(path)
    if os.path.isdir('/proc/self/fd'):
        for fd in os.listdir('/proc/self/fd'):
            try:
                if os.path.realpath(os.readlink(f'/proc/self/fd/{fd}')) == target:
                    return True
            except OSError:
                pass
        return False
    probe = path + '.probe'
    try:
        os.replace(path, probe)
    except PermissionError:
        return True
    os.replace(probe, path)
    return False


def _params_key(p):
    return json.dumps(p, sort_keys=True)


# ------------------------------------------------------------------ one worker, one segment
def segment(job):
    """Advance one worker's ensemble from its checkpoint (or from scratch) to step job['s_end']."""
    f = structure_constants(job['group'])
    m = Model(f)
    p = job['params']
    E, N, seed, dt, odt, renorm = p['E'], p['N'], p['seed'], p['dt'], p['odt'], p['renorm']
    path = job['ckpt']
    if os.path.exists(path):
        z = _load(path)
        if str(z['params']) != _params_key(p):
            raise RuntimeError(f'checkpoint {path} was made with different parameters:\n  {z["params"]}\n'
                               f'  now {_params_key(p)}')
        A, P = z['A'], z['P']
        tan = [[z['x0'], z['v0']], [z['x1'], z['v1']]]
        logs = [z['L0'], z['L1']]
        s0, H0, G0 = int(z['s']), z['H0'], z['G0']
        G0v = z['G0v'] if 'G0v' in z else None     # checkpoints from before 26 Oct have only max|G0|
    else:
        A, P = initial_state(f, N, E, seed, p.get('L0', False))
        rng = np.random.default_rng(seed + 1)
        nothing = [[np.zeros_like(A), np.zeros_like(A)], [np.zeros_like(A), np.zeros_like(A)]]
        for _ in range(int(round(p['t_burn'] / dt))):
            A, P = _advance(m, A, P, nothing, dt, odt)
        tan = []
        for _ in range(2):
            x = rng.standard_normal(A.shape); v = rng.standard_normal(A.shape)
            nrm = np.sqrt(np.sum(x * x + v * v, axis=(1, 2)))[:, None, None]
            tan.append([x / nrm, v / nrm])
        H0 = 0.5 * np.sum(P * P, axis=(1, 2)) + m.V(A)
        G0v = m.gauss(A, P)                         # full signed Gauss vector, (N, dim G)
        G0 = np.max(np.abs(G0v), axis=1)
        logs = [np.zeros(N), np.zeros(N)]
        s0 = 0
    for s in range(s0 + 1, job['s_end'] + 1):
        A, P = _advance(m, A, P, tan, dt, odt)
        if s % renorm == 0:
            for k in range(2):
                x, v = tan[k]
                nrm = np.sqrt(np.sum(x * x + v * v, axis=(1, 2)))
                logs[k] += np.log(nrm)
                tan[k] = [x / nrm[:, None, None], v / nrm[:, None, None]]
    s1 = max(s0, job['s_end'])
    _atomic_savez(path, A=A, P=P, x0=tan[0][0], v0=tan[0][1], x1=tan[1][0], v1=tan[1][1], L0=logs[0], L1=logs[1],
                  s=np.int64(s1), H0=H0, G0=G0, params=np.array(_params_key(p)),
                  **({} if G0v is None else {'G0v': G0v}))
    H1 = 0.5 * np.sum(P * P, axis=(1, 2)) + m.V(A)
    G1 = np.max(np.abs(m.gauss(A, P)), axis=1)
    G1v = m.gauss(A, P)
    Lv = angmom(A, P)
    # signed values kept alongside the magnitudes: dE_signed = (H(t) - H0)/E, dG_vec = G(t) - G0 per colour,
    # L = the conserved angular-momentum vector (N, 3)
    return dict(s=s1, L0=logs[0].copy(), L1=logs[1].copy(), dE=np.abs(H1 - H0) / E, dG=np.abs(G1 - G0),
                Lmag=np.linalg.norm(Lv, axis=1), L=Lv, dE_signed=(H1 - H0) / E,
                dG_vec=(G1v - G0v) if G0v is not None else np.full_like(G1v, np.nan))


# ------------------------------------------------------------------ estimators
def estimates(t_hist, L_hist, good):
    """From cumulative logs L_hist[k, i] at times t_hist[k] (t_hist[0] = 0): lam_T, lam_inc (+ SEM), b."""
    t = t_hist[-1]
    L = L_hist[-1, good]
    out = dict(t=t, lam_T=float(L.mean() / t), n=int(good.sum()))
    later = np.nonzero((t_hist >= t / 2) & (t_hist < t))[0]
    if len(later) == 0:
        out.update(lam_inc=np.nan, sem_inc=np.nan, b=np.nan)
        return out
    h = later[0]
    inc = (L - L_hist[h, good]) / (t - t_hist[h])
    out.update(lam_inc=float(inc.mean()), sem_inc=float(inc.std() / np.sqrt(inc.size)), t_half=float(t_hist[h]))
    sel = t_hist >= t_hist[h]
    if sel.sum() >= 3:
        mL = L_hist[sel][:, good].mean(axis=1)
        (_, b), *_ = np.linalg.lstsq(np.vstack([t_hist[sel], np.ones(sel.sum())]).T, mL, rcond=None)
        out['b'] = float(b)
    else:
        out['b'] = float(L.mean() - out['lam_inc'] * t)
    return out


# ------------------------------------------------------------------ one energy
def run_energy(E, a, pool, run_id):
    t0 = time.time()
    dt = a.dt
    if abs(a.t_seg / dt - round(a.t_seg / dt)) > 1e-9 or round(a.t_seg / dt) % a.renorm:
        raise ValueError('--t-seg must be a whole number of renorm intervals (renorm * dt)')
    seg_steps = int(round(a.t_seg / dt))
    s_max = int(round(a.tmax / dt)) // a.renorm * a.renorm
    sizes = [a.N // a.workers + (1 if k < a.N % a.workers else 0) for k in range(a.workers)]
    sizes = [n for n in sizes if n > 0]
    ck = os.path.join(a.out, 'ckpt')
    os.makedirs(ck, exist_ok=True)
    tag = f'E{E:g}'
    mpath = os.path.join(ck, f'{tag}_master.npz')
    base = dict(group=a.group, E=E, dt=dt, odt=a.omega_dt_max, renorm=a.renorm, t_burn=a.t_burn)
    if getattr(a, 'L0', False):
        base['L0'] = True            # only when set, so checkpoints made before --L0 existed still resume
    jobs = [dict(group=a.group, ckpt=os.path.join(ck, f'{tag}_w{k:02d}.npz'),
                 params=dict(base, N=n, seed=a.seed + 1000 * k)) for k, n in enumerate(sizes)]
    master_key = _params_key(dict(base, N=a.N, workers=len(sizes), seed=a.seed))

    exists = os.path.exists(mpath) or any(os.path.exists(j['ckpt']) for j in jobs)
    if exists and not a.resume:
        raise RuntimeError(f'checkpoints for E={E:g} already exist in {ck}; add --resume to continue them, '
                           f'or choose a new --out')
    if os.path.exists(mpath):
        z = _load(mpath)
        if str(z['params']) != master_key:
            raise RuntimeError(f'{mpath} was made with different parameters:\n  {z["params"]}\n  now {master_key}')
        t_hist, L0h, L1h = list(z['t']), list(z['L0']), list(z['L1'])
        reason = str(z['reason'])
        streak = int(z['streak'])
        if reason == 'converged' and not a.no_stop:
            print(f'E={E:g}: already converged at t={t_hist[-1]:g} (resume with --no-stop to go further)',
                  flush=True)
            return None
        print(f'E={E:g}: resuming at t={t_hist[-1]:g}', flush=True)
    else:
        t_hist, L0h, L1h = [0.0], [np.zeros(a.N)], [np.zeros(a.N)]
        streak = 0
    s = int(round(t_hist[-1] / dt))
    if os.path.exists(mpath) and any(not os.path.exists(j['ckpt']) for j in jobs):
        raise RuntimeError(f'master checkpoint exists but a worker checkpoint is missing in {ck}')

    reason = 'tmax'
    est0 = est1 = None
    dE = dG = None
    while s < s_max:
        s_end = min(s + seg_steps, s_max)
        for j in jobs:
            j['s_end'] = s_end
        parts = pool.map(segment, jobs) if pool is not None else [segment(j) for j in jobs]
        if any(p['s'] != s_end for p in parts):
            raise RuntimeError(f'worker checkpoints disagree with the master (expected step {s_end}): '
                               f'{[p["s"] for p in parts]}')
        s = s_end
        t_hist.append(s * dt)
        L0h.append(np.concatenate([p['L0'] for p in parts]))
        L1h.append(np.concatenate([p['L1'] for p in parts]))
        dE = np.concatenate([p['dE'] for p in parts]); dG = np.concatenate([p['dG'] for p in parts])
        Lmag = np.concatenate([p['Lmag'] for p in parts])
        Lvec = np.concatenate([p['L'] for p in parts])
        dE_s = np.concatenate([p['dE_signed'] for p in parts]); dG_v = np.concatenate([p['dG_vec'] for p in parts])
        th = np.array(t_hist); L0a = np.array(L0h); L1a = np.array(L1h)
        good = np.isfinite(L0a[-1]) & np.isfinite(L1a[-1]) & np.isfinite(dE)
        est0, est1 = estimates(th, L0a, good), estimates(th, L1a, good)
        li, se = est1['lam_inc'], est1['sem_inc']
        ok_now = (np.isfinite(li) and li > 0 and li * s * dt >= a.min_lamT and se / li <= a.target_rel)
        streak = streak + 1 if ok_now else 0
        stop = (streak >= 2) and not a.no_stop
        if stop:
            reason = 'converged'
        _atomic_savez(mpath, t=th, L0=L0a, L1=L1a, params=np.array(master_key),
                      reason=np.array(reason if stop or s >= s_max else 'running'), streak=np.int64(streak))
        rel = se / li if np.isfinite(li) and li else np.nan
        print(f'  E={E:g} t={s * dt:10.0f}  ym: T={est0["lam_T"]:.5f} inc={est0["lam_inc"]:.5f}  '
              f'ghost: T={est1["lam_T"]:.7f} inc={li:.7f} +- {se:.7f} ({rel:.2%})  b={est1["b"]:+.3f}  '
              f'lamT={li * s * dt:6.1f}  max|dE|/E={dE[good].max():.1e}  wall={time.time() - t0:.0f}s',
              flush=True)
        if stop:
            break

    if est1 is None:
        print(f'E={E:g}: nothing to do (t={t_hist[-1]:g} >= tmax={a.tmax:g}; raise --tmax to lengthen)', flush=True)
        return None
    th = np.array(t_hist); L0a = np.array(L0h); L1a = np.array(L1h)
    good = np.isfinite(L0a[-1]) & np.isfinite(L1a[-1]) & np.isfinite(dE)
    T = th[-1]
    h = np.nonzero(th >= est1['t_half'])[0][0] if 't_half' in est1 else 0   # one segment only: inc = lam_T
    np.savez(os.path.join(a.out, f'lam_E{E:g}.npz'),
             lam_ym=L0a[-1] / T, lam_ghost=L1a[-1] / T,
             inc_ym=(L0a[-1] - L0a[h]) / (T - th[h]), inc_ghost=(L1a[-1] - L1a[h]) / (T - th[h]),
             dE=dE, dG=dG, t=th, L_ym=L0a, L_ghost=L1a, Lmag=Lmag, ell=Lmag / E ** 0.75,
             L=Lvec, dE_signed=dE_s, dG_vec=dG_v)
    return dict(run=run_id, group=a.group, E=E, N=a.N, dropped=a.N - int(good.sum()), t=T, dt=dt, reason=reason,
                lam_ym_T=est0['lam_T'], lam_ym_inc=est0['lam_inc'], lam_ym_inc_sem=est0['sem_inc'],
                lam_ghost_T=est1['lam_T'], lam_ghost_inc=est1['lam_inc'], lam_ghost_inc_sem=est1['sem_inc'],
                b_ghost=est1['b'], lamT_ghost=est1['lam_inc'] * T,
                max_dE=float(dE[good].max()), max_dG=float(dG[good].max()), L0=bool(getattr(a, 'L0', False)),
                max_L=float(Lmag[good].max()), wall_s=time.time() - t0)


# ------------------------------------------------------------------ self-test
def selftest():
    import shutil, tempfile
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import su3_full as sf
    ok = True

    def rep(name, err, tol):
        nonlocal ok
        good = bool(err <= tol)
        ok &= good
        print(f"[{'OK ' if good else 'FAIL'}] {name}: {err:.2e}", flush=True)

    f3 = structure_constants('su3')
    rep('su3 structure constants == su3_full', np.abs(f3 - sf.structure_constants()).max(), 0)
    m, ms = Model(f3), sf.Model(np)
    rng = np.random.default_rng(0)
    A = rng.standard_normal((50, 3, 8)); v = rng.standard_normal((50, 3, 8))
    Fs = m.F(A)
    rep('su3 V, grad, hvp bit-identical to su3_full.Model',
        max(np.abs(m.V(A, Fs) - ms.V(A)).max(), np.abs(m.grad(A, Fs) - ms.grad(A, ms.F(A))).max(),
            np.abs(m.hvp(A, Fs, v) - ms.hvp(A, ms.F(A), v)).max()), 0)
    A1, P1 = initial_state(f3, 40, 0.7, 5)
    A2, P2 = sf.initial_state(40, 0.7, 5)
    rep('su3 initial data bit-identical to su3_full', max(np.abs(A1 - A2).max(), np.abs(P1 - P2).max()), 0)

    # segmented + killed + resumed run vs one uninterrupted su3_full.run_chunk
    tmp = tempfile.mkdtemp(prefix='ym_run_selftest_')
    try:
        E, N, seed, dt, renorm, tb, odt = 0.7, 12, 3, 0.01, 20, 1.0, 0.25
        ref = sf.run_chunk(E, N, seed, dt, 12.0, tb, renorm, odt, 'numpy', 4.0)
        job = dict(group='su3', ckpt=os.path.join(tmp, 'w.npz'),
                   params=dict(group='su3', E=E, N=N, seed=seed, dt=dt, odt=odt, renorm=renorm, t_burn=tb))
        for s_end in (400, 800, 1200):          # three segments, a fresh load from disk each time
            job['s_end'] = s_end
            r = segment(job)
        T = 12.0
        rep('segmented/resumed ghost logs bit-identical to su3_full', np.abs(r['L1'] / T - ref['lam_ghost']).max(), 0)
        rep('segmented/resumed YM logs bit-identical to su3_full', np.abs(r['L0'] / T - ref['lam_ym']).max(), 0)
        job['s_end'] = 1200
        r2 = segment(job)
        rep('re-running a finished segment changes nothing', np.abs(r2['L1'] - r['L1']).max(), 0)
        rep('worker checkpoint is not left open after a segment (Windows replace)',
            float(_held_open(job['ckpt'])), 0)
        bad = dict(job, params=dict(job['params'], dt=0.02))
        try:
            segment(bad); refused = 1.0
        except RuntimeError:
            refused = 0.0
        rep('checkpoint with different parameters is refused', refused, 0)

        # full driver: run to t=20, then 'raise tmax' to 40 with --resume, vs straight to 40
        def args(out, tmax, resume, workers=2):
            return argparse.Namespace(group='su3', N=10, workers=workers, dt=0.01, tmax=tmax, t_burn=1.0, t_seg=4.0,
                                      renorm=20, omega_dt_max=0.25, seed=7, out=out, resume=resume, no_stop=True,
                                      min_lamT=20.0, target_rel=0.01)
        oa, ob = os.path.join(tmp, 'a'), os.path.join(tmp, 'b')
        run_energy(0.5, args(oa, 20.0, False), None, 'x')
        ra = run_energy(0.5, args(oa, 40.0, True), None, 'x')
        rb = run_energy(0.5, args(ob, 40.0, False), None, 'x')
        za, zb = _load(os.path.join(oa, 'lam_E0.5.npz')), _load(os.path.join(ob, 'lam_E0.5.npz'))
        rep('driver: tmax 20 then --resume to 40 == straight to 40',
            max(np.abs(za['L_ghost'] - zb['L_ghost']).max(), np.abs(za['L_ym'] - zb['L_ym']).max()), 0)
        rep('driver: lam_ghost_T matches', abs(ra['lam_ghost_T'] - rb['lam_ghost_T']), 0)
        rep('master checkpoint is not left open after a resume (Windows replace)',
            float(_held_open(os.path.join(oa, 'ckpt', 'E0.5_master.npz'))), 0)
        try:
            run_energy(0.5, args(ob, 40.0, False), None, 'x'); refused = 1.0
        except RuntimeError:
            refused = 0.0
        rep('existing checkpoints without --resume are refused', refused, 0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # estimator: exact L = lam t + b recovers lam and b
    th = np.arange(0, 11) * 100.0
    Lh = 0.003 * th[:, None] + 0.4 + np.zeros((1, 5))
    Lh[0] = 0
    e = estimates(th, Lh, np.ones(5, bool))
    rep('estimator: lam_inc on L = lam t + b', abs(e['lam_inc'] - 0.003), 1e-15)
    rep('estimator: intercept b', abs(e['b'] - 0.4), 1e-12)
    rep('estimator: lam_T carries b/t', abs(e['lam_T'] - (0.003 + 0.4 / 1000)), 1e-15)

    # su2: generic 9-d SU(2); colour-diagonal point reproduces hdym's V = 1/2 (x^2y^2 + y^2z^2 + z^2x^2)
    f2 = structure_constants('su2')
    m2 = Model(f2)
    q = rng.standard_normal((20, 3))
    Ad = np.zeros((20, 3, 3))
    for i in range(3):
        Ad[:, i, i] = q[:, i]
    x, y, z = q.T
    rep('su2: V on the diagonal == hdym V', np.abs(m2.V(Ad) - 0.5 * (x * x * y * y + y * y * z * z + z * z * x * x)).max(),
        1e-13)
    A = rng.standard_normal((20, 3, 3)); v = rng.standard_normal((20, 3, 3)); eps = 1e-6
    fd = (m2.V(A + eps * v) - m2.V(A - eps * v)) / (2 * eps)
    rep('su2: grad vs finite difference', np.abs(np.sum(m2.grad(A, m2.F(A)) * v, axis=(1, 2)) - fd).max(), 1e-7)
    fdh = (m2.grad(A + eps * v, m2.F(A + eps * v)) - m2.grad(A - eps * v, m2.F(A - eps * v))) / (2 * eps)
    rep('su2: hvp vs finite difference', np.abs(m2.hvp(A, m2.F(A), v) - fdh).max(), 1e-6)
    A0, P0 = initial_state(f2, 100, 0.4, 2)
    rep('su2: initial energy == E', np.abs(0.5 * np.sum(P0 * P0, axis=(1, 2)) + m2.V(A0) - 0.4).max(), 1e-12)
    rep('su2: initial Gauss G = 0', np.abs(m2.gauss(A0, P0)).max(), 1e-12)
    tmp = tempfile.mkdtemp(prefix='ym_run_selftest_')
    try:
        job = dict(group='su2', ckpt=os.path.join(tmp, 'w.npz'), s_end=2000,
                   params=dict(group='su2', E=0.4, N=16, seed=1, dt=0.01, odt=0.25, renorm=20, t_burn=1.0))
        r = segment(job)
        rep('su2: energy drift over t = 20 (rel)', r['dE'].max(), 1e-4)
        rep('su2: Gauss drift over t = 20', r['dG'].max(), 1e-10)
        job = dict(group='so4', ckpt=os.path.join(tmp, 'w4.npz'), s_end=2000,
                   params=dict(group='so4', E=0.4, N=16, seed=1, dt=0.01, odt=0.25, renorm=20, t_burn=1.0))
        r = segment(job)
        rep('so4: energy drift over t = 20 (rel)', r['dE'].max(), 1e-4)
        rep('so4: Gauss drift over t = 20', r['dG'].max(), 1e-10)
        for g in ('su3', 'su2'):
            job = dict(group=g, ckpt=os.path.join(tmp, f'L{g}.npz'), s_end=2000,
                       params=dict(group=g, E=0.4, N=16, seed=1, dt=0.01, odt=0.25, renorm=20, t_burn=1.0, L0=True))
            r = segment(job)
            rep(f'{g} --L0: |L| after t = 21 (projected to 0, conserved)', r['Lmag'].max(), 1e-10)
            rep(f'{g} --L0: energy drift over t = 20 (rel)', r['dE'].max(), 1e-4)
            rep(f'{g} --L0: Gauss drift over t = 20', r['dG'].max(), 1e-10)
        job = dict(group='su3', ckpt=os.path.join(tmp, 'sg.npz'), s_end=1000,
                   params=dict(group='su3', E=0.4, N=16, seed=2, dt=0.01, odt=0.25, renorm=20, t_burn=1.0))
        r1 = segment(job)
        job['s_end'] = 2000
        r2 = segment(job)                        # second segment: G0v read back from the checkpoint
        A0, P0 = initial_state(f3, 16, 0.4, 2)
        rep('signed outputs: |dE_signed| == dE, L vector == initial L (conserved), dG_vec finite and small',
            max(np.abs(np.abs(r2['dE_signed']) - r2['dE']).max(), np.abs(r2['L'] - angmom(A0, P0)).max(),
                np.abs(r2['dG_vec']).max() if np.isfinite(r2['dG_vec']).all() else 1.0), 1e-10)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for g, f in (('su3', f3), ('su2', f2)):
        A0, P0 = initial_state(f, 100, 0.4, 3, L0=True)
        mm = Model(f)
        rep(f'{g} --L0 initial data: |L|, |G|, |H - E|',
            max(np.abs(angmom(A0, P0)).max(), np.abs(mm.gauss(A0, P0)).max(),
                np.abs(0.5 * np.sum(P0 * P0, axis=(1, 2)) + mm.V(A0) - 0.4).max()), 1e-12)
    A0, P0 = initial_state(f3, 100, 0.4, 3)
    rep('without --L0 the initial |L| is generic (not projected): min |L|', -np.linalg.norm(angmom(A0, P0), axis=1).min(), -1e-3)
    # su2 with G = 0 and L = 0 is the colour-diagonal sector in a fixed frame: A(t) = U diag(q(t)) V^T, U, V constant
    A, P = initial_state(f2, 30, 0.4, 4, L0=True)
    U, _, Vt = np.linalg.svd(A)
    none = [[np.zeros_like(A), np.zeros_like(A)], [np.zeros_like(A), np.zeros_like(A)]]
    for _ in range(2000):
        A, P = _advance(m2, A, P, none, 0.01, 0.25)
    D = np.transpose(U, (0, 2, 1)) @ A @ np.transpose(Vt, (0, 2, 1))
    off = D - D * np.eye(3)
    rep('su2 --L0 stays colour-diagonal in its initial frame over t = 20 (max off-diagonal)', np.abs(off).max(), 1e-9)
    print('\nall checks passed' if ok else '\nSOME CHECKS FAILED', flush=True)
    return ok


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--selftest', action='store_true', help='run the checks and exit')
    ap.add_argument('--group', choices=sorted(GROUPS), default='su3')
    ap.add_argument('--E', type=float, nargs='+')
    ap.add_argument('--N', type=int, default=1100)
    ap.add_argument('--workers', type=int, default=1, help='processes (one core each)')
    ap.add_argument('--dt', type=float, default=0.01)
    ap.add_argument('--tmax', type=float, default=1000.0, help='hard cap; may be raised on --resume')
    ap.add_argument('--t-burn', type=float, default=100.0)
    ap.add_argument('--t-seg', type=float, default=1000.0, help='checkpoint / report interval (time units)')
    ap.add_argument('--renorm', type=int, default=20)
    ap.add_argument('--omega-dt-max', type=float, default=0.25)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--L0', action='store_true',
                    help='zero spatial angular momentum (for su2 this is exactly the colour-diagonal sector)')
    ap.add_argument('--out', default=os.path.join('results', 'run'))
    ap.add_argument('--resume', action='store_true', help='continue from checkpoints in --out')
    ap.add_argument('--no-stop', action='store_true', help='ignore the convergence stop; run to --tmax')
    ap.add_argument('--min-lamT', type=float, default=20.0, help='stop needs lam_inc * t >= this')
    ap.add_argument('--target-rel', type=float, default=0.01, help='stop needs SEM(lam_inc)/lam_inc <= this')
    a = ap.parse_args()
    if a.selftest:
        t0 = time.time()
        print('# START ym_run selftest', flush=True)
        good = selftest()
        print(f'# END ym_run selftest status={"OK" if good else "FAIL"} wall={time.time() - t0:.0f}s', flush=True)
        sys.exit(0 if good else 1)
    if not a.E:
        ap.error('--E is required (or --selftest)')
    run_id = time.strftime('%Y%m%d-%H%M%S') + f'-p{os.getpid()}'
    t0 = time.time()
    print(f'# START ym_run run={run_id} group={a.group} E={" ".join(f"{e:g}" for e in a.E)} N={a.N} '
          f'workers={a.workers} dt={a.dt} tmax={a.tmax} t_burn={a.t_burn} t_seg={a.t_seg} seed={a.seed} '
          f'L0={a.L0} resume={a.resume} stop={"off" if a.no_stop else f"lamT>={a.min_lamT:g},rel<={a.target_rel:g}"} '
          f'out={a.out}', flush=True)
    status, pool = 'OK', None
    try:
        os.makedirs(a.out, exist_ok=True)
        if a.workers > 1:
            import multiprocessing as mp
            for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
                os.environ[k] = '1'
            pool = mp.get_context('spawn').Pool(a.workers)
        path = os.path.join(a.out, 'ym_run.csv')
        for E in a.E:
            row = run_energy(E, a, pool, run_id)
            if row is None:
                continue
            new = not os.path.exists(path)
            fields = list(row)
            if not new:                      # an older file keeps its columns (new ones are in the npz and the log)
                with open(path, newline='') as fh:
                    fields = next(csv.reader(fh))
            with open(path, 'a', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
                if new:
                    w.writeheader()
                w.writerow(row)
            print(f"E={E:g} DONE ({row['reason']}) t={row['t']:g}  lam_ym_T={row['lam_ym_T']:.5f}  "
                  f"lam_ghost_inc={row['lam_ghost_inc']:.7f} +- {row['lam_ghost_inc_sem']:.7f}  "
                  f"lam_ghost_T={row['lam_ghost_T']:.7f}  b={row['b_ghost']:+.3f}  lamT={row['lamT_ghost']:.1f}  "
                  f"dropped={row['dropped']}  max|dE|/E={row['max_dE']:.1e}  max|dG|={row['max_dG']:.1e}  max|L|={row['max_L']:.1e}  "
                  f"wall={row['wall_s']:.0f}s", flush=True)
    except KeyboardInterrupt:
        status = 'INTERRUPTED'
    except Exception:
        status = 'FAIL'
        traceback.print_exc()
        sys.stdout.flush()
    finally:
        if pool is not None:
            pool.terminate()
        print(f'# END ym_run run={run_id} status={status} wall={time.time() - t0:.0f}s', flush=True)
    sys.exit(0 if status == 'OK' else 1)


if __name__ == '__main__':
    main()
