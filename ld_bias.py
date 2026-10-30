"""
ld_bias.py -- a central-limit (quadratic large-deviation) model of the single-tangent bias, tested on SU(2) and
applied to SU(3).  Reads committed results only; integrates nothing.

Model: each sector k of a trajectory has cumulative log L_k(t) = lambda t + b + sqrt(D t) g_k, g_k standard normal,
correlated across sectors with rho.  A single full tangent rides the fastest sector, so its log is max_k L_k(t).
With e_n, v_n the mean and variance of the maximum of n such normals:
    lam_T(full) - lam_T(one)     = e_n sqrt(D/t)
    lam_inc(full) - lam_inc(one) = 2 (1 - 1/sqrt 2) e_n sqrt(D/t)      (ym_run's second-half increment)
    var L_full(t) / t            = v_n D
Part 1 tests the model on SU(2) at L = 0, E = 0.1, where the sectors are measured (results/su2_blocks_L0check/,
results/su2_L0_check/).  Part 2 applies it to SU(3) (results/run_L0/, results/run_low/), taking D from each run's
own full-tangent variance via v_n, for n = 2 .. 8 competing sectors and SU(2)'s rho; it is valid only where var L/t is
flat in t (diffusive), which part 2 prints.  Monte Carlo for e_n, v_n with a fixed seed.

    python ld_bias.py > results/ld_bias.txt
Seconds.  Added 2026-10-29 (notes/NOTE_single_tangent_bias.md, "A large-deviation model of the bias").
"""
import glob, os, sys, time
import numpy as np

T0 = time.time()
print(f'# START ld_bias  python {sys.version.split()[0]} numpy {np.__version__}', flush=True)
rng = np.random.default_rng(1)
C = 2 * (1 - 1 / np.sqrt(2))


def maxstats(n, rho, m=400000):
    cov = np.full((n, n), rho) + np.eye(n) * (1 - rho)
    g = rng.multivariate_normal(np.zeros(n), cov, m).max(1)
    return g.mean(), g.var()


def incr(th, L, k):
    t = th[k]; h = np.nonzero(th >= t / 2)[0][0]
    return (L[k] - L[h]) / (t - th[h])


# ------------------------------------------------------------------ part 1: SU(2)
Z = [np.load(f) for f in sorted(glob.glob(os.path.join('results', 'su2_blocks_L0check', 'logs_E0.1_s*.npz')))]
th = Z[0]['t_hist']; LH = np.concatenate([z['logs_hist'] for z in Z], 1)          # (nrep, 1100, 4): vec, b12, b13, b23
B = LH[:, :, 1:]
R = np.load(os.path.join('results', 'su2_L0_check', 'lam_E0.1.npz')); rt, rL = R['t'], R['L_ghost']
print(f'# part 1: SU(2), L = 0, E = 0.1: {B.shape[1]} trajectories, three off-diagonal blocks; full = ym_run single tangent')
print(f"{'t':>6} {'one_T':>10} {'maxL_T':>10} {'gauss_T':>10} {'full_T':>10} | {'one_inc':>10} {'maxL_inc':>10} {'full_inc':>10} | {'D':>9} {'rho':>6}")
rho_last = D_last = None
for t in (5000, 10000, 20000, 30000, 40000, 47000):
    k = np.where(th == t)[0][0]; kr = np.where(rt == t)[0][0]
    one = B[k].mean() / t; M = B.max(2); maxT = M[k].mean() / t
    fT = rL[kr].mean() / t
    oneI = np.mean([incr(th, B[:, :, c], k).mean() for c in range(3)]); mI = incr(th, M, k).mean()
    fI = incr(rt, rL, kr).mean()
    X = B[k] - B[k].mean(); D = X.var() / t
    Cm = np.corrcoef(X.T); rho = (Cm[0, 1] + Cm[0, 2] + Cm[1, 2]) / 3
    e3, v3 = maxstats(3, rho)
    print(f'{t:6d} {one:10.4e} {maxT:10.4e} {one + e3 * np.sqrt(D / t):10.4e} {fT:10.4e} | {oneI:10.4e} {mI:10.4e} {fI:10.4e} | '
          f'{D:9.3e} {rho:+6.3f}', flush=True)
    rho_last, D_last, t_last, one_last, oneI_last, fI_last = rho, D, t, one, oneI, fI
e3, v3 = maxstats(3, rho_last)
print(f'  vector ghost leads (largest L) on {np.mean(LH[-1].argmax(1) == 0):.3f} of trajectories at t = {th[-1]:g}')
print(f'  lam_inc bias at t = {t_last}: model C e_3 sqrt(D/t) = {C * e3 * np.sqrt(D_last / t_last):.3e}, measured full - one = '
      f'{fI_last - oneI_last:.3e}; from the full tangent alone (D = var L_full/(t v_3)) = '
      f'{C * e3 * np.sqrt(rL[-1].var() / (rt[-1] * v3) / rt[-1]):.3e}  (e_3 = {e3:.3f}, v_3 = {v3:.3f}, rho = {rho_last:.3f})')

# ------------------------------------------------------------------ part 2: SU(3)
tab = {n: maxstats(n, rho_last) for n in (2, 3, 4, 6, 8)}
print(f'# part 2: SU(3), predicted lam_inc bias [est], D from each run\'s full-tangent variance, rho = {rho_last:.3f}')
for run in ('run_L0', 'run_low'):
    rows = []
    for E in (0.3, 0.1, 0.05):
        p = os.path.join('results', run, f'lam_E{E:g}.npz')
        if not os.path.exists(p):
            continue
        z = np.load(p); tt, L = z['t'], z['L_ghost']; T = tt[-1]
        li = incr(tt, L, len(tt) - 1).mean(); Df = L[-1].var() / T
        idx = [max(1, len(tt) // 8), max(1, len(tt) // 4), len(tt) // 2, len(tt) - 1]
        diff = ' '.join(f'{L[i].var() / tt[i]:.2e}' for i in idx)
        b = {n: C * e * np.sqrt(Df / v / T) / li for n, (e, v) in tab.items()}
        rows.append((E, li, b))
        print(f'  {run} E = {E:g}: lam_inc {li:.4e} at t = {T:g}; var L/t at t = {", ".join(f"{tt[i]:g}" for i in idx)}: {diff}; '
              f'bias ' + ', '.join(f'n={n}: {100 * x:.1f}%' for n, x in b.items()), flush=True)
    for (E1, l1, b1), (E2, l2, b2) in zip(rows, rows[1:]):
        s = np.log(l1 / l2) / np.log(E1 / E2)
        cs = ', '.join(f'n={n}: {np.log(l1 / (1 + b1[n]) / (l2 / (1 + b2[n]))) / np.log(E1 / E2):.2f}' for n in b1)
        print(f'    local slope {E1:g} -> {E2:g}: measured {s:.2f}; bias-corrected {cs}', flush=True)
print('  (run_low: var L/t grows with t, i.e. trajectories have different true rates (the |L| hidden parameter), so D is'
      ' not a diffusion constant there and its lines are not predictions)')
print(f'# END ld_bias status=OK wall={time.time() - T0:.0f}s', flush=True)
