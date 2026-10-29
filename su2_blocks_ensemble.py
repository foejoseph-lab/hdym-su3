"""
su2_blocks_ensemble.py -- su2_blocks.py on all 1,100 initial conditions of results/su2_L0_check/, compared at
matched t with that run's single full tangent.

Why: su2_blocks.txt (50 trajectories, t <= 18,000) leaves two things open (notes/NOTE_single_tangent_bias.md):
  1. the one-block value sits about 6 % above hdym's GPU (1.641 against 1.543 x 10^-3), about 3 sigma for N = 50;
  2. whether lam_inc (what run_low and run_L0 report) carries the max-over-sectors bias or only lam_T does
     (results/su2_L0_check: lam_inc carries about half of lam_T's excess against the GPU, but that comparison mixes
     ensembles).
Running su2_blocks.py for each of ym_run's worker seeds (seed + 1000 k, k = 0..21, 50 trajectories each) puts the
per-sector tangents on exactly ym_run's initial ensemble, to the same t = 47,000, so both are measured on one
ensemble: one block against the GPU with a standard error about sqrt(22) smaller, and max-of-three-blocks against
ym_run's full tangent with both estimators (lam_T and lam_inc formed as in ym_run.estimates).
Same initial data, independent integrators (DOP853 against Yoshida-4), so the chaotic trajectories are not the same;
the pairing is of ensembles, not of trajectories.

    python su2_blocks_ensemble.py --E 0.1 --tmax 47000 --workers 22 [--resume]
    python su2_blocks_ensemble.py --combine-only           (rebuild the table from whatever the children have saved)

Each child writes results/su2_blocks_L0check/log_E0.1_s<seed>.txt ('w' fresh, 'a' with --resume) and its npz and
checkpoint beside it; the combined table goes to stdout (redirect it to results/su2_blocks_L0check.txt). One thread
per child. Cost: the 50-trajectory log took 2631 s to t = 18,000 on one core, slowing as it went (DOP853 steps
are set by the stiffest trajectory in each batch); to t = 47,000 an estimate of 4-8 h per child with 22 running at
once. Added 2026-10-28.
"""
import argparse, os, subprocess, sys, time, warnings
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--E', type=float, default=0.1)
ap.add_argument('--tmax', type=float, default=47000.0)
ap.add_argument('--workers', type=int, default=22, help='children running at once')
ap.add_argument('--nseeds', type=int, default=22, help="ym_run's worker count in results/su2_L0_check (22)")
ap.add_argument('--seed', type=int, default=11, help="ym_run's --seed in results/su2_L0_check (11)")
ap.add_argument('--N', type=int, default=50, help="trajectories per ym_run worker (1100/22)")
ap.add_argument('--report', type=float, default=1000.0)
ap.add_argument('--out', default=os.path.join('results', 'su2_blocks_L0check'))
ap.add_argument('--ref', default=os.path.join('results', 'su2_L0_check'), help="ym_run's output folder")
ap.add_argument('--gpu', type=float, default=1.543e-3, help='hdym GPU one block at t = 20,000 (npz)')
ap.add_argument('--resume', action='store_true')
ap.add_argument('--combine-only', action='store_true')
ap.add_argument('--poll', type=float, default=30.0)
a = ap.parse_args()
T0 = time.time()
SEEDS = [a.seed + 1000 * k for k in range(a.nseeds)]
os.makedirs(a.out, exist_ok=True)
print(f'# START su2_blocks_ensemble E={a.E:g} tmax={a.tmax:g} seeds={SEEDS[0]}..{SEEDS[-1]} (step 1000) '
      f'N_each={a.N} workers={a.workers} report={a.report:g} resume={a.resume} combine_only={a.combine_only}', flush=True)


def child_log(s):
    return os.path.join(a.out, f'log_E{a.E:g}_s{s}.txt')


def child_ok(s):
    """Finished to --tmax: the child's log ends with its OK line and its npz has reached tmax."""
    try:
        with open(child_log(s), encoding='utf-8') as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
        z = np.load(os.path.join(a.out, f'logs_E{a.E:g}_s{s}.npz'))
        return bool(lines) and lines[-1].startswith('# END su2_blocks status=OK') and float(z['t']) >= a.tmax - 1e-9
    except (OSError, KeyError, ValueError):
        return False


# ------------------------------------------------------------------ run the children
status = 'OK'
if not a.combine_only:
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    todo = [s for s in SEEDS if not (a.resume and child_ok(s))]
    print(f'# children to run: {len(todo)} of {len(SEEDS)}', flush=True)
    running, failed = {}, []
    while todo or running:
        while todo and len(running) < a.workers:
            s = todo.pop(0)
            fh = open(child_log(s), 'a' if a.resume else 'w', encoding='utf-8')
            cmd = [sys.executable, '-B', '-u', os.path.join(HERE, 'su2_blocks.py'), '--E', repr(a.E),
                   '--N', str(a.N), '--seed', str(s), '--tmax', repr(a.tmax), '--report', repr(a.report),
                   '--out', a.out, '--tag', f'_s{s}'] + (['--resume'] if a.resume else [])
            running[s] = (subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=os.getcwd()), fh)
        time.sleep(a.poll)
        for s in list(running):
            pr, fh = running[s]
            if pr.poll() is not None:
                fh.close()
                ok = pr.returncode == 0 and child_ok(s)
                if not ok:
                    failed.append(s)
                print(f'  seed {s}: exit {pr.returncode} {"OK" if ok else "FAILED"}  '
                      f'({len(running) - 1} running, {len(todo)} queued, wall={time.time() - T0:.0f}s)', flush=True)
                del running[s]
    if failed:
        status = f'FAILED seeds={failed}'
        print(f'# failed children: {failed}; rerun with --resume to continue them', flush=True)


# ------------------------------------------------------------------ combine
def inc(t_hist, L_hist, k):
    """ym_run.estimates' lam_T and lam_inc at report index k: L_hist[k, i] cumulative, t_hist[0] = 0.
    Returns per-trajectory lam_T and lam_inc (nan when there is no earlier time >= t/2)."""
    t = t_hist[k]
    later = np.nonzero((t_hist[:k + 1] >= t / 2) & (t_hist[:k + 1] < t))[0]
    lt = L_hist[k] / t
    if len(later) == 0:
        return lt, np.full_like(lt, np.nan)
    h = later[0]
    return lt, (L_hist[k] - L_hist[h]) / (t - t_hist[h])


have = [s for s in SEEDS if os.path.exists(os.path.join(a.out, f'logs_E{a.E:g}_s{s}.npz'))]
if not have:
    print('# no child npz found; nothing to combine', flush=True)
    print(f'# END su2_blocks_ensemble status=FAILED(no data) wall={time.time() - T0:.0f}s', flush=True)
    sys.exit(1)
Z = [np.load(os.path.join(a.out, f'logs_E{a.E:g}_s{s}.npz')) for s in have]
nrep = min(len(z['t_hist']) for z in Z)
th = Z[0]['t_hist'][:nrep]
for z in Z:
    if not np.allclose(z['t_hist'][:nrep], th):
        raise SystemExit('children report at different times; use the same --report')
LH = np.concatenate([z['logs_hist'][:nrep] for z in Z], axis=1)       # (nrep, N_total, 4): vec, b12, b13, b23
dE = np.concatenate([z['dE'] for z in Z])
Ntot = LH.shape[1]
print(f'# combined: {len(have)} of {len(SEEDS)} seeds, {Ntot} trajectories, common t <= {th[-1]:g}; '
      f'max|dE|/E at each child\'s last report {np.abs(dE).max() / a.E:.1e}', flush=True)

ref_npz = os.path.join(a.ref, f'lam_E{a.E:g}.npz')
R = np.load(ref_npz) if os.path.exists(ref_npz) else None
if R is not None:
    rt, rL = R['t'], R['L_ghost']            # (48,), (48, 1100): ym_run's cumulative ghost logs
    print(f'# reference: {ref_npz} (ym_run full tangent, {rL.shape[1]} trajectories, t <= {rt[-1]:g})', flush=True)
else:
    print(f'# reference {ref_npz} not found: full-tangent columns left blank', flush=True)

print('# columns: one block = mean over the three off-diagonal blocks (what hdym\'s GPU follows), +- SE over '
      'trajectories;\n#          max3 = max over the three blocks per trajectory (what one full tangent rides);\n'
      '#          full = ym_run single tangent on the same initial ensemble (results/su2_L0_check);\n'
      '#          _T = L(t)/t, _inc = second-half increment as in ym_run.estimates; vec = vector ghost (lam_T)')
hdr = (f"{'t':>7} {'one_T':>10} {'+-':>8} {'one_inc':>10} {'+-':>8} {'max3_T':>10} {'max3_inc':>10} "
       f"{'full_T':>10} {'full_inc':>10} {'+-':>8} {'full_T/max3_T':>13} {'full_inc/max3_inc':>17} {'vec_T':>10} "
       f"{'one_T/gpu':>9}")
print(hdr, flush=True)
warnings.filterwarnings('ignore', category=RuntimeWarning)   # nan columns before a second-half window exists
for k in range(1, nrep):
    t = th[k]
    bT = np.empty((Ntot, 3)); bI = np.empty((Ntot, 3))
    for c in range(3):
        bT[:, c], bI[:, c] = inc(th, LH[:, :, 1 + c], k)
    oneT, oneI = bT.mean(1), bI.mean(1)
    mxT, mxI = bT.max(1), bI.max(1)
    vT = LH[k, :, 0] / t
    sq = np.sqrt(Ntot)
    row = (f"{t:7.0f} {oneT.mean():10.4e} {oneT.std() / sq:8.1e} {np.nanmean(oneI):10.4e} "
           f"{np.nanstd(oneI) / sq:8.1e} {mxT.mean():10.4e} {np.nanmean(mxI):10.4e} ")
    j = np.nonzero(np.isclose(rt, t))[0] if R is not None else []
    if len(j):
        fT, fI = inc(rt, rL, int(j[0]))
        row += (f"{fT.mean():10.4e} {np.nanmean(fI):10.4e} {np.nanstd(fI) / np.sqrt(fI.size):8.1e} "
                f"{fT.mean() / mxT.mean():13.4f} {np.nanmean(fI) / np.nanmean(mxI):17.4f} ")
    else:
        row += f"{'':>10} {'':>10} {'':>8} {'':>13} {'':>17} "
    row += f"{vT.mean():10.4e} {oneT.mean() / a.gpu:9.4f}"
    print(row, flush=True)
print(f'# gpu reference {a.gpu:.4e} is hdym\'s one block at t = 20,000 (ym/data/results_rung1/history_E0.1_cj0.npy); '
      f'one_T/gpu is meaningful near that t only', flush=True)
if len(have) < len(SEEDS) and status == 'OK':
    status = f'PARTIAL({len(have)}/{len(SEEDS)})'
print(f'# END su2_blocks_ensemble status={status} wall={time.time() - T0:.0f}s', flush=True)
