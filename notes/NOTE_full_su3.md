# Full SU(3): the first scan and the long runs

Moved unchanged from `README.md` on 28 Oct 2026 (its history before that date is in `README.md`'s git log);
only cross-references were updated. Paths and commands are relative to the repository root. Conventions, including the claim notation and digits, are hdym's (its
top-level `README.md`).

## Full SU(3): the first scan (`su3_full.py`, 25 Oct)

`su3_full.py` evolves generic homogeneous SU(3) Yang–Mills (24 components, initial data projected onto the Gauss shell
G = 0 and scaled to energy E exactly) with two 24-d tangent vectors: the YM tangent (λ_YM, scale-free, so ∝ E^{1/4})
and the ghost w'' = −(1 + Hess V)w (λ_ghost, the SU(3) analogue of the paper's transverse exponent, naive operator;
a single full tangent, so it measures the fastest sector: see `NOTE_single_tangent_bias.md`).
Self-test: V, ∇V, Hess V agree with `su3_core.py` to 10⁻¹⁴; energy and Gauss conserved; the embedded SU(2) sector is
invariant under the 24-d flow. dt = 0.01 and 0.005 agree within errors on identical trajectories (not logged).

This section was the run plan; the runs are done and it is kept as their record (28 Oct 2026). From the repository
folder, in the order they were run:

```
python su3_full.py --selftest
python su3_full.py --E 1.0 --N 2200 --tmax 300 --workers 22 --out results/full_cpu_timing
python su3_full.py --E 1.0 --N 20000 --tmax 300 --backend cupy --out results/full_gpu_timing
```

The last line needs CuPy. When this was the plan it was the first run of the cupy path; it has since run
(`results/full_gpu_timing.txt`: N = 20,000 to t = 300, wall 1529 s, against N = 2,200 in 53 s on 22 CPU workers in
`results/full_cpu_timing.txt`: the CPU did about 3× the GPU's trajectory·time units per second). The first line of
`full_gpu_timing.txt` is a failed call from the old `su3\` layout, kept as logged. Production scan, as run (CPU):

```
python su3_full.py --E 1.0 0.5 0.3 --N 4400 --tmax 4000 --workers 22 --out results/full_scan
python su3_full.py --E 0.2 0.1 --N 4400 --tmax 20000 --t-block 1000 --workers 22 --out results/full_scan
```

Each energy appends a row to `results/full_scan/su3_full.csv` and writes `lam_E<E>.npz` (per-trajectory exponents and the
running-mean history: a λ still drifting in the progress lines needs a longer tmax). The ghost needs λ_ghost·tmax ≳ 20
to be trustworthy; if the lower energies come out below ~10⁻³, double tmax before reading anything into the law. The
E = 0.1 row fails that rule (λ_T = 2.97×10⁻⁴ at t = 20,000: λ·t = 5.9, `results/full_scan.txt`); `run_low` supersedes it
(2.949×10⁻⁴ at λ·t = 20.3, `results/run_low.txt`), and the two agree. That row also enters the |L| figures below.
The first scan's logs have no END line (`su3_full.py` predates that convention and stays unchanged as the record;
`ym_group.py` gets one if it is used again). `transverse_su3.py` and `doublet_physical.py` print one from 28 Oct 2026;
their committed logs predate it.

## Long runs: `ym_run.py`

`su3_full.py` stays as the record of the first scan; new runs go through `ym_run.py`. For `--group su3` its
per-trajectory logs are bit-identical to `su3_full.py`'s, including across a kill and `--resume` (checked by
`--selftest`, `results/ym_run_selftest.txt`).

- **Checkpoints.** The run is cut into segments of `--t-seg` (default 1000); each worker checkpoints to
  `<out>/ckpt/` (git-ignored). A kill loses at most one segment; rerun the same command with `--resume`.
  `--tmax` can be raised on a resume to lengthen a finished energy.
- **Estimators.** Each segment prints λ_T = ⟨L(t)⟩/t (the `su3_full.py` estimator, which carries a start-up offset
  b/t), λ_inc = ⟨L(t) − L(t/2)⟩/(t/2) (offset-free) with its SEM, and b from a straight-line fit over the second half.
- **Convergence stop** on the ghost: λ_inc·t ≥ `--min-lamT` (20) and SEM/λ_inc ≤ `--target-rel` (1 %) at two
  consecutive segment ends; otherwise it runs to `--tmax`. `--no-stop` disables it. On `--resume` an energy that has
  converged is skipped unless `--no-stop` is given. The stop bounds the statistical error only: the full-tangent bias
  (`NOTE_single_tangent_bias.md`) fades like t^(−1/2) and the stop cannot see it.
- **Logs** start with `# START ym_run run=<id>` and end with `# END ym_run run=<id> status=OK|FAIL|INTERRUPTED`;
  a log with no END line belongs to a process that died. Results: `<out>/ym_run.csv` (one row per finished energy)
  and `<out>/lam_E<E>.npz` (per-trajectory exponents and the full L_i(t) history).

```
python ym_run.py --selftest
python ym_run.py --group su3 --E 0.3 0.1 0.05 --N 1100 --workers 22 --tmax 600000 --out results/run_low
python ym_run.py --group su3 --E 0.3 0.1 0.05 --N 1100 --workers 22 --tmax 600000 --out results/run_low --resume
```

**Angular momentum.** The spatial angular momentum L_i = ε_ijk A_j^a P_k^a is conserved, and each trajectory's
exponents depend strongly on it: in the first scan the rank correlation of |L| with λ_ghost is +0.40 to +0.69 and
with λ_YM −0.88 to −0.96, and the 10 % of trajectories with the largest |L| carry a ghost rate 1.13–1.47 times the
mean (`L_correlation.py`, `results/L_correlation.txt`; the first scan's E = 0.1 row has λ·t = 5.9, see above). In
`run_low` (second block of the same log) it strengthens as E falls: +0.466, +0.682, +0.765 with λ_ghost and −0.912,
−0.986, −0.997 with λ_YM at E = 0.3, 0.1, 0.05, top-10 % factor 1.19, 1.44, 1.65. These use the full-tangent
λ_ghost (`NOTE_single_tangent_bias.md`), whose bias is per trajectory and could itself correlate with |L|. An
ensemble average over L therefore depends on how L was sampled. `--L0` projects the initial data onto L = 0; for
SU(2), G = 0 with L = 0 is exactly hdym's colour-diagonal sector in a fixed frame (checked by `--selftest` to
6 × 10⁻¹⁴), so `--L0` is the like-for-like setting for comparisons with hdym. The `run_low` scan is the generic-L
ensemble.

```
python ym_run.py --group su3 --E 1 0.5 0.3 0.1 0.05 --N 1100 --workers 22 --tmax 600000 --L0 --out results/run_L0
```

**Results** (added 28 Oct 2026 from the committed logs; naive operator, one full 24-d tangent). λ_ghost is λ_inc ±
SEM at the convergence stop, λ_YM is λ_T (`results/run_low.txt`, `results/run_L0.txt`, the `DONE` lines); rows marked
† are the first scan's λ_T (`results/full_scan.txt`):

| E | λ_ghost, generic L | λ_ghost, L = 0 | λ_YM, generic L | λ_YM, L = 0 |
|---|---|---|---|---|
| 1 | 0.046632 ± 0.000064 † | 0.0437913 ± 0.0000958 | 0.26216 † | 0.27520 |
| 0.5 | 0.012604 ± 0.000016 † | 0.0121205 ± 0.0000348 | 0.22043 † | 0.23160 |
| 0.3 | 0.0048219 ± 0.0000191 | 0.0045457 ± 0.0000128 | 0.19398 | 0.20396 |
| 0.2 | 0.001873 ± 0.000004 † | — | 0.17532 † | — |
| 0.1 | 0.0002949 ± 0.0000020 | 0.0002638 ± 0.0000009 | 0.14740 | 0.15488 |
| 0.05 | 0.0000430 ± 0.0000004 | 0.0000363 ± 0.0000001 | 0.12393 | 0.13022 |

- λ_YM/E^{1/4} ∈ [0.26208, 0.26216] on every generic-L row of this table and ∈ [0.27520, 0.27559] on every L = 0 row
  [grid]: the scale-free law, with L = 0 about 5 % more chaotic.
- Local ghost slopes d ln λ/d ln E between neighbouring rows: L = 0: 1.85, 1.92, 2.59, 2.86; generic L (`run_low`
  only): 2.54 (0.3 → 0.1), 2.78 (0.1 → 0.05). Not final: see `NOTE_single_tangent_bias.md`.
- Against SU(2) at L = 0, same naive operator: E = 0.3: 0.0045457 against 0.016506 (hdym `ym/data/transverse_E0.3.npz`,
  `lam_v`, the vector ghost, which leads there), ratio 0.275; E = 0.1: 0.0002638 against 1.543×10⁻³ (hdym's GPU kernel,
  one off-diagonal block, `ym/data/results_rung1/history_E0.1_cj0.npy` at t = 20,000), ratio 0.171. The SU(3) values
  are full-tangent (biased upward), the SU(2) ones single-sector, so the bias works against the conclusion: SU(3)
  suppresses the low-E ghost at least this much [est].
