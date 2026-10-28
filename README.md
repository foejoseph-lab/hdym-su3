# hdym-su3 — Lee–Wick Yang–Mills mechanics moved up to SU(3): the isospin embedding, the doublet, and an escape that is not the ghost's

A separate project from [hdym](https://github.com/foejoseph-lab/hdym) (the SU(2) paper, *The Lee–Wick ghost of
Yang–Mills mechanics is unstable at every energy*), started 24 Oct 2026 as hdym's `su3/` folder and moved here with its
history so that the SU(2) project stays SU(2). The question is the paper's "move up to another matrix group": embed
hdym's 12-d colour-diagonal SU(2) system in homogeneous SU(3) Lee–Wick Yang–Mills and ask what the new directions do.
Units g = M = 1 and hdym's conventions (L = −¼F² + (1/2M²)(DF)², F^a_ij = f^abc A_i^b A_j^c). The SU(2) model is
`hdym_core.py`, a verbatim copy of hdym's `ym/hdym_core.py` at commit 1d65e30 (header says so); this repository needs
nothing else from hdym. Python ≥ 3.10 with numpy and scipy (sympy for `derive_doublet.py`, CuPy for GPU runs).
Documentation conventions, including the claim notation of 28 Oct 2026 ([grid], [exact], [cert], [est]), are hdym's
(its top-level `README.md`, "Conventions" and "Claim notation").

## Results

**The embedding (exact).** With A_i^a = δ_i^a q_i for a ∈ {1, 2, 3} (Gell-Mann λ₁, λ₂, λ₃, the isospin subalgebra),
the SU(3) potential and gradient equal `hdym_core.V` and `gradV` to 10⁻¹⁴, and the gradient has no component off
the embedded sector, so the sector is invariant (`check_embedding.py`, all 11 checks pass). The 24 × 24 Hessian on
the sector is block-diagonal, adjoint (9) ⊕ doublet λ₄…λ₇ (12) ⊕ singlet λ₈ (3), with zero mixing; the singlet block
is identically zero — λ₈ never feels the background.

**The doublet block (exact).** The 12 × 12 block is four identical copies of one 3 × 3 matrix
(`derive_doublet.py`, sympy, checked to 4 × 10⁻¹⁶ against the numeric Hessian; copy 1 is (A₁⁴, A₂⁵, −A₃⁶)):

    M(q) = ¼ diag(y² + z², z² + x², x² + y²) + ¾ [xy, xz, yz off-diagonal]

against Hess V = diag(y² + z², …) + 2 [xy, …] for SU(2) (`doublet.py`). In each copy:
- gauge mode: M q = ∇V exactly, so δ = q(t) solves δ̈ = −Mδ and is bounded;
- Gauss charge G = q·δ̇ − q̇·δ is conserved (M symmetric);
- min eig M ≥ −½√(2V), half the SU(2) bound, so the doublet ghost's mass matrix 1 + M can go non-positive only
  for E ≥ 2 (SU(2): E ≥ ½); deep in a channel min eig M ≈ −2(y² + z²) against −3(y² + z²) for SU(2). (Derived in
  session; no committed check or log.)

**Transverse exponents (measured).** All four on the same pure-YM trajectories, N = 400, dt = 0.005 with the
per-trajectory channel sub-stepping of `hdym_core.ym_step_substepped` (with ω floored at 1), no trajectory dropped,
max |ΔE|/E ≤ 5.5 × 10⁻⁴ (`transverse_su3.py`; `results/transverse_E1_E05.txt`, `results/transverse_E03.txt`):

| E | in-sector YM chaos | doublet, pure YM | SU(2) ghost | doublet ghost |
|---|---|---|---|---|
| 1.0 | 0.3749 ± 0.0020 | 0.1025 ± 0.0008 | 0.0884 ± 0.0006 | 0.0387 ± 0.0003 |
| 0.5 | 0.3162 ± 0.0015 | 0.0864 ± 0.0007 | 0.0305 ± 0.0002 | 0.0091 ± 0.0001 |
| 0.3 | 0.2767 ± 0.0011 | 0.0758 ± 0.0005 | 0.0165 ± 0.0001 | 0.00408 ± 0.00004 |

- The doublet column divided by E^{1/4} is 0.1025, 0.1028, 0.1024 (E = 1, 0.5, 0.3; order corrected 28 Oct 2026):
  **the pure-YM escape from the sector is λ_esc = 0.1025 E^{1/4}**, about 27 % of in-sector chaos (0.375 E^{1/4}), as
  the scale-free quartic requires.
- It is physical: starting on G = 0 and removing the gauge mode at every renormalisation gives 0.1047 and 0.1044
  against 0.1050 unconstrained, ± 0.0012, with G conserved to 10⁻¹² (`doublet_physical.py`,
  `results/doublet_physical_E1.txt`, E = 1, N = 300, tmax = 1000).
- λ_esc / (SU(2) ghost) = 1.16, 2.84, 4.59 at E = 1, 0.5, 0.3; if the ghost reaches the paper's 0.0120 E^{9/4}
  (hdym `ym/README.md`, the parameter-free comparison) at low E the ratio grows like E⁻², about 10⁵ at E = 0.008 (an
  extrapolation, not a measurement).
- The doublet ghost is the weaker ghost: 0.44, 0.30, 0.25 of the SU(2) ghost at E = 1, 0.5, 0.3.
- The ghost local slopes here are 1.2–2.1, not 9/4: E ≥ 0.3 is not the paper's asymptotic regime, and nothing here
  tests the E^{9/4} law.

**Reading.** In SU(2) the colour-diagonal sector is protected by a conserved quantity; inside SU(3) it is not, and a
generic trajectory leaves it at 0.1025 E^{1/4} — not a blow-up (pure-YM energy is bounded below) but a wandering into
the full homogeneous SU(3) phase space. The SU(3) analogue of the paper's measurement is therefore the ghost's
transverse exponent about generic SU(3) trajectories, with the full 24 × 24 operator 1 + Hess V_SU(3) — measured
25–27 Oct: "Long runs", results below.

## Open

- The full-SU(3) ghost exponent above: measured 25–27 Oct ("Long runs: `ym_run.py`", results there). Open: the
  size of the full-tangent bias in SU(3) ("A single full tangent measures the fastest sector"), which needs the
  sectors followed separately. (Until 28 Oct this item read "production runs to be done".)
- The ghost lines use the naive operator ẅ = −(1 + M)w, the level of the paper's diagonal ghost; the Lee–Wick
  Gauss-law reduction of the doublet ghost (cf. hdym `rabbit_holes/ym/` for the SU(2) off-diagonal block) is not done.
- The second SU(2) embedding (λ₂, λ₅, λ₇, embedding index 4) is not done.

## Files

| file | what it does |
|---|---|
| `su3_core.py` | homogeneous SU(3) V, ∇V, exact Hessian–vector product, Gell-Mann f, the isospin embedding |
| `check_embedding.py` | the 11 checks of the embedding and the block structure |
| `derive_doublet.py` | sympy derivation of the doublet block and its four-fold degeneracy |
| `doublet.py` | the 3 × 3 doublet matrix M(q) |
| `transverse_su3.py` | the four transverse exponents on shared trajectories (sub-stepped) |
| `doublet_physical.py` | raw vs Gauss-constrained vs gauge-removed doublet exponent |
| `su3_full.py` | generic 24-component SU(3) YM on the Gauss shell, with λ_YM and the full 24 × 24 ghost λ_ghost; `--backend numpy --workers K` (K cores) or `--backend cupy` (GPU); `--selftest` |
| `groups.py` | structure constants for su2, su3, so4 in one convention, and the so(4) = su(2) ⊕ su(2) basis |
| `ym_group.py` | `su3_full.py` with the group as a parameter (`--group su2 / su3 / so4`); `--group su3` reproduces `su3_full.py` exactly |
| `check_so4.py` | SO(4) as a should-match check of the group-general code (14 checks) |
| `ym_run.py` | the long-run engine: `su3_full.py`'s physics for `--group su2 / su3 / so4`, with checkpoint/`--resume`, the offset-free estimator λ_inc and a convergence stop; `--selftest` |
| `su2_blocks.py` | SU(2) at L = 0 with an independent integrator (DOP853) and the naive ghost split by sector: one off-diagonal block vs the max over blocks that a single full tangent measures |
| `lame_isotropic.py` | the naive ghost on SU(3)'s two isotropic orbits (λ = 1 isospin, λ = 1/2 real SO(3)) reduced to Lamé equations per Hessian eigen-family; windows, pure-YM limits, direct 24-d Floquet cross-check |
| `L_correlation.py` | per-trajectory exponents against spatial angular momentum \|L\|, recomputed from a finished run's seeds (no rerun) |
| `results/` | the logs every number in this README is taken from, except those marked as not logged or not committed |

Run commands (one process each; logged wall times: 18 + 11 min for the first transverse command (E = 1, 0.5), 23 min
for the second, 4 min for the last):

```
python check_embedding.py
python transverse_su3.py --E 1.0 0.5 --N 400 --dt 0.005 --tmax 2000 --t-burn 100 --seed 1
python transverse_su3.py --E 0.3 --N 400 --dt 0.005 --tmax 4000 --t-burn 100 --seed 2
python doublet_physical.py --E 1.0 --N 300 --tmax 1000
```

## SO(4): a should-match check

so(4) = su(2) ⊕ su(2), each half with coupling g = √2 in the basis L± = (J ± K)/√2, so homogeneous SO(4) Yang–Mills is
two decoupled SU(2) systems. Scaling g away (B = gA, same time, ghost mass unchanged), an SO(4) state with all its
energy E in one half is a generic SU(2) state at energy 2E — background and both tangents. `check_so4.py` tests this
and the rest of the group-general code; all 14 checks pass (`results/check_so4.txt`, 25 Oct, run on Claude's side):

- f antisymmetric and Jacobi for su2, su3, so4; so(4) in the split basis is block-diag(√2 ε, √2 ε) to 4 × 10⁻¹⁶;
- `ym_group.py --group su3` gives the same exponents as `su3_full.py`, bit for bit;
- `--group su2` on the colour-diagonal sector gives hdym's V and ∇V (`hdym_core.py`) to 10⁻¹⁴;
- generic SO(4): each half drifts by no more than the total energy (ratio 1.00 at t = 20, energy genuinely shared,
  smaller half ≥ 8.9 %, from the log's 1/11.2) — no exchange between the halves, only integration error (16 × smaller
  per halving of dt: a comment at `check_so4.py:108`, not in the log);
- half-loaded SO(4) at E = 0.5 vs generic SU(2) at E = 1, 64 trajectories, t = 30: per-trajectory log-growth of the
  YM tangent agrees to 2 × 10⁻⁹ (growth up to e^14.5), of the ghost tangent to 4 × 10⁻¹⁰, background to 3 × 10⁻¹⁰.

```
python check_so4.py
```

## Full SU(3): the first scan (`su3_full.py`, 25 Oct)

`su3_full.py` evolves generic homogeneous SU(3) Yang–Mills (24 components, initial data projected onto the Gauss shell
G = 0 and scaled to energy E exactly) with two 24-d tangent vectors: the YM tangent (λ_YM, scale-free, so ∝ E^{1/4})
and the ghost w'' = −(1 + Hess V)w (λ_ghost, the SU(3) analogue of the paper's transverse exponent, naive operator;
a single full tangent, so it measures the fastest sector: see "A single full tangent" below).
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
The first scan's logs have no END line (`su3_full.py` predates that convention).

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
  ("A single full tangent" below) fades like t^(−1/2) and the stop cannot see it.
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
λ_ghost ("A single full tangent" below), whose bias is per trajectory and could itself correlate with |L|. An
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
  only): 2.54 (0.3 → 0.1), 2.78 (0.1 → 0.05). Not final: see the bias section below.
- Against SU(2) at L = 0, same naive operator: E = 0.3: 0.0045457 against 0.016506 (hdym `ym/data/transverse_E0.3.npz`,
  `lam_v`, the vector ghost, which leads there), ratio 0.275; E = 0.1: 0.0002638 against 1.543×10⁻³ (hdym's GPU kernel,
  one off-diagonal block, `ym/data/results_rung1/history_E0.1_cj0.npy` at t = 20,000), ratio 0.171. The SU(3) values
  are full-tangent (biased upward), the SU(2) ones single-sector, so the bias works against the conclusion: SU(3)
  suppresses the low-E ghost at least this much [est].

## Isotropic orbits: the ghost as Lamé equations (`lame_isotropic.py`)

The isotropic backgrounds of SU(N)-natural inflation, A_i = s(t)T_i with the T_i an SU(2) subalgebra, are
straight-line orbits; in SU(3) there are two (λ = 1 on λ₁–λ₃, λ = 1/2 on λ₂, λ₅, λ₇). Along each, Hess V(sÂ) = s² Hess
V(Â), so the naive ghost splits into one Lamé equation per eigen-family μ of Hess V(Â), with n(n+1) = 2μ/α and the
ghost mass entering only the spectral parameter. Integer n is finite-gap (exactly n gaps). Results
(`results/lame_isotropic.txt`, naive operator):

- λ = 1: families n = −½ (×8, the doublet λ₄–λ₇), 0 (×8), 1 (×7), 2 (×1, radial). The radial window is SU(2)'s,
  E ∈ (1/8, 1/6) (hdym `rabbit_holes/ym/code/rh5_lame.py`). The doublet is the only non-integer family and the only
  one along which pure Yang–Mills itself is unstable; its ghost windows found are E ∈ (0.578, 0.678) and E > 117.7
  (non-integer n also has infinitely many thinner tongues, below this search's resolution).
- λ = 1/2: families n = 0 (×5), 1 (×15), 2 (×1), 3 (×3): entirely finite-gap. Radial window E ∈ (1/2, 2/3) (the
  SU(2) window rescaled); n = 3: E ∈ (0.0962, 0.1) and E > 1.664. Pure Yang–Mills is stable or marginal along
  every family.
- Direct 24-d Floquet along the integrated orbit agrees with the family prediction at all 11 test energies.

Caveat: the naive operator. The radial family does not involve the Lee–Wick Gauss term and stands; the n = 1
families (E > 3/2 and E > 6) include off-diagonal-type directions, where on SU(2)'s x = y = z orbit the exact
operator shows no instability up to E = 3 (hdym `RH5_nonplanar.md`), so they need the exact operator before any claim.

```
python lame_isotropic.py
```

## A single full tangent measures the fastest sector, not one sector (`su2_blocks.py`)

`ym_run.py` follows one tangent through all components. In finite time it rides whichever sector is growing fastest on
each trajectory, so its estimate is the expected **maximum** of the sectors' noisy finite-time exponents: biased upward.
The exponents agree in the infinite-time limit (by ergodicity every block of a trajectory has the same exponent), but
the bias fades slowly.

Found on 27 Oct by a same-code check of hdym's SU(2) numbers. `ym_run.py --group su2 --L0 --E 0.1 --N 1100` gave
λ_T = 1.804×10⁻³ at t = 20,000 against hdym's GPU kernel (`ym/hdym_rung1_gpu.py`, which follows **one** off-diagonal
block) at 1.543×10⁻³ at the same t, flat at 1.514–1.543 for t = 15,000–30,000 (hdym
`ym/data/results_rung1/history_E0.1_cj0.npy`). At E = 0.3 the same run matched hdym's vector ghost, which leads there,
to 0.7 % (0.016394 against 0.016506, hdym `ym/data/transverse_E0.3.npz`). The run's files are to go in
`results/su2_L0_check/`, which is not committed yet (28 Oct 2026); until then 1.804×10⁻³, 0.016394 and the +16.9 % /
+15.5 % below have no committed log.

`su2_blocks.py` integrates the first 50 initial conditions of that run (same initial data; its DOP853 burn-in differs
from `ym_run.py`'s, so the chaotic trajectories are not the same) in their fixed diagonal frame with DOP853 (rtol
10⁻¹⁰, energy drift ≤ 2.4×10⁻⁵), with separate tangents for the vector ghost and each off-diagonal block
(`results/su2_blocks.txt`):

| t | one block | max of three blocks | ratio |
|---|---|---|---|
| 9,000 | 1.692×10⁻³ ± 3.6×10⁻⁵ | 2.038×10⁻³ | 1.204 |
| 18,000 | 1.641×10⁻³ ± 3.0×10⁻⁵ | 1.858×10⁻³ | 1.132 |

- `ym_run.py`'s single tangent from the same initial conditions tracks the max-of-blocks value (within about 2 % at
  t = 8,000; a session check, not logged), so its integrator is not the cause.
- The excess decays like t^(−1/2): the 1,100-trajectory run against the GPU gives +16.9 % at t = 20,000 and +15.5 % at
  23,000, where t^(−1/2) predicts 15.8 %. At that rate it is still about 12 % at t = 40,000 (15.5 % × (23,000/40,000)^½
  = 11.8 %; stated as 10 % until 28 Oct 2026).
- The one-block value is about 6 % above the GPU's at these times (1.641 against 1.543×10⁻³; about 3σ for 50
  trajectories). The initial ensembles differ (`ym_run.py`'s L = 0 projection against the GPU's Gaussian diagonal
  sampling), and the 1,100-trajectory run divided by this log's ratio (1.804/1.132 = 1.59×10⁻³) lands about 3 % above
  the GPU. No sign of a kernel problem, but not an exact match either: a larger `su2_blocks.py` run (more trajectories)
  would settle the remaining few per cent.
- The vector ghost is slower at E = 0.1 (4.3×10⁻⁴), so the off-diagonal blocks lead there.

**Consequence for SU(3).** The full-tangent λ_ghost values in `run_low` and `run_L0` carry the same kind of upward bias,
of unknown size, probably larger with more sectors. Its direction leaves "SU(3) suppresses the low-E ghost more than
SU(2)" intact, but the low-E slope (2.86 at L = 0) should not be quoted as final until the bias is measured by following
sectors separately.

```
python su2_blocks.py --E 0.1 --tmax 18000
```

The code and the derivations were developed in an interactive session with Claude (Anthropic); the judgement calls
are the author's.
