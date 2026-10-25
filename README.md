# hdym-su3 — Lee–Wick Yang–Mills mechanics moved up to SU(3): the isospin embedding, the doublet, and an escape that is not the ghost's

A separate project from [hdym](https://github.com/foejoseph-lab/hdym) (the SU(2) paper, *The Lee–Wick ghost of
Yang–Mills mechanics is unstable at every energy*), started 24 Oct 2026 as hdym's `su3/` folder and moved here with its
history so that the SU(2) project stays SU(2). The question is the paper's "move up to another matrix group": embed
hdym's 12-d colour-diagonal SU(2) system in homogeneous SU(3) Lee–Wick Yang–Mills and ask what the new directions do.
Units g = M = 1 and hdym's conventions (L = −¼F² + (1/2M²)(DF)², F^a_ij = f^abc A_i^b A_j^c). The SU(2) model is
`hdym_core.py`, a verbatim copy of hdym's `hdym_core.py` at commit 1d65e30 (header says so); this repository needs
nothing else from hdym. Python ≥ 3.10 with numpy and scipy (sympy for `derive_doublet.py`, CuPy for GPU runs).

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
  for E ≥ 2 (SU(2): E ≥ ½); deep in a channel min eig M ≈ −2(y² + z²) against −3(y² + z²) for SU(2).

**Transverse exponents (measured).** All four on the same pure-YM trajectories, N = 400, dt = 0.005 with the
per-trajectory channel sub-stepping of `hdym_core.ym_step_substepped`, no trajectory dropped, max |ΔE|/E ≤ 5.5 × 10⁻⁴
(`transverse_su3.py`; `results/transverse_E1_E05.txt`, `results/transverse_E03.txt`):

| E | in-sector YM chaos | doublet, pure YM | SU(2) ghost | doublet ghost |
|---|---|---|---|---|
| 1.0 | 0.3749 ± 0.0020 | 0.1025 ± 0.0008 | 0.0884 ± 0.0006 | 0.0387 ± 0.0003 |
| 0.5 | 0.3162 ± 0.0015 | 0.0864 ± 0.0007 | 0.0305 ± 0.0002 | 0.0091 ± 0.0001 |
| 0.3 | 0.2767 ± 0.0011 | 0.0758 ± 0.0005 | 0.0165 ± 0.0001 | 0.00408 ± 0.00004 |

- The doublet column divided by E^{1/4} is 0.1024, 0.1028, 0.1025: **the pure-YM escape from the sector is
  λ_esc = 0.1025 E^{1/4}**, about 27 % of in-sector chaos (0.375 E^{1/4}), as the scale-free quartic requires.
- It is physical: starting on G = 0 and removing the gauge mode at every renormalisation gives 0.1047 and 0.1044
  against 0.1050 unconstrained, ± 0.0012, with G conserved to 10⁻¹² (`doublet_physical.py`,
  `results/doublet_physical_E1.txt`, E = 1, N = 300, tmax = 1000).
- λ_esc / (SU(2) ghost) = 1.16, 2.84, 4.59 at E = 1, 0.5, 0.3; if the ghost reaches the paper's 0.0120 E^{9/4}
  at low E the ratio grows like E⁻², about 10⁵ at E = 0.008 (an extrapolation, not a measurement).
- The doublet ghost is the weaker ghost: 0.44, 0.30, 0.25 of the SU(2) ghost at E = 1, 0.5, 0.3.
- The ghost local slopes here are 1.2–2.1, not 9/4: E ≥ 0.3 is not the paper's asymptotic regime, and nothing here
  tests the E^{9/4} law.

**Reading.** In SU(2) the colour-diagonal sector is protected by a conserved quantity; inside SU(3) it is not, and a
generic trajectory leaves it at 0.1025 E^{1/4} — not a blow-up (pure-YM energy is bounded below) but a wandering into
the full homogeneous SU(3) phase space. The SU(3) analogue of the paper's measurement is therefore the ghost's
transverse exponent about generic SU(3) trajectories, with the full 24 × 24 operator 1 + Hess V_SU(3) — not yet done.

## Open

- The full-SU(3) ghost exponent above: code ready (`su3_full.py`, self-test passes), production runs to be done on the
  author's machine — see "Full SU(3): run plan" below. No full-SU(3) numbers are claimed here until those logs exist.
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
| `results/` | the logs every number above is taken from |

Run commands (two CPU cores, about 18, 11 and 23 minutes, and 4 minutes for the last):

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
  smaller half ≥ 9 %) — no exchange between the halves, only integration error (16 × smaller per halving of dt);
- half-loaded SO(4) at E = 0.5 vs generic SU(2) at E = 1, 64 trajectories, t = 30: per-trajectory log-growth of the
  YM tangent agrees to 2 × 10⁻⁹ (growth up to e^14.5), of the ghost tangent to 4 × 10⁻¹⁰, background to 3 × 10⁻¹⁰.

```
python check_so4.py
```

## Full SU(3): run plan

`su3_full.py` evolves generic homogeneous SU(3) Yang–Mills (24 components, initial data projected onto the Gauss shell
G = 0 and scaled to energy E exactly) with two 24-d tangent vectors: the YM tangent (λ_YM, scale-free, so ∝ E^{1/4})
and the ghost w'' = −(1 + Hess V)w (λ_ghost, the SU(3) analogue of the paper's transverse exponent, naive operator).
Self-test: V, ∇V, Hess V agree with `su3_core.py` to 10⁻¹⁴; energy and Gauss conserved; the embedded SU(2) sector is
invariant under the 24-d flow. dt = 0.01 and 0.005 agree within errors on identical trajectories.

From the repository folder, in order (each step tells you whether the next is worth it):

```
python su3_full.py --selftest
python su3_full.py --E 1.0 --N 2200 --tmax 300 --workers 22 --out results/full_cpu_timing
python su3_full.py --E 1.0 --N 20000 --tmax 300 --backend cupy --out results/full_gpu_timing
```

The last line needs CuPy (already there if hdym's `ym/hdym_gpu.py --backend cupy` runs) and is the first time the cupy path runs — it was written for the GPU but tested only through numpy. Compare its `wall=` with the CPU line's and use the faster
backend below. Production scan (CPU form shown; for the GPU replace `--workers 22` by `--backend cupy` and raise N):

```
python su3_full.py --E 1.0 0.5 0.3 --N 4400 --tmax 4000 --workers 22 --out results/full_scan
python su3_full.py --E 0.2 0.1 --N 4400 --tmax 20000 --t-block 1000 --workers 22 --out results/full_scan
```

Each energy appends a row to `results/full_scan/su3_full.csv` and writes `lam_E<E>.npz` (per-trajectory exponents and the
running-mean history: a λ still drifting in the progress lines needs a longer tmax). The ghost needs λ_ghost·tmax ≳ 20
to be trustworthy; if the lower energies come out below ~10⁻³, double tmax before reading anything into the law.

The code and the derivations were developed in an interactive session with Claude (Anthropic); the judgement calls
are the author's.
