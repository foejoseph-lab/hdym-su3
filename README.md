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

Each result is described in full in one note in `notes/` and stated here in one line; numbers are from the committed
logs named in the note. Naive ghost operator ẅ = −(1 + Hess V)w throughout.

## Results

- **The isospin embedding** of hdym's SU(2) system is an invariant sector of SU(3); on it the Hessian is adjoint ⊕
  doublet ⊕ singlet, the singlet identically zero (11 checks to 10⁻¹⁴). `notes/NOTE_embedding_doublet.md`
- **The doublet block** is four copies of one 3 × 3 matrix M(q) = ¼ diag(y² + z², …) + ¾ [xy, …] [exact].
  `notes/NOTE_embedding_doublet.md`
- **Pure Yang–Mills leaves the sector** at λ_esc = 0.1025 E^{1/4} (E = 1, 0.5, 0.3), and the escape is physical
  (gauge mode removed). The doublet ghost is the weaker ghost: 0.44, 0.30, 0.25 of SU(2)'s.
  `notes/NOTE_embedding_doublet.md`
- **SO(4)** = two decoupled SU(2) systems checks the group-general code: 14 checks pass. `notes/NOTE_so4.md`
- **Full SU(3)**: λ_YM/E^{1/4} ∈ [0.26208, 0.26216] (generic L) and ∈ [0.27520, 0.27559] (L = 0); ghost local slopes at
  L = 0 1.85, 1.92, 2.59, 2.86, not final (full tangent); against SU(2) at L = 0 the ratio is 0.275 at E = 0.3 and 0.171
  at E = 0.1, so SU(3) suppresses the low-E ghost at least this much [est]. `notes/NOTE_full_su3.md`
- **|L| is a hidden parameter**: in `run_low` its rank correlation with λ_YM is −0.912, −0.986, −0.997 at E = 0.3, 0.1,
  0.05. `notes/NOTE_full_su3.md`, "Angular momentum"
- **Isotropic orbits** reduce to one Lamé equation per Hessian eigen-family [exact]; the doublet is the only family
  along which pure Yang–Mills is unstable; the n = 1 windows need the exact operator. `notes/NOTE_lame_isotropic.md`
- **A single full tangent measures the fastest sector**, biased upward, the bias fading slowly. SU(2), L = 0,
  E = 0.1, one ensemble of 1,100: at t = 47,000 the full tangent's λ_inc is 5.8 % ± 0.7 % above one sector's (λ_T
  9.0 %). One sector agrees with hdym's GPU to about 1 %. `notes/NOTE_single_tangent_bias.md`

## Notes

| note | what it is | status |
|---|---|---|
| `notes/NOTE_embedding_doublet.md` | the embedding, the doublet block, the four transverse exponents, the escape from the sector | closed; Gauss-law reduction of the doublet ghost open |
| `notes/NOTE_so4.md` | SO(4) as a should-match check | closed |
| `notes/NOTE_full_su3.md` | the first scan (`su3_full.py`), the long runs (`ym_run.py`), angular momentum, the results table | runs done; slopes wait for the sector split |
| `notes/NOTE_lame_isotropic.md` | the naive ghost on the two isotropic orbits as Lamé equations | done for the naive operator |
| `notes/NOTE_single_tangent_bias.md` | the max-over-sectors bias of one tangent, measured on SU(2) | SU(2) calibrated on one ensemble (29 Oct); SU(3) sector split open |

The notes were the sections of this README until 28 Oct 2026, moved unchanged; the history of their text before that
date is in this file's git log.

## Open

- The full-SU(3) ghost exponent above: measured 25–27 Oct (`notes/NOTE_full_su3.md`, "Long runs", results there).
  Open: the size of the full-tangent bias in SU(3) (`notes/NOTE_single_tangent_bias.md`), which needs the sectors
  followed separately. (Until 28 Oct this item read "production runs to be done".)
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
| `su3_full.py` | **record (frozen)** of the first scan; generic 24-component SU(3) YM on the Gauss shell, with λ_YM and the full 24 × 24 ghost λ_ghost; `--backend numpy --workers K` (K cores) or `--backend cupy` (GPU); `--selftest` |
| `groups.py` | structure constants for su2, su3, so4 in one convention, and the so(4) = su(2) ⊕ su(2) basis |
| `ym_group.py` | **superseded by `ym_run.py`**, kept because `check_so4.py` and its log depend on it; `su3_full.py` with the group as a parameter (`--group su2 / su3 / so4`); `--group su3` reproduces `su3_full.py` exactly |
| `check_so4.py` | SO(4) as a should-match check of the group-general code (14 checks) |
| `ym_run.py` | **live engine**; the long-run engine: `su3_full.py`'s physics for `--group su2 / su3 / so4`, with checkpoint/`--resume`, the offset-free estimator λ_inc and a convergence stop; `--selftest` |
| `su2_blocks.py` | **kept here by decision (28 Oct 2026)**: SU(2)-only, but it is the test of this repository's own `ym_run.py` that any later review needs; an exception to the placement rule (SU(2) work goes in hdym). SU(2) at L = 0 with an independent integrator (DOP853) and the naive ghost split by sector: one off-diagonal block vs the max over blocks that a single full tangent measures |
| `su2_blocks_ensemble.py` | `su2_blocks.py` on all 1,100 initial conditions of `results/su2_L0_check/` (ym_run's 22 worker seeds), tabulated at matched t against that run's full tangent, λ_T and λ_inc; `--resume`, `--combine-only` (added 28 Oct 2026) |
| `lame_isotropic.py` | the naive ghost on SU(3)'s two isotropic orbits (λ = 1 isospin, λ = 1/2 real SO(3)) reduced to Lamé equations per Hessian eigen-family; windows, pure-YM limits, direct 24-d Floquet cross-check |
| `L_correlation.py` | per-trajectory exponents against spatial angular momentum \|L\|, recomputed from a finished run's seeds (no rerun) |
| `notes/` | the results in full, one note per topic (table above) |
| `MATH.md` | index of the exact results and methods, with pointers |
| `results/` | the logs every number in this README and in `notes/` is taken from, except those marked as not logged or not committed |

The three engines share their physics: `su3_full.py` is the record, `ym_group.py` the group-general step that
`check_so4.py` tests, `ym_run.py` the one to use. The explanation of the two tangents and the naive operator is in
`notes/NOTE_full_su3.md`, "Full SU(3): the first scan"; the scripts' docstrings repeat it (28 Oct 2026: listed for
consolidation, not changed).

The code and the derivations were developed in an interactive session with Claude (Anthropic); the judgement calls
are the author's.
