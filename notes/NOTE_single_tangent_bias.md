# A single full tangent measures the fastest sector

Moved unchanged from `README.md` on 28 Oct 2026 (its history before that date is in `README.md`'s git log);
only cross-references were updated, and the `results/su2_L0_check/` numbers were put in where the text
said that run was not committed. Sections "On one ensemble" and "A large-deviation model" added 29 Oct 2026. Paths and commands are relative to the
repository root. Conventions, including the claim notation and digits, are hdym's (its top-level `README.md`).

## A single full tangent measures the fastest sector, not one sector (`su2_blocks.py`)

`ym_run.py` follows one tangent through all components. In finite time it rides whichever sector is growing fastest on
each trajectory, so its estimate is the expected **maximum** of the sectors' noisy finite-time exponents: biased upward.
The exponents agree in the infinite-time limit (by ergodicity every block of a trajectory has the same exponent), but
the bias fades slowly.

Found on 27 Oct by a same-code check of hdym's SU(2) numbers. `ym_run.py --group su2 --L0 --E 0.1 --N 1100` gave
λ_T = 1.804×10⁻³ at t = 20,000 against hdym's GPU kernel (`ym/hdym_rung1_gpu.py`, which follows **one** off-diagonal
block) at 1.543×10⁻³ at the same t, flat at 1.514–1.543 for t = 15,000–30,000 (hdym
`ym/data/results_rung1/history_E0.1_cj0.npy`). At E = 0.3 the same run matched hdym's vector ghost, which leads there,
to 0.7 % (0.016394 against 0.016506, hdym `ym/data/transverse_E0.3.npz`). The run is
`results/su2_L0_check/` (`log.txt`, `ym_run.csv`, `lam_E0.3.npz`, `lam_E0.1.npz`; committed 28 Oct 2026, 7baf44a), run
to the convergence stop: t = 7,000 at E = 0.3 and t = 47,000 at E = 0.1. Until 28 Oct this paragraph said those files
were not committed and that 1.804×10⁻³, 0.016394 and the +16.9 % / +15.5 % below had no committed log; all four are in
`log.txt`.

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
  23,000, where t^(−1/2) predicts 15.7 % (stated as 15.8 % until 28 Oct 2026, from the rounded 16.9 %). At that rate
  it is still about 12 % at t = 40,000 (15.5 % × (23,000/40,000)^½ = 11.8 %; stated as 10 % until 28 Oct 2026); the run
  measured 11.3 % there (table below).
- The one-block value is about 6 % above the GPU's at these times (1.641 against 1.543×10⁻³; about 3σ for 50
  trajectories). The initial ensembles differ (`ym_run.py`'s L = 0 projection against the GPU's Gaussian diagonal
  sampling), and the 1,100-trajectory run divided by this log's ratio (1.804/1.132 = 1.59×10⁻³) lands about 3 % above
  the GPU. No sign of a kernel problem, but not an exact match either: a larger `su2_blocks.py` run (more trajectories)
  would settle the remaining few per cent. (29 Oct 2026: both figures compare λ_T with the GPU's offset-free slope.
  With matching estimators the gap is about 1 %: "On one ensemble" below.)
- The vector ghost is slower at E = 0.1 (4.3×10⁻⁴), so the off-diagonal blocks lead there.

**The decay to t = 47,000** (`results/su2_L0_check/log.txt`, the E = 0.1 progress lines; added 28 Oct 2026). Excess of
the 1,100-trajectory full-tangent run over the GPU's one block, 1.543×10⁻³ at t = 20,000:

| t | λ_T | t^(−1/2) from t = 20,000 | λ_inc | λ_inc excess / λ_T excess |
|---|---|---|---|---|
| 20,000 | +16.9 % | (anchor) | +10.3 % | 0.61 |
| 23,000 | +15.5 % | +15.7 % | +8.6 % | 0.56 |
| 34,000 | +12.4 % | +12.9 % | +6.2 % | 0.50 |
| 40,000 | +11.3 % | +11.9 % | +5.8 % | 0.51 |
| 47,000 | +10.4 % | +11.0 % | +5.1 % | 0.49 |

- The λ_T excess falls like t^(−1/2), slightly faster than the anchor predicts.
- λ_inc, the estimator `run_low` and `run_L0` report, carries about half of λ_T's excess, not none. If the bias is an
  offset growing like √t in the cumulative log, λ_inc's excess is 2(1 − 1/√2) = 0.59 of λ_T's [est]; the measured
  0.49–0.61 is close to that.
- At the stop, λ_T − λ_inc = 8.12×10⁻⁵ (1.7034×10⁻³ against 1.6222×10⁻³) and b/t = 3.969/47,000 = 8.44×10⁻⁵: the
  excess behaves as the growing offset b.
- Part of each excess is the difference between the ensembles (`ym_run.py`'s L = 0 projection against the GPU's
  sampling), not the estimator [est]. (29 Oct 2026: measured on one ensemble in the next section, which supersedes
  this table's reading: λ_inc carries 0.57–0.74 of λ_T's excess there, and 5.8 % in absolute terms at t = 47,000.)

## On one ensemble: all 1,100 initial conditions, per sector (`su2_blocks_ensemble.py`, 29 Oct 2026)

`su2_blocks.py` run for each of `ym_run.py`'s 22 worker seeds (11 + 1000 k, 50 trajectories each), so the per-sector
tangents start from exactly the initial ensemble of `results/su2_L0_check/`, to the same t = 47,000, and are compared
at matched t with that run's full tangent (`results/su2_blocks_L0check.txt`; children's logs and npz in
`results/su2_blocks_L0check/`; 22 of 22 OK, energy drift ≤ 3.3×10⁻⁴, 26,521 s on 16 workers). Different integrators
(DOP853 against Yoshida-4), so ensembles are paired, not trajectories. "one" = mean over the three off-diagonal blocks,
"full" = `ym_run.py`'s single tangent.

**The GPU gap was the estimator.** hdym's GPU value is a least-squares slope of the log over the second half of the
run (`ym/hdym_rung1_gpu.py`, `run_energy`), offset-free like λ_inc. The 6 % and 3 % above were λ_T against that slope.
One block's λ_inc against the GPU at matched t (GPU from `history_E0.1_cj0.npy`):

| t | one block, λ_inc | GPU | ratio | difference / combined SE |
|---|---|---|---|---|
| 20,000 | (1.5544 ± 0.0078)×10⁻³ | 1.5434×10⁻³ | 1.007 | 1.4 |
| 25,000 | (1.5616 ± 0.0071)×10⁻³ | 1.5432×10⁻³ | 1.012 | 2.6 |
| 30,000 | (1.5590 ± 0.0065)×10⁻³ | 1.5414×10⁻³ | 1.011 | 2.7 |

- One block agrees with the GPU to about 1 % (λ_T would give +3.6 % at t = 20,000). No kernel problem at that level.
  The rest is the ensembles (L = 0 projection, burn-in 100, against Gaussian sampling with burn-in 300 E^{−1/4}) or the
  two offset-free estimators (two-point increment against a least-squares slope) [est]. The GPU's t = 15,000 row is its
  first, with a short window, and is not compared.

**The bias of the full tangent, same ensemble, matched t:**

| t | full / one, λ_T | full / one, λ_inc | ratio of excesses | max of 3 / one, λ_T | full / max of 3, λ_T |
|---|---|---|---|---|---|
| 5,000 | +20.6 % | +14.8 % ± 2.1 % | 0.72 | +29.5 % | 0.931 |
| 10,000 | +15.9 % | +11.0 % ± 1.3 % | 0.69 | +21.0 % | 0.958 |
| 20,000 | +12.8 % | +9.5 % ± 1.0 % | 0.74 | +15.2 % | 0.979 |
| 30,000 | +10.5 % | +6.4 % ± 0.8 % | 0.61 | +12.6 % | 0.982 |
| 34,000 | +9.8 % | +5.6 % ± 0.7 % | 0.57 | +11.8 % | 0.983 |
| 40,000 | +9.4 % | +5.8 % ± 0.7 % | 0.62 | +10.9 % | 0.986 |
| 47,000 | +9.0 % | +5.8 % ± 0.7 % | 0.65 | +10.1 % | 0.989 |

(± is the combined SE of the two λ_inc means; the λ_T ratios are tighter.)

- **At the convergence stop (t = 47,000, λ_inc·t = 76) the λ_inc that `ym_run.py` reports is 5.8 % ± 0.7 % above
  one sector's**: λ_inc 1.6222×10⁻³ against 1.5328×10⁻³ ± 0.0058×10⁻³. λ_T is 9.0 % above.
- Fits over t ≥ 10,000 give excess ∝ t^(−0.42) for λ_T and t^(−0.62) for λ_inc [grid fit]; the λ_inc excess is flat
  at 5.6–5.8 % from t = 34,000 to 47,000. The ratio of excesses, 0.57–0.74, brackets the √t-offset model's 0.59 at its
  low end only.
- The full tangent sits 1–2 % below the per-trajectory max over the three blocks for t ≥ 16,000 (0.979–0.989 [grid]),
  which logs the "within about 2 %" of 27 Oct.
- One sector has its own start-up offset: λ_T − λ_inc = 3.07×10⁻⁵ at t = 47,000 (b ≈ 1.4). And its λ_inc still falls,
  1.5544 → 1.5328×10⁻³ from t = 20,000 to 47,000 (−1.4 %), so the one-sector limit is below 1.533×10⁻³ [est].
- Vector ghost 4.333–4.357×10⁻⁴ for t ≥ 20,000.

**Consequence for SU(3).** The full-tangent λ_ghost values in `run_low` and `run_L0` carry the same kind of upward bias,
of unknown size, probably larger with more sectors. Its direction leaves "SU(3) suppresses the low-E ghost more than
SU(2)" intact, but the low-E slope (2.86 at L = 0) should not be quoted as final until the bias is measured by following
sectors separately.

29 Oct 2026, from the section above: in SU(2) at L = 0, E = 0.1, the full tangent's λ_inc at the stop is 5.8 % ± 0.7 %
too high, with three off-diagonal blocks and the vector ghost competing. That calibrates the SU(3) sector split, not
the SU(3) bias itself, which has more sectors and is likely larger [est]. If the bias differs by Δ between E = 0.1 and
0.05, the slope there moves by ln(1 + Δ)/ln 2; Δ = 6 % would move 2.86 by 0.08 [est, arithmetic only].

## A large-deviation model of the bias (`ld_bias.py`, 29 Oct 2026) [est]

A model that explains the numbers above and predicts the SU(3) bias without the sector split. It reads committed
results only (`results/ld_bias.txt`, seconds).

**Model.** Each sector k of a trajectory has cumulative log L_k(t) = λt + b + √(Dt)·g_k, the g_k standard normal and
correlated across sectors with ρ: the central-limit, quadratic-rate-function level of large-deviation theory. A single
full tangent rides the fastest sector, so its log is max_k L_k(t). With e_n and v_n the mean and variance of the
maximum of n such normals:
- λ_T(full) − λ_T(one) = e_n √(D/t);
- λ_inc(full) − λ_inc(one) = 2(1 − 1/√2) e_n √(D/t);
- var L_full(t)/t = v_n D.

The factor 2(1 − 1/√2) = 0.586 is the √t-offset ratio of "The decay to t = 47,000".

**Tested on SU(2)** (L = 0, E = 0.1, the 1,100 trajectories above):
- D = 1.70–1.86 × 10⁻³ for t = 5,000–47,000, nearly flat (diffusive); ρ = 0.00–0.13 between blocks.
- "The full tangent's log is the maximum over sectors" holds. max_k L_k gives λ_inc within 0.3–1.2 % of the full
  tangent's at every t (47,000: 1.6265 against 1.6222 × 10⁻³). The vector ghost never leads (0 of 1,100).
- The Gaussian model gives the max-of-three λ_T within 0.1–0.3 % (47,000: 1.7199 against 1.7220 × 10⁻³).
- At the stop it gives the λ_inc bias as 9.20 × 10⁻⁵ against the measured 8.94 × 10⁻⁵. From the full tangent alone,
  with D = var L_full/(t v₃), the route open for SU(3), it gives 9.61 × 10⁻⁵.

**Applied to SU(3)**, D from each run's full-tangent variance, ρ = 0.133, n = 2 … 8 competing sectors:
- `run_L0`: predicted λ_inc bias 2.3–7.3 % at E = 0.3 and 2.8–8.9 % at E = 0.1 and 0.05.
  - It is nearly the same at every E: the convergence stop fixes λt and the relative SEM, and with them √(D/t)/λ.
  - So it cancels in the slopes: 0.1 → 0.05 stays 2.86 for every n, and 0.3 → 0.1 goes from 2.59 to 2.60.
  - The SU(3)/SU(2) ratios carry the few per cent.
  - The model's assumption holds there: var L/t is flat in t (E = 0.1: 2.96, 2.85, 2.99, 3.17 × 10⁻⁵ at t = 9,000 …
    77,000; E = 0.05: 4.77 … 4.35 × 10⁻⁶).
- `run_low` (generic L): **the model does not apply.** var L/t grows with t (E = 0.1: 6.9 → 24.5 × 10⁻⁵ from t = 8,000
  to 69,000), because trajectories differ in their true rates (the |L| hidden parameter), so D is not a diffusion
  constant. The 4–34 % the script prints for `run_low` are not predictions.

**Reading.** If the model carries over to SU(3) at L = 0, the full-tangent bias moves the L = 0 absolute values by a
few per cent and leaves the slopes. The sector split (Open) then confirms the L = 0 slopes and remains the measurement
for generic L and the |L| figures. What would break it: SU(3) sectors with clearly unequal rates, or strong
correlations between them. The judgement is the author's.

```
python ld_bias.py > results/ld_bias.txt
python su2_blocks.py --E 0.1 --tmax 18000
python ym_run.py --group su2 --E 0.3 0.1 --N 1100 --workers 22 --tmax 100000 --L0 --target-rel 0.005 --out results/su2_L0_check
python su2_blocks_ensemble.py --E 0.1 --tmax 47000 --workers 16 > results/su2_blocks_L0check.txt
```

The `ym_run.py` line is reconstructed from the START line of `results/su2_L0_check/log.txt` (the run wrote to
`C:\temp\su2check` and was copied in).
