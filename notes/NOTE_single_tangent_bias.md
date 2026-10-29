# A single full tangent measures the fastest sector

Moved unchanged from `README.md` on 28 Oct 2026 (its history before that date is in `README.md`'s git log);
only cross-references were updated, and the `results/su2_L0_check/` numbers were put in where the text
said that run was not committed. Paths and commands are relative to the repository root. Conventions, including the claim notation and digits, are hdym's (its
top-level `README.md`).

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
  would settle the remaining few per cent.
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
  sampling), not the estimator [est]. `su2_blocks_ensemble.py` (added 28 Oct 2026) runs `su2_blocks.py` on all 1,100
  initial conditions of this run to t = 47,000 and compares at matched t on one ensemble; its output is
  `results/su2_blocks_L0check.txt` once run.

**Consequence for SU(3).** The full-tangent λ_ghost values in `run_low` and `run_L0` carry the same kind of upward bias,
of unknown size, probably larger with more sectors. Its direction leaves "SU(3) suppresses the low-E ghost more than
SU(2)" intact, but the low-E slope (2.86 at L = 0) should not be quoted as final until the bias is measured by following
sectors separately.

```
python su2_blocks.py --E 0.1 --tmax 18000
python ym_run.py --group su2 --E 0.3 0.1 --N 1100 --workers 22 --tmax 100000 --L0 --target-rel 0.005 --out results/su2_L0_check
python su2_blocks_ensemble.py --E 0.1 --tmax 47000 --workers 22 > results/su2_blocks_L0check.txt
```

The `ym_run.py` line is reconstructed from the START line of `results/su2_L0_check/log.txt` (the run wrote to
`C:\temp\su2check` and was copied in).
