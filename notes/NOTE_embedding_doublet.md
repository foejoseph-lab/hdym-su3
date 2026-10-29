# The isospin embedding, the doublet block and the escape from the sector

Moved unchanged from `README.md` on 28 Oct 2026 (its history before that date is in `README.md`'s git log);
only cross-references were updated. Paths and commands are relative to the repository root. Conventions, including the claim notation and digits, are hdym's (its
top-level `README.md`).

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
25–27 Oct: `NOTE_full_su3.md`, "Long runs", results there.

## Run commands

Run commands (one process each; logged wall times: 18 + 11 min for the first transverse command (E = 1, 0.5), 23 min
for the second, 4 min for the last):

```
python check_embedding.py
python transverse_su3.py --E 1.0 0.5 --N 400 --dt 0.005 --tmax 2000 --t-burn 100 --seed 1
python transverse_su3.py --E 0.3 --N 400 --dt 0.005 --tmax 4000 --t-burn 100 --seed 2
python doublet_physical.py --E 1.0 --N 300 --tmax 1000
```
