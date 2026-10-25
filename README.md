# su3/ — the colour-diagonal sector inside SU(3): the isospin embedding, the doublet, and an escape that is not the ghost's

Work of 24 Oct 2026, on the branch `su3-isospin` only; `main` is the SU(2) project and does not contain this folder.
The question is the paper's "move up to another matrix group": embed the 12-d system of `ym/` in homogeneous SU(3)
Lee–Wick Yang–Mills and ask what the new directions do. Units g = M = 1 and the conventions of `ym/hdym_derive.py`
(L = −¼F² + (1/2M²)(DF)², F^a_ij = f^abc A_i^b A_j^c). The code imports `ym/hdym_core.py` read-only and changes
nothing outside this folder.

## Results

**The embedding (exact).** With A_i^a = δ_i^a q_i for a ∈ {1, 2, 3} (Gell-Mann λ₁, λ₂, λ₃, the isospin subalgebra),
the SU(3) potential and gradient equal `ym/hdym_core.V` and `gradV` to 10⁻¹⁴, and the gradient has no component off
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
per-trajectory channel sub-stepping of `ym/hdym_core.ym_step_substepped`, no trajectory dropped, max |ΔE|/E ≤ 5.5 × 10⁻⁴
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

- The full-SU(3) ghost exponent above (all pieces are in `su3_core.py`).
- The ghost lines use the naive operator ẅ = −(1 + M)w, the level of the paper's diagonal ghost; the Lee–Wick
  Gauss-law reduction of the doublet ghost (cf. `rabbit_holes/ym/` for the SU(2) off-diagonal block) is not done.
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
| `results/` | the logs every number above is taken from |

Run commands (two CPU cores, about 18, 11 and 23 minutes, and 4 minutes for the last):

```
python su3/check_embedding.py
python su3/transverse_su3.py --E 1.0 0.5 --N 400 --dt 0.005 --tmax 2000 --t-burn 100 --seed 1
python su3/transverse_su3.py --E 0.3 --N 400 --dt 0.005 --tmax 4000 --t-burn 100 --seed 2
python su3/doublet_physical.py --E 1.0 --N 300 --tmax 1000
```

The code and the derivations were developed in an interactive session with Claude (Anthropic); the judgement calls
are the author's.
