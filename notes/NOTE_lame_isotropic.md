# Isotropic orbits: the naive ghost as Lamé equations

Moved unchanged from `README.md` on 28 Oct 2026 (its history before that date is in `README.md`'s git log);
only cross-references were updated. Paths and commands are relative to the repository root. Conventions, including the claim notation and digits, are hdym's (its
top-level `README.md`).

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
