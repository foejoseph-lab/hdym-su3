# SO(4) as a should-match check of the group-general code

Moved unchanged from `README.md` on 28 Oct 2026 (its history before that date is in `README.md`'s git log);
only cross-references were updated. Paths and commands are relative to the repository root. Conventions, including the claim notation and digits, are hdym's (its
top-level `README.md`).

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
