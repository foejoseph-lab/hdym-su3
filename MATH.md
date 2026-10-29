# Math in this repository: an index

Started 28 Oct 2026. A pointer list, not a home: each result stays described in its note in `notes/`, and files stay where
they are. Tags as in hdym's top-level README, "Claim notation"; [method] marks a technique, not a claim. Numerical
measurements of the model are not listed. hdym's own index is its `MATH.md`.

| statement | tag | where |
|---|---|---|
| The isospin embedding A_i^a = δ_i^a q_i (a ∈ {1, 2, 3}) is an invariant sector of homogeneous SU(3) Yang–Mills; on it the 24 × 24 Hessian is block-diagonal, adjoint (9) ⊕ doublet λ₄…λ₇ (12) ⊕ singlet λ₈ (3), with the singlet block identically zero | numerical checks to 10⁻¹⁴ (11 checks) | `check_embedding.py`, `results/check_embedding.txt`; `notes/NOTE_embedding_doublet.md`, "The embedding" |
| The doublet block is four identical copies of one 3 × 3 matrix M(q) = ¼ diag(y² + z², z² + x², x² + y²) + ¾ [xy, xz, yz off-diagonal]; gauge mode M q = ∇V; Gauss charge conserved | [exact] (sympy) | `derive_doublet.py`, `results/derive_doublet.txt`; `notes/NOTE_embedding_doublet.md`, "The doublet block" |
| On SU(3)'s two isotropic orbits (λ = 1 isospin, λ = 1/2 real SO(3)) the naive ghost splits into one Lamé equation per Hessian eigen-family, n(n + 1) = 2μ/α, with the ghost mass only in the spectral parameter | [exact] reduction; windows numerical, checked against direct 24-d Floquet at 11 energies | `lame_isotropic.py`, `results/lame_isotropic.txt`; `notes/NOTE_lame_isotropic.md` |
| so(4) = su(2) ⊕ su(2) with coupling √2 per half: a half-loaded SO(4) state at energy E is a generic SU(2) state at 2E | [exact] (structure constants); numerical check of the dynamics | `groups.py`, `check_so4.py`, `results/check_so4.txt`; `notes/NOTE_so4.md` |
| A single tangent through all sectors estimates the expected maximum of the sectors' finite-time exponents, biased upward; the bias fades like t^(−1/2) | [method] (estimator fact); measured on SU(2) | `su2_blocks.py`, `results/su2_blocks.txt`; `notes/NOTE_single_tangent_bias.md` |
