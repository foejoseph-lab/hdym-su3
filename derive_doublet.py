"""derive_doublet.py -- closed form of the 12x12 doublet (lambda_4..7) Hessian block on the
isospin embedding, by sympy; checked against su3_core.hessV numerically."""
import numpy as np, sympy as sp
import su3_core as s3

x, y, z = sp.symbols('x y z', real=True)
q = [x, y, z]
fS = sp.MutableDenseNDimArray.zeros(8, 8, 8)
half, r3 = sp.Rational(1, 2), sp.sqrt(3) / 2
for (a, b, c), v in {(1,2,3):1,(1,4,7):half,(1,5,6):-half,(2,4,6):half,(2,5,7):half,
                     (3,4,5):half,(3,6,7):-half,(4,5,8):r3,(6,7,8):r3}.items():
    a, b, c = a-1, b-1, c-1
    for (i, j, k), s in (((a,b,c),1),((b,c,a),1),((c,a,b),1),((b,a,c),-1),((a,c,b),-1),((c,b,a),-1)):
        fS[i, j, k] = s * v
d = sp.symbols('d0:24', real=True)              # perturbation, flat (i*8+a)
A = [[(q[i] if a == i else 0) + d[i*8+a] for a in range(8)] for i in range(3)]
Vexpr = 0
for i in range(3):
    for j in range(3):
        for a_ in range(8):
            Fa = sum(fS[a_, b, c] * A[i][b] * A[j][c] for b in range(8) for c in range(8) if fS[a_, b, c] != 0)
            Vexpr += Fa**2
Vexpr = sp.Rational(1, 4) * Vexpr
idx = s3.block_indices(s3.DOUBLET)
sub0 = {di: 0 for di in d}
H = sp.Matrix(12, 12, lambda r, c: sp.simplify(sp.diff(Vexpr, d[idx[r]], d[idx[c]]).subs(sub0)))
labels = [f'A{i+1}^{a+1}' for i in range(3) for a in s3.DOUBLET]
print('doublet block, rows/cols =', labels)
sp.pprint(H, wrap_line=False)
# eigenvalues
print('\neigenvalues:')
for ev, m in H.eigenvals().items():
    print(f'  mult {m}:', sp.factor(ev))
# numeric check
f_num = sp.lambdify((x, y, z), H, 'numpy')
rng = np.random.default_rng(3)
err = 0
for _ in range(50):
    qq = rng.standard_normal(3)
    Hn = s3.hessV(s3.embed(qq))[np.ix_(idx, idx)]
    err = max(err, np.abs(np.array(f_num(*qq), float) - Hn).max())
print('\nmax |sympy - numeric| =', err)
