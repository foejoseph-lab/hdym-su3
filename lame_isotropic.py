"""
lame_isotropic.py -- the naive Lee-Wick ghost on SU(3)'s two isotropic orbits, reduced to Lame equations.

The isotropic backgrounds of SU(N)-natural inflation (Murata, Fujita, Kobayashi, arXiv:2211.09489) are
A_i = s(t) T_i with the T_i spanning an SU(2) subalgebra; SU(3) has two:
    lambda = 1    isospin   (T_i on lambda_1, lambda_2, lambda_3)      V = 3/2 s^4,  s'' = -2 s^3
    lambda = 1/2  real SO(3) (T_i on lambda_2, lambda_5, lambda_7)     V = 3/8 s^4,  s'' = -s^3/2
Each is a straight-line orbit of a quartic potential: s = A cn(sqrt(alpha) A t | k^2 = 1/2), E = c A^4. Since
Hess V(s Ahat) = s^2 Hess V(Ahat), the 24 directions split along the fixed eigenvectors of Hess V(Ahat)
(eigenvalue mu), and the naive ghost w'' = -(1 + Hess V) w (units g = M = 1; the operator of ym_run.py) splits into
    w'' + (1 + mu s^2) w = 0   ->   Lame:  w_uu + (h - n(n+1) k^2 sn^2 u) w = 0,
    u = sqrt(alpha) A t,  n(n+1) = 2 mu / alpha,  h = mu/alpha + 1/(alpha A^2)   (h -> mu/alpha is the pure-YM limit).
Integer n is a finite-gap potential (exactly n gaps); otherwise there are infinitely many, thinning fast.

Prints, per embedding and per eigen-family: mu, multiplicity, colour content, n, whether pure YM is unstable along
the family (its no-ghost limit), and the E-windows where the ghost is unstable (gaps of the Lame discriminant found
on an h-grid and refined by bisection; h up to mu/alpha + HMAX, i.e. E down to c/(alpha HMAX)^2). Checks: the
sector is invariant; multiplicities sum to 24; the radial (n = 2) window is the SU(2) window rescaled, exactly
(1/8, 1/6) and (1/2, 2/3); the n = 1 window starts exactly at E = 3/2 and 6; a direct 24-dimensional Floquet
computation of the naive ghost along the integrated orbit agrees with the family prediction at test energies.

CAVEAT (stated in the log): the naive operator. The radial family does not involve the Lee-Wick Gauss term (c_j)
and stands as computed; families containing off-diagonal-type directions need the exact operator before any
claim (hdym rabbit_holes/ym/notes/RH5_nonplanar.md: on SU(2)'s x = y = z orbit the exact c_j = 1 off-diagonal
block shows no instability up to E = 3, where the naive operator's n = 1 family is unstable for E > 3/2).

    python lame_isotropic.py > results/lame_isotropic.txt
Single core, a few minutes. Added 2026-10-27.
"""
import os, sys, time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import ellipj, ellipk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ym_run
from groups import structure_constants

T0 = time.time()
K2 = 0.5
PER = 2 * ellipk(K2)              # period of sn^2 in u
HMAX = 60.0
OK = True
f = structure_constants('su3')
m = ym_run.Model(f)


def rep(name, err, tol):
    global OK
    good = bool(err <= tol)
    OK &= good
    print(f"[{'OK ' if good else 'FAIL'}] {name}: {err:.2e}", flush=True)


def trace(h, nn1):
    """Trace of the Lame monodromy over one period of sn^2."""
    def rhs(u, y):
        q = h - nn1 * K2 * ellipj(u, K2)[0] ** 2
        return [y[1], -q * y[0], y[3], -q * y[2]]
    y = solve_ivp(rhs, (0, PER), [1, 0, 0, 1], rtol=1e-11, atol=1e-12, method='DOP853').y[:, -1]
    return y[0] + y[3]


def unstable(h, nn1):
    return abs(trace(h, nn1)) > 2 + 1e-9


def bisect(a, b, nn1):
    """a stable, b unstable (or the reverse): return the edge."""
    ua = unstable(a, nn1)
    for _ in range(45):
        c = 0.5 * (a + b)
        if unstable(c, nn1) == ua:
            a = c
        else:
            b = c
    return 0.5 * (a + b)


def gaps(nn1, h0, npts=3000):
    """Unstable h-intervals in (h0, h0 + HMAX], refined; an interval touching h0 is reported open at h0."""
    hs = h0 + np.concatenate([np.geomspace(1e-6, 1.0, npts // 3, endpoint=False), np.linspace(1.0, HMAX, 2 * npts // 3)])
    us = np.array([unstable(h, nn1) for h in hs])
    out, i = [], 0
    while i < len(hs):
        if us[i]:
            j = i
            while j + 1 < len(hs) and us[j + 1]:
                j += 1
            lo = h0 if i == 0 else bisect(hs[i - 1], hs[i], nn1)
            hi = bisect(hs[j], hs[j + 1], nn1) if j + 1 < len(hs) else hs[j]
            peak = max(abs(trace(h, nn1)) for h in np.linspace(lo, hi, 7)[1:-1]) if hi > lo else abs(trace(lo, nn1))
            out.append((lo, hi, peak))
            i = j + 1
        else:
            i += 1
    return out


def E_of(h, h0, alpha, c):
    return np.inf if h <= h0 else c / (alpha * (h - h0)) ** 2


def hessian(Ahat):
    H = np.zeros((24, 24))
    for k in range(24):
        e = np.zeros((1, 3, 8)); e.flat[k] = 1
        H[:, k] = m.hvp(Ahat[None], m.F(Ahat[None]), e).ravel()
    return 0.5 * (H + H.T)


def floquet_maxmu(E, Ahat, c, alpha):
    """Direct check: the 24-d naive ghost w'' = -(1 + Hess V) w along the integrated orbit, one orbit period."""
    A0 = (E / c) ** 0.25
    T = 4 * ellipk(K2) / (np.sqrt(alpha) * A0)

    def rhs(t, Y):
        A = Y[:24].reshape(1, 3, 8)
        Fs = m.F(A)
        W = Y[48:].reshape(48, 48)
        X, Vv = W[:24], W[24:]
        HX = np.stack([m.hvp(A, Fs, X[:, k].reshape(1, 3, 8)).ravel() for k in range(48)], axis=1)
        return np.concatenate([Y[24:48], -m.grad(A, Fs).ravel(), np.vstack([Vv, -(X + HX)]).ravel()])
    Y0 = np.concatenate([(A0 * Ahat).ravel(), np.zeros(24), np.eye(48).ravel()])
    sol = solve_ivp(rhs, (0, T), Y0, rtol=1e-10, atol=1e-11, method='DOP853')
    M = sol.y[48:, -1].reshape(48, 48)
    return np.abs(np.linalg.eigvals(M)).max(), np.abs(sol.y[:48, -1] - Y0[:48]).max() / A0


print('# START lame_isotropic', flush=True)
print('naive ghost w\'\' = -(1 + Hess V) w on the two isotropic SU(3) orbits, as Lame equations (k^2 = 1/2)')
print(f'gap search: h in (mu/alpha, mu/alpha + {HMAX:g}]; thinner tongues below the grid resolution are not seen')
EMB = {'lambda=1 (isospin, colours 1,2,3)': [0, 1, 2], 'lambda=1/2 (real SO(3), colours 2,5,7)': [1, 4, 6]}
RADIAL_EXPECT = {'lambda=1 (isospin, colours 1,2,3)': (1 / 8, 1 / 6), 'lambda=1/2 (real SO(3), colours 2,5,7)': (1 / 2, 2 / 3)}
N1_EXPECT = {'lambda=1 (isospin, colours 1,2,3)': 1.5, 'lambda=1/2 (real SO(3), colours 2,5,7)': 6.0}
predictions = {}
for name, cols in EMB.items():
    Ahat = np.zeros((3, 8))
    for i, a in enumerate(cols):
        Ahat[i, a] = 1.0
    Fs = m.F(Ahat[None])[0]
    rep(f'{name}: sector invariant (F outside the sector)', np.abs(np.delete(Fs, cols, axis=2)).max(), 0)
    c = float(m.V(Ahat[None])[0])
    alpha = 4 * c / 3
    ev, U = np.linalg.eigh(hessian(Ahat))
    vals = np.round(ev, 9)
    uniq, mult = np.unique(vals, return_counts=True)
    rep(f'{name}: multiplicities sum to 24', abs(mult.sum() - 24), 0)
    print(f'\n=== {name}:  V = {c:g} s^4,  s\'\' = -{alpha:g} s^3,  E = {c:g} A^4')
    fams = []
    for mu, k in zip(uniq, mult):
        sel = np.abs(vals - mu) < 1e-7
        w = (U[:, sel].reshape(3, 8, -1) ** 2).sum(axis=(0, 2)) / k
        content = ' '.join(f'{a + 1}:{w[a]:.2f}' for a in range(8) if w[a] > 1e-9)
        nn1 = 2 * mu / alpha
        disc = 1 + 4 * nn1
        n = f'{(-1 + np.sqrt(disc)) / 2:.4f}' if disc >= 0 else f'-1/2 + {np.sqrt(-disc) / 2:.4f} i'
        h0 = mu / alpha
        ym = abs(trace(h0, nn1)) - 2
        g = gaps(nn1, h0)
        wins = [(E_of(hi, h0, alpha, c), E_of(lo, h0, alpha, c), pk) for lo, hi, pk in g]
        fams.append((mu, nn1, wins))
        print(f'  mu = {mu:+9.6f}  x{k:<2d}  n(n+1) = {nn1:+8.4f}  n = {n:>18}   colours  {content}')
        print(f'      pure-YM limit (no ghost): |tr| - 2 = {ym:+.2e}  -> '
              f'{"UNSTABLE: the orbit itself is unstable along this family" if ym > 1e-8 else "stable / marginal"}')
        if not wins:
            print('      ghost: no instability window found')
        for Elo, Ehi, pk in wins:
            mu_max = (pk / 2 + np.sqrt(max(pk * pk / 4 - 1, 0))) if pk > 2 else 1.0
            print(f'      ghost unstable for E in ({Elo:.6g}, {Ehi:.6g})   peak |multiplier| per sn^2-period = {mu_max:.6f}')
        if abs(nn1 - 6) < 1e-9:
            lo, hi = RADIAL_EXPECT[name]
            rep(f'{name}: radial (n = 2) window == ({lo:g}, {hi:g}) exactly (rel; edges by bisection of a numerical trace)',
                max(abs(wins[0][0] / lo - 1), abs(wins[0][1] / hi - 1)) if len(wins) == 1 else 1.0, 1e-7)
        if abs(nn1 - 2) < 1e-9:
            rep(f'{name}: n = 1 window starts at E = {N1_EXPECT[name]:g} and is open above (rel)',
                abs(wins[0][0] / N1_EXPECT[name] - 1) if len(wins) == 1 and np.isinf(wins[0][1]) else 1.0, 1e-7)
        if abs(nn1 - round(nn1)) < 1e-9 and round(nn1) in (0, 2, 6, 12, 20):
            nint = int(round((-1 + np.sqrt(1 + 4 * round(nn1))) / 2))
            rep(f'{name}: integer n = {nint} has at most n reachable gaps (found {len(wins)})', max(len(wins) - nint, 0), 0)
    predictions[name] = (Ahat, c, alpha, fams)

print('\n=== direct 24-d Floquet of the naive ghost along the integrated orbit vs the family prediction')
TESTS = {'lambda=1 (isospin, colours 1,2,3)': (0.05, 0.15, 0.4, 0.63, 1.0, 2.0),
         'lambda=1/2 (real SO(3), colours 2,5,7)': (0.0975, 0.3, 0.6, 1.0, 2.5)}
for name, Es in TESTS.items():
    Ahat, c, alpha, fams = predictions[name]
    for E in Es:
        pred = any(lo < E < hi for _, _, wins in fams for lo, hi, _ in wins)
        mx, clo = floquet_maxmu(E, Ahat, c, alpha)
        got = mx > 1 + 1e-6
        print(f'  {name.split(" (")[0]:12s} E = {E:6.4f}: max|mu| over an orbit period = {mx:.6f} (closure {clo:.1e}); '
              f'family prediction {"UNSTABLE" if pred else "stable"} -> {"agree" if got == pred else "DISAGREE"}')
        rep(f'  direct Floquet agrees with the Lame families at {name.split(" (")[0]}, E = {E:g}', 0 if got == pred else 1, 0)

print('\nCAVEAT: naive operator (no Lee-Wick Gauss term). The radial n = 2 family is exact as computed; families with')
print('off-diagonal-type directions (the n = 1 families in particular) need the exact operator before any claim.')
print('\nall checks passed' if OK else '\nSOME CHECKS FAILED')
print(f'# END lame_isotropic status={"OK" if OK else "FAIL"} wall={time.time() - T0:.0f}s', flush=True)
