# Copied verbatim from the hdym project, ym/hdym_core.py at commit 1d65e30 (24 Oct 2026),
# https://github.com/foejoseph-lab/hdym -- the SU(2) colour-diagonal model this project embeds in SU(3).
# Do not edit here; if hdym's copy changes, copy it again and note the commit.
"""
hdym_core.py -- numerical core for the homogeneous (diagonal) sector of SU(2)
Lee-Wick Yang-Mills, in units g = M = 1 (see hdym_derive.py for the derivation
and the checks that justify every formula here).

State per trajectory: (p, w, P, W), each a 3-vector.
    q = p - w              physical diagonal gauge-field amplitudes
    p                      normal (positive-kinetic) coordinate
    w                      ghost coordinate; on shell w = qddot + grad V(q)
    H = 1/2|P|^2 - 1/2|W|^2 + V(q) + w.gradV(q) - 1/2|w|^2
    pdot = P, wdot = -W, Pdot = -dU/dp, Wdot = -dU/dw

Key structural facts used below (all follow from H; easy to re-derive):
  * w = W = 0 is an INVARIANT submanifold on which the dynamics is exactly
    ordinary Yang-Mills mechanics  pddot = -grad V(p).
  * Linearising in w around it, p decouples at O(w) and the ghost obeys the
    Hill-type equation   wddot = -(1 + Hess V(p(t))) w.
    Hess V has a negative eigenvalue bounded by  lam_min >= -sqrt(2 V)  with
    equality on the diagonals x = +-y, z = 0 (and permutations), so the ghost
    'mass matrix' 1 + Hess V can go non-positive only for E_YM >= 1/2 exactly
    (a trajectory at rest on such a diagonal), and the tachyonic configurations
    lie near the ridges between channels, not deep inside them (in a channel
    lam_min ~ -3 y^2 -> 0 with depth).

Everything is vectorised over N trajectories (arrays of shape (N, 3)).
"""
import numpy as np

# ------------------------------------------------------------------ physics


def V(q):
    x, y, z = q[:, 0], q[:, 1], q[:, 2]
    return 0.5 * (x * x * y * y + y * y * z * z + z * z * x * x)


def gradV(q):
    x, y, z = q[:, 0], q[:, 1], q[:, 2]
    x2, y2, z2 = x * x, y * y, z * z
    return np.stack([x * (y2 + z2), y * (z2 + x2), z * (x2 + y2)], axis=1)


def hessV_dot(q, v):
    """(Hess V(q)) v, row-wise."""
    x, y, z = q[:, 0], q[:, 1], q[:, 2]
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    hx = (y * y + z * z) * vx + 2 * x * y * vy + 2 * x * z * vz
    hy = 2 * x * y * vx + (z * z + x * x) * vy + 2 * y * z * vz
    hz = 2 * x * z * vx + 2 * y * z * vy + (x * x + y * y) * vz
    return np.stack([hx, hy, hz], axis=1)


def hessV_min_eig(q):
    x, y, z = q[:, 0], q[:, 1], q[:, 2]
    Hm = np.empty((q.shape[0], 3, 3))
    Hm[:, 0, 0] = y * y + z * z
    Hm[:, 1, 1] = z * z + x * x
    Hm[:, 2, 2] = x * x + y * y
    Hm[:, 0, 1] = Hm[:, 1, 0] = 2 * x * y
    Hm[:, 0, 2] = Hm[:, 2, 0] = 2 * x * z
    Hm[:, 1, 2] = Hm[:, 2, 1] = 2 * y * z
    return np.linalg.eigvalsh(Hm)[:, 0]


def dU(p, w):
    q = p - w
    gv = gradV(q)
    hw = hessV_dot(q, w)
    return gv + hw, -hw - w          # dU/dp, dU/dw


def hamiltonian(s):
    p, w, P, W = s
    q = p - w
    return (0.5 * np.sum(P * P, 1) - 0.5 * np.sum(W * W, 1)
            + V(q) + np.sum(w * gradV(q), 1) - 0.5 * np.sum(w * w, 1))


def energy_parts(s):
    """Diagnostic split (not separately conserved):
    E_norm = 1/2|P|^2 + V(p)  >= 0 ; E_ghost = -(1/2|W|^2 + 1/2|w|^2) <= 0 ;
    E_int = H - E_norm - E_ghost."""
    p, w, P, W = s
    En = 0.5 * np.sum(P * P, 1) + V(p)
    Eg = -0.5 * (np.sum(W * W, 1) + np.sum(w * w, 1))
    return En, Eg, hamiltonian(s) - En - Eg


def abs_energy_scale(s):
    """Positive scale for relative energy errors (H itself can be ~0)."""
    p, w, P, W = s
    q = p - w
    return (0.5 * np.sum(P * P, 1) + 0.5 * np.sum(W * W, 1) + V(q)
            + np.abs(np.sum(w * gradV(q), 1)) + 0.5 * np.sum(w * w, 1))


# ------------------------------------------------------------------ initial data


def ym_initial(N, E, rng):
    """N random pure-Yang-Mills phase points at exactly energy E (w = W = 0),
    using the exact scaling symmetry q -> lam q, P -> lam^2 P, E -> lam^4 E."""
    q = rng.standard_normal((N, 3))
    P = rng.standard_normal((N, 3))
    E0 = 0.5 * np.sum(P * P, 1) + V(q)
    lam = (E / E0) ** 0.25
    return q * lam[:, None], P * (lam ** 2)[:, None]


def full_initial(N, E_ym, eps, rng):
    """Pure-YM point at energy E_ym plus a ghost excitation whose free ghost
    energy is  -eps * E_ym  (eps >= 0), with random phase/direction.
    Returns state tuple (p, w, P, W) with q = p - w equal to the YM point."""
    q, Pq = ym_initial(N, E_ym, rng)
    nw = rng.standard_normal((N, 3))
    nW = rng.standard_normal((N, 3))
    nrm = np.sqrt(np.sum(nw * nw, 1) + np.sum(nW * nW, 1))
    amp = np.sqrt(2.0 * eps * E_ym) / nrm
    w = nw * amp[:, None]
    W = nW * amp[:, None]
    # q fixed; p = q + w ; qdot fixed = pdot - wdot = P + W  => P = Pq - W
    p = q + w
    P = Pq - W
    return (p, w, P, W)


# ------------------------------------------------------------------ integrators

_c = 2.0 ** (1.0 / 3.0)
_w1 = 1.0 / (2.0 - _c)
_w0 = -_c / (2.0 - _c)
YOSHIDA_C = (_w1 / 2, (_w0 + _w1) / 2, (_w0 + _w1) / 2, _w1 / 2)
YOSHIDA_D = (_w1, _w0, _w1, 0.0)


def yoshida_step(s, dt):
    """4th-order symplectic step for the separable H = T(P,W) + U(p,w)."""
    p, w, P, W = s
    for c, d in zip(YOSHIDA_C, YOSHIDA_D):
        p = p + c * dt * P
        w = w - c * dt * W
        if d != 0.0:
            dUp, dUw = dU(p, w)
            P = P - d * dt * dUp
            W = W - d * dt * dUw
    return (p, w, P, W)


def evolve_ensemble(s0, dt, tmax, R_escape, n_samples=200, progress=False):
    """Fixed-step Yoshida-4 evolution of an ensemble.

    A trajectory is frozen ('escaped') when max|component| > R_escape or a
    non-finite value appears.  Returns dict of per-trajectory diagnostics and a
    coarse time series of ensemble statistics.
    """
    p, w, P, W = [a.copy() for a in s0]
    N = p.shape[0]
    H0 = hamiltonian((p, w, P, W))
    Escale = abs_energy_scale((p, w, P, W))
    alive = np.ones(N, bool)
    t_esc = np.full(N, np.inf)
    max_w = np.sqrt(np.sum(w * w, 1))
    max_q = np.sqrt(np.sum((p - w) ** 2, 1))
    max_relerr = np.zeros(N)
    nsteps = int(np.ceil(tmax / dt))
    every = max(1, nsteps // n_samples)
    ts, med_w, med_En, frac_alive = [], [], [], []
    for k in range(1, nsteps + 1):
        idx = np.nonzero(alive)[0]
        if idx.size == 0:
            break
        s = yoshida_step((p[idx], w[idx], P[idx], W[idx]), dt)
        big = ~np.all(np.isfinite(np.concatenate(s, 1)), 1)
        big |= np.max(np.abs(np.concatenate(s, 1)), 1) > R_escape
        tnow = k * dt
        if np.any(big):
            dead = idx[big]
            alive[dead] = False
            t_esc[dead] = tnow
        ok = ~big
        ii = idx[ok]
        p[ii], w[ii], P[ii], W[ii] = s[0][ok], s[1][ok], s[2][ok], s[3][ok]
        if k % every == 0 or k == nsteps:
            st = (p[ii], w[ii], P[ii], W[ii])
            wn = np.sqrt(np.sum(st[1] ** 2, 1))
            qn = np.sqrt(np.sum((st[0] - st[1]) ** 2, 1))
            max_w[ii] = np.maximum(max_w[ii], wn)
            max_q[ii] = np.maximum(max_q[ii], qn)
            err = np.abs(hamiltonian(st) - H0[ii]) / Escale[ii]
            max_relerr[ii] = np.maximum(max_relerr[ii], err)
            En, Eg, Ei = energy_parts(st)
            ts.append(tnow)
            med_w.append(np.median(wn) if wn.size else np.nan)
            med_En.append(np.median(En) if En.size else np.nan)
            frac_alive.append(alive.mean())
            if progress:
                print(f"  t={tnow:10.2f} alive={alive.mean():.3f} "
                      f"median|w|={med_w[-1]:.3e} max relerr={max_relerr.max():.2e}",
                      flush=True)
    return dict(t_escape=t_esc, max_w=max_w, max_q=max_q, max_relerr=max_relerr,
                H0=H0, final=(p, w, P, W),
                series=dict(t=np.array(ts), median_w=np.array(med_w),
                            median_En=np.array(med_En),
                            frac_alive=np.array(frac_alive)))


# ------------------------------------------------------------------ transverse stability


def _lin_step(p, w, P, Pi, dt):
    for c, d in zip(YOSHIDA_C, YOSHIDA_D):
        p = p + c * dt * P
        w = w + c * dt * Pi
        if d != 0.0:
            P = P - d * dt * gradV(p)
            Pi = Pi - d * dt * (w + hessV_dot(p, w))
    return p, w, P, Pi


def _lin_step_substepped(p, w, P, Pi, dt, omega_dt_max=0.25):
    """One step of dt for the YM + linearised-ghost system, sub-stepping any
    trajectory that has wandered far down a Yang-Mills 'channel'.

    In a channel (|x| large, y,z ~ 0) the transverse frequencies are ~|x|,
    and a fixed-step symplectic integrator goes unstable once |x| dt > ~2.
    Pure YM is bounded but a trajectory with a tiny transverse action can
    travel very far along the channel over long runs (this produced NaNs in
    testing at E = 0.1 after t ~ 500).  Trajectories are grouped by the
    number of substeps they need, so the common case costs nothing extra."""
    omega = np.max(np.abs(p), axis=1)
    nsub = np.maximum(1, np.ceil(omega * dt / omega_dt_max)).astype(int)
    if np.all(nsub == 1):
        return _lin_step(p, w, P, Pi, dt)
    p, w, P, Pi = p.copy(), w.copy(), P.copy(), Pi.copy()
    for n in np.unique(nsub):
        idx = np.nonzero(nsub == n)[0]
        sub = (p[idx], w[idx], P[idx], Pi[idx])
        for _ in range(n):
            sub = _lin_step(*sub, dt / n)
        p[idx], w[idx], P[idx], Pi[idx] = sub
    return p, w, P, Pi


def transverse_lyapunov(E_ym, N, dt, tmax, rng, renorm_every=50, t_burn=50.0):
    """Largest Lyapunov exponent of the LINEARISED ghost along pure-YM
    trajectories at energy E_ym:
        pddot = -grad V(p),     wddot = -(1 + Hess V(p)) w.
    Integrated with the same Yoshida splitting (kick uses current p).
    Returns per-trajectory exponents and the YM energy drift.
    Also returns the fraction of time the ghost mass matrix 1 + Hess V has a
    non-positive eigenvalue (local tachyonic direction)."""
    p, P = ym_initial(N, E_ym, rng)
    w = rng.standard_normal((N, 3))
    Pi = rng.standard_normal((N, 3))
    nrm = np.sqrt(np.sum(w * w, 1) + np.sum(Pi * Pi, 1))
    w /= nrm[:, None]
    Pi /= nrm[:, None]
    E0 = 0.5 * np.sum(P * P, 1) + V(p)
    logsum = np.zeros(N)
    tach = np.zeros(N)
    nsteps = int(np.ceil(tmax / dt))
    nburn = int(np.ceil(t_burn / dt))
    for k in range(1, nsteps + 1):
        p, w, P, Pi = _lin_step_substepped(p, w, P, Pi, dt)
        if k % renorm_every == 0:
            nrm = np.sqrt(np.sum(w * w, 1) + np.sum(Pi * Pi, 1))
            if k > nburn:
                logsum += np.log(nrm)
            w /= nrm[:, None]
            Pi /= nrm[:, None]
            if k > nburn:
                tach += (1.0 + hessV_min_eig(p) <= 0.0) * renorm_every
    T = (nsteps - nburn) * dt
    E1 = 0.5 * np.sum(P * P, 1) + V(p)
    return dict(lam=logsum / T, tach_frac=tach / (nsteps - nburn),
                ym_relerr=np.abs(E1 - E0) / E0)


def transverse_lyapunov_adaptive(E_ym, N, dt, rng, target_abs=1e-4, target_rel=0.05,
                                 t_block=200.0, t_burn=50.0, tmax=2e5, wall=None,
                                 renorm_every=50, n_consec=3, log=None,
                                 t_burn_ym=None, trend_check=True):
    """Same linearised-ghost problem as transverse_lyapunov, but integrated in
    blocks with an automatic stopping rule.

    Estimator.  For each trajectory the accumulated log-growth S(T) is a
    straight line lambda*T + c  once transients die away; when the true
    lambda is 0, S(T) just oscillates (bounded ghost), so S/T decays like 1/T
    and never 'levels off' cleanly.  We therefore estimate lambda from the
    SLOPE of S(T) over the trailing half of the run (least squares over the
    block checkpoints), which removes the offset c and the 1/T tail.

    Stopping.  After every block, compute the ensemble mean and standard
    error of the slope estimate, and its drift = |lambda(now) - lambda(one
    block ago)|.  Stop when, for n_consec consecutive blocks,
          se < tol   and   drift < tol,     tol = max(target_abs, target_rel*|lambda|)
    Also stop at tmax or after `wall` seconds (converged=False).

    Interpretation.  converged with |lambda| < target_abs  => 'consistent
    with zero at resolution target_abs'; converged with lambda > tol =>
    resolved positive exponent.  target_abs sets the floor of what you can
    claim, so lower it (and pay in tmax) if the low-E points come out
    'consistent with zero' and you want a tighter bound.

    Burn-in.  `t_burn` is in absolute time.  The Yang-Mills relaxation time
    scales like E^{-1/4}, so 50 time units is only ~17 YM units at E = 0.015.
    `t_burn_ym` (if given) sets the burn-in in YM units: t_burn = t_burn_ym *
    E^{-1/4}.  Initial data from ym_initial are NOT microcanonical, and the
    channel-depth distribution is heavy-tailed, so a short burn-in leaves a
    slow downward drift in the estimate that the stopping rule cannot see.

    Trend check.  Consecutive trailing-half slopes share most of their data,
    so the block-to-block drift is tiny even while the estimate is sliding
    like a + b/T.  With trend_check the trailing half of the history is fitted
    to a + b/T and the extrapolated shift |lam - a| must also be < tol.
    `lam_extrap` (= a) is returned alongside `lam_mean`.
    """
    import time as _time
    t0 = _time.time()
    if t_burn_ym is not None:
        t_burn = t_burn_ym * E_ym ** -0.25
    p, P = ym_initial(N, E_ym, rng)
    w = rng.standard_normal((N, 3))
    Pi = rng.standard_normal((N, 3))
    nrm = np.sqrt(np.sum(w * w, 1) + np.sum(Pi * Pi, 1))
    w /= nrm[:, None]
    Pi /= nrm[:, None]
    E0 = 0.5 * np.sum(P * P, 1) + V(p)
    logsum = np.zeros(N)
    tach = 0.0
    lam_extrap = np.nan
    nburn = int(np.ceil(t_burn / dt))
    nblock = int(np.ceil(t_block / dt))
    k = 0
    chk_T, chk_S = [], []          # checkpoints: time since burn-in, S per traj
    hist = []                      # (T, lam, se, drift)
    lam_prev, good, converged = None, 0, False
    while True:
        for _ in range(nblock):
            k += 1
            p, w, P, Pi = _lin_step_substepped(p, w, P, Pi, dt)
            if k % renorm_every == 0:
                nrm = np.sqrt(np.sum(w * w, 1) + np.sum(Pi * Pi, 1))
                if k > nburn:
                    logsum += np.log(nrm)
                    tach += np.mean(1.0 + hessV_min_eig(p) <= 0.0) * renorm_every
                w /= nrm[:, None]
                Pi /= nrm[:, None]
        if k <= nburn:
            continue
        T = (k - nburn) * dt
        if not np.all(np.isfinite(logsum)):
            raise FloatingPointError(
                f"non-finite log-growth at E={E_ym}, T={T}: integrator blew up; "
                "lower dt0 or omega_dt_max in _lin_step_substepped")
        chk_T.append(T)
        chk_S.append(logsum.copy())
        # slope over trailing half of checkpoints (need >= 3 points)
        Ta = np.array(chk_T)
        i0 = max(0, int(len(Ta) / 2) - 1)
        if len(Ta) - i0 < 3:
            continue
        Ts = Ta[i0:] - Ta[i0:].mean()
        Ss = np.array(chk_S[i0:])                       # (n_chk, N)
        lam_i = (Ts[:, None] * (Ss - Ss.mean(0))).sum(0) / (Ts * Ts).sum()
        lam = lam_i.mean()
        se = lam_i.std(ddof=1) / np.sqrt(N)
        drift = np.inf if lam_prev is None else abs(lam - lam_prev)
        lam_prev = lam
        tol = max(target_abs, target_rel * abs(lam))
        hist.append((T, lam, se, drift))
        trend = 0.0
        if trend_check and len(hist) >= 4:
            h = np.array(hist)[len(hist) // 2:]
            A = np.stack([np.ones(len(h)), 1.0 / h[:, 0]], 1)
            lam_extrap = np.linalg.lstsq(A, h[:, 1], rcond=None)[0][0]
            trend = abs(lam - lam_extrap)
        if log:
            log(f"    E={E_ym:.4g} T={T:9.0f} lam={lam:+.3e} se={se:.1e} "
                f"drift={drift:.1e} trend={trend:.1e} tol={tol:.1e} ({_time.time() - t0:.0f}s)")
        good = good + 1 if (se < tol and drift < tol and trend < tol) else 0
        if good >= n_consec:
            converged = True
            break
        if T >= tmax or (wall is not None and _time.time() - t0 > wall):
            break
    E1 = 0.5 * np.sum(P * P, 1) + V(p)
    return dict(lam=lam_i, lam_mean=lam, lam_se=se, lam_extrap=lam_extrap,
                converged=converged, T=T, t_burn=t_burn,
                tach_frac=tach / (k - nburn), ym_relerr=np.abs(E1 - E0) / E0,
                history=np.array(hist), seconds=_time.time() - t0)


def _ym_step(p, P, h):
    for c, d in zip(YOSHIDA_C, YOSHIDA_D):
        p = p + c * h * P
        if d != 0.0:
            P = P - d * h * gradV(p)
    return p, P


def ym_step_substepped(p, P, dt, omega_dt_max=0.25):
    """Pure-YM Yoshida step with per-trajectory channel sub-stepping (same
    grouping as _lin_step_substepped)."""
    omega = np.max(np.abs(p), axis=1)
    nsub = np.maximum(1, np.ceil(omega * dt / omega_dt_max)).astype(int)
    if np.all(nsub == 1):
        return _ym_step(p, P, dt)
    p, P = p.copy(), P.copy()
    for n in np.unique(nsub):
        idx = np.nonzero(nsub == n)[0]
        a, b = p[idx], P[idx]
        for _ in range(n):
            a, b = _ym_step(a, b, dt / n)
        p[idx], P[idx] = a, b
    return p, P


def ym_lyapunov(E_ym, N, dt, tmax, rng, d0=1e-8, renorm_every=50, t_burn=0.0,
                omega_dt_max=0.25):
    """Max Lyapunov exponent of pure YM mechanics (two-trajectory method);
    sanity check against the known ~ E^{1/4} scaling (exact here by symmetry).

    Sub-steps channel excursions (both copies of a pair identically, using the
    reference copy's depth) and excludes `t_burn` from the average.  Channel
    depths are heavy-tailed -- at E = 1 the max|p| over ~400 time units has
    median ~10, 99th percentile ~80, exceedance ~ D^-2 -- so a long fixed-step
    run will eventually push a trajectory past the Yoshida stability limit
    omega*dt = 1.57 and silently corrupt the ensemble mean."""
    p, P = ym_initial(N, E_ym, rng)
    dp = rng.standard_normal((N, 3))
    dp *= d0 / np.linalg.norm(dp, axis=1)[:, None]
    p2, P2 = p + dp, P.copy()
    logsum = np.zeros(N)
    nsteps = int(np.ceil(tmax / dt))
    nburn = int(np.ceil(t_burn / dt))
    for k in range(1, nsteps + 1):
        omega = np.max(np.abs(p), axis=1)
        nsub = np.maximum(1, np.ceil(omega * dt / omega_dt_max)).astype(int)
        if np.all(nsub == 1):
            p, P = _ym_step(p, P, dt)
            p2, P2 = _ym_step(p2, P2, dt)
        else:
            p, P, p2, P2 = p.copy(), P.copy(), p2.copy(), P2.copy()
            for n in np.unique(nsub):
                idx = np.nonzero(nsub == n)[0]
                a, b, a2, b2 = p[idx], P[idx], p2[idx], P2[idx]
                for _ in range(n):
                    a, b = _ym_step(a, b, dt / n)
                    a2, b2 = _ym_step(a2, b2, dt / n)
                p[idx], P[idx], p2[idx], P2[idx] = a, b, a2, b2
        if k % renorm_every == 0:
            dz = np.concatenate([p2 - p, P2 - P], 1)
            n = np.linalg.norm(dz, axis=1)
            if k > nburn:
                logsum += np.log(n / d0)
            p2 = p + (p2 - p) * (d0 / n)[:, None]
            P2 = P + (P2 - P) * (d0 / n)[:, None]
    return logsum / ((nsteps - nburn) * dt)


# ------------------------------------------------------------------ adaptive confirmation


def rhs_flat(t, y):
    p, w, P, W = y[0:3], y[3:6], y[6:9], y[9:12]
    dUp, dUw = dU(p[None], w[None])
    return np.concatenate([P, -W, -dUp[0], -dUw[0]])


def confirm_runaway(state, t_max, radii=tuple(10.0 ** k for k in range(1, 9)),
                    rtol=1e-11, atol=1e-13):
    """Re-integrate ONE trajectory with adaptive DOP853 at tight tolerance.
    Records the first time max|state| crosses each radius (one per decade).

    Growth-law diagnostic from the per-decade crossing gaps Delta_k:
      finite-time blow-up  R ~ (t*-t)^(-alpha): Delta_k shrink geometrically,
                           slope s = dlog10(Delta)/dk < 0, alpha = -1/s
      exponential growth   Delta_k ~ const   (s ~ 0)
      power-law growth     Delta_k increase  (s > 0)
    Labels: 'bounded', 'finite-time', 'exponential', 'power-law', 'growing'
    (too few crossings to fit), 'inconclusive' (solver failure / energy drift).
    The radii are in units of M/g (fields) -- scale them if you change units.

    Per-variable crossing times.  `crossing_times_by_var` gives the decade
    crossings of |q|, |w|, |P|, |W| separately and `alpha_by_var` the fitted
    exponent of each.  Do not over-interpret alpha: in test cases the
    singularity is NOT a clean power law.  |w|, |P|, |W| grow monotonically
    with per-decade local exponents fluctuating between ~1 and ~10, while |q|
    is non-monotonic and sign-flipping.  `profile` samples the last stretch
    before t*: it shows q thrown along a channel axis with a transverse
    displacement rho for which 1 + lam_min(Hess V) ~ 1 - 3 rho^2 << 0, i.e.
    the finite-time blow-up is a self-accelerating tachyonic loop (large w
    drives q, q's transverse displacement makes the ghost mass matrix hugely
    negative, which drives w), not an isotropic scale-invariant balance.
    """
    from scipy.integrate import solve_ivp
    y0 = np.concatenate([a.ravel() for a in state])
    H0 = hamiltonian(tuple(a[None] for a in state))[0]
    Es = abs_energy_scale(tuple(a[None] for a in state))[0]
    groups = dict(state=lambda y: np.max(np.abs(y)),
                  q=lambda y: np.max(np.abs(y[0:3] - y[3:6])),
                  w=lambda y: np.max(np.abs(y[3:6])),
                  P=lambda y: np.max(np.abs(y[6:9])),
                  W=lambda y: np.max(np.abs(y[9:12])))
    events = []
    for gname, gfun in groups.items():
        for R in radii:
            f = (lambda gfun, R: (lambda t, y: gfun(y) - R))(gfun, R)
            f.terminal = (gname == 'state' and R == radii[-1])
            f.direction = 1
            events.append(f)
    sol = solve_ivp(rhs_flat, (0, t_max), y0, method='DOP853', rtol=rtol,
                    atol=atol, events=events, dense_output=False)
    nR = len(radii)
    cross_all = {}
    for gi, gname in enumerate(groups):
        cross_all[gname] = [ev[0] if len(ev) else np.nan
                            for ev in sol.t_events[gi * nR:(gi + 1) * nR]]
    cross = cross_all['state']

    def _alpha(c):
        c = np.asarray(c, float)
        valid = c[~np.isnan(c)]
        gaps = np.diff(valid)
        if len(gaps) >= 3 and np.all(gaps > 0):
            sl = np.polyfit(np.arange(len(gaps)), np.log10(gaps), 1)[0]
            return (-1.0 / sl) if sl < 0 else np.nan
        return np.nan
    alpha_by_var = {g: _alpha(cross_all[g]) for g in ('q', 'w', 'P', 'W')}
    profile = []
    if not np.isnan(cross[-1]) if len(cross) else False:
        tstar = cross[-1]
        sol2 = solve_ivp(rhs_flat, (0, tstar), y0, method='DOP853', rtol=rtol,
                         atol=atol, dense_output=True)
        for tau in (1.0, 0.3, 0.1, 0.03, 0.01, 0.003):
            if tstar - tau <= 0:
                continue
            y = sol2.sol(tstar - tau)
            q = (y[0:3] - y[3:6])[None]
            i = int(np.argmax(np.abs(q[0])))
            rho = np.sqrt(np.sum(q[0] ** 2) - q[0, i] ** 2)
            profile.append(dict(tau=tau, q_max=float(np.abs(q).max()), rho=float(rho),
                                one_plus_lam_min=float(1.0 + hessV_min_eig(q)[0]),
                                w_max=float(np.abs(y[3:6]).max())))
    yend = sol.y[:, -1]
    st_end = (yend[0:3][None], yend[3:6][None], yend[6:9][None], yend[9:12][None])
    # relative to the CURRENT magnitude of the energy terms: near a blow-up the
    # individual terms are huge and cancel, so compare against their size.
    relerr = abs(hamiltonian(st_end)[0] - H0) / max(Es, abs_energy_scale(st_end)[0])
    c = np.array(cross)
    slope = np.nan
    if np.all(np.isnan(c)):
        label = 'bounded' if sol.status == 0 else 'inconclusive'
    elif sol.status == -1 or relerr > 1e-6:
        label = 'inconclusive'
    else:
        valid = c[~np.isnan(c)]
        gaps = np.diff(valid)
        if len(gaps) >= 3 and np.all(gaps > 0):
            k = np.arange(len(gaps))
            slope = np.polyfit(k, np.log10(gaps), 1)[0]
            label = ('finite-time' if slope < -0.1 else
                     'exponential' if slope <= 0.1 else 'power-law')
        else:
            label = 'growing'
    alpha = -1.0 / slope if (np.isfinite(slope) and slope < 0) else np.nan
    return dict(label=label, crossing_times=cross, energy_relerr=relerr,
                gap_slope=slope, blowup_exponent=alpha,
                alpha_by_var=alpha_by_var, crossing_times_by_var=cross_all,
                profile=profile,
                t_end=sol.t[-1], status=sol.status, message=sol.message)
