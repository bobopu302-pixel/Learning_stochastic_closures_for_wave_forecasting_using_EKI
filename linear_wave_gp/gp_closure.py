"""Ten-mode wave closure whose damping is replaced by two learned GP functions
(the thesis Chapter 4 model): case constants, simulators, statistics assembly,
coordinates, priors, and the linear-stability gate.

Model
-----
    dq_j  = [ p_j + Phi_q(q_j) ] dt
    dp_j  = [ -omega_j^2 q_j + Phi_p(p_j) ] dt + sqrt(sigma_j) dW_j,   j = 1..M

The fixed damping ``-delta_j p_j`` of the linear-wave case is gone.  In its place
are TWO functions, one per equation, SHARED by all ten modes and represented
exactly as in the Lorenz 63 case b of ``1. Reproduce_papers``: Gaussian-process
mean functions through R fixed nodes, parameterised by node values plus three
kernel hyperparameters each.

Parameters (physical), 2*(R+3) + M = 34 at R = 9, M = 10:

    theta = [ Phi_q nodes (R) | obs_noise, amplitude, lengthscale ]
            [ Phi_p nodes (R) | obs_noise, amplitude, lengthscale ]
            [ sqrt(sigma_1) .. sqrt(sigma_M) ]

Node values are FREE SIGN; the six hyperparameters and the ten noise amplitudes
are strictly positive and held in log coordinates.

Why the functions are shared rather than per mode
-------------------------------------------------
Two functions per mode would be 10*(R+3)*2 + M = 250 parameters at R = 9.  That
fails twice over.  An EKI update lives in the span of the ensemble anomalies,
of rank at most J-1 = 99, so most directions would never move from their
initial values.  And the data cannot support it either: q = 38 statistics whose
correlation matrix puts 99% of its trace in about 10 directions cannot
determine hundreds of parameters.  34 is already generous.

What sharing costs
------------------
Linearised at the origin,

    A = [[ Phi_q'(0), 1 ], [ -omega^2, Phi_p'(0) ]]

so trace = Phi_q'(0) + Phi_p'(0) and det = Phi_q'(0) Phi_p'(0) + omega^2.  For
the eigenvalues are approximately trace/2 +/- i*omega at large omega, the linear
damping rate is the SAME for every mode.  The linear-wave case fitted ten
independent delta_j spanning a factor of 47.  Shared functions cannot reproduce
that at linear order; the modes are differentiated only through amplitude,
because sd(p_j) spans a factor of 800 at the old solution and the large modes
sample the nonlinear far field of Phi_p.  Whether that suffices is the
experiment.

Stationarity is not assumed: trace < 0 and det > 0 are required for an invariant
measure to exist, and nothing here forces them.  A proposal that violates them
diverges and returns the failure sentinel, which is counted and reported.

Integrator
----------
The linear-wave case uses an EXACT Ornstein-Uhlenbeck propagator, valid only for
linear drift.  Here the drift is nonlinear, and an explicit Euler step is not an
alternative: at omega*dt = 0.57 the Euler map of a harmonic oscillator has
modulus sqrt(1 + (omega dt)^2) = 1.15, so the energy grows 15% per step -- an
earlier undamped test with Euler reported Var(q) = 3e42 after 50 s.

So each drift is split into the part that fits inside a matrix exponential and
the part that does not,

    Phi_q(q) = Phi_q(0) + Phi_q'(0) q + psi_q(q)
    Phi_p(p) = Phi_p(0) + Phi_p'(0) p + psi_p(p)

the affine part is carried exactly by expm(A dt) with its noise covariance from
van Loan's method -- valid whether or not A is stable, unlike the closed-form
stationary expression, which divides by the damping -- and the two nonlinear
remainders are added explicitly.  Exact in the oscillation and the noise, first
order in the nonlinearity; ``timestep_error`` measures what that costs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm

# Release layout: shared algorithms at the code_rp root, the modal-closure case
# in code_rp/linear_wave/.  Spawned workers re-import this module and run this
# bootstrap again, so they resolve both packages on their own.
_CODE_RP_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_CODE_RP_ROOT), str(_CODE_RP_ROOT / "linear_wave")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from algorithms.gpr import make_gp_mean_from_theta as make_gp_mean_function  # noqa: E402
from algorithms.parameterization import (                    # noqa: E402
    decode_positive_columns, encode_positive_columns)
from modal_closure import experiment as case                 # noqa: E402

# --- case constants, from the linear wave case so the two are comparable ----
M_MODES = case.M_MODES
OMEGA = case.OMEGA_GRID
OMEGA2 = case.OMEGA2_GRID
K_GRID = case.K_GRID
GAUGES = case.GAUGES
DT_DATA = case.DT_DATA
T_BURN = case.T_BURN
AUTO_LAGS = case.AUTO_LAGS
CROSS_LAGS = case.CROSS_LAGS
FORWARD_SENTINEL = case.FORWARD_SENTINEL

# --- GP closure -------------------------------------------------------------
# Node ranges follow the state each function actually sees.  At the linear-wave
# solution the pooled sd is 1.00 for q and 6.37 for p, and the widest single
# mode reaches sd(p) = 3.7; +-3 and +-8 are about three standard deviations of
# the widest mode in each variable.  The first run used +-12 on p, wider than
# the data support, which let the prior wander into shapes nothing constrains.
#
# Nine nodes rather than five.  The five-node fit returned Phi_p node values
# alternating in sign, +1.35, +2.94, -2.84, +4.34, -0.75, which a coarse grid
# cannot distinguish from a genuinely oscillatory closure -- with a spacing of
# 4 in p there is simply nothing between the knots to say which it is.  The
# spacing is now 2.0 in p and 0.75 in q, and the lengthscale prior starts at the
# spacing so the interpolant cannot be more wiggly than the grid can resolve.
N_NODES = 9
NODES_Q = np.linspace(-3.0, 3.0, N_NODES)
NODES_P = np.linspace(-8.0, 8.0, N_NODES)

# Reject a closure with no invariant measure BEFORE simulating it.
#
# The first run without this gate ended with 0 of 100 members linearly
# stationary: EKI had walked into the region where trace > 0 and the state is
# held bounded only by the nonlinear far field of Phi_p.  Phi there is set by
# nonlinear saturation rather than by the physics, and it is occasionally low by
# luck, so the objective oscillated over seven orders of magnitude and its final
# value was a fortunate draw rather than a converged fit.
#
# The gate makes the forward map discontinuous at the stationarity boundary,
# which is a real cost to the ensemble covariance.  It is accepted because "the
# model must have a long-run distribution" is not an optional modelling
# preference -- without it y, Gamma and Phi are all undefined.
STATIONARITY_GATE = True

BLOCK = N_NODES + 3
IDX_Q = slice(0, BLOCK)
IDX_P = slice(BLOCK, 2 * BLOCK)
IDX_SIGMA = slice(2 * BLOCK, 2 * BLOCK + M_MODES)
N_THETA = 2 * BLOCK + M_MODES

# Positive columns: the three hyperparameters of each block, and every sigma.
POSITIVE = tuple(
    [N_NODES, N_NODES + 1, N_NODES + 2,
     BLOCK + N_NODES, BLOCK + N_NODES + 1, BLOCK + N_NODES + 2]
    + list(range(2 * BLOCK, N_THETA)))

# The prior on the node values is drawn as a decreasing ramp plus a
# perturbation, value_i = -slope * node_i + jitter, rather than as independent
# uniforms.  A dissipative closure is a DECREASING function through the origin,
# so a ramp-centred prior puts most of its mass where an invariant measure
# exists; independent uniforms put only about 38% of it there, and with the
# stationarity gate on, the other 62% would have arrived at EKI as sentinels and
# contaminated the ensemble covariance from the first iteration.
#
# This biases the prior towards dissipation without imposing it: the jitter can
# outweigh the ramp, so a non-dissipative closure remains reachable and is still
# something EKI can discover rather than something assumed.
SLOPE_PRIOR_Q = (0.05, 1.5)
SLOPE_PRIOR_P = (0.02, 0.6)
NODE_JITTER_Q = 0.6
NODE_JITTER_P = 0.6
OBS_NOISE_PRIOR = (0.05, 1.0)
AMPLITUDE_PRIOR = (0.5, 8.0)
LENGTHSCALE_PRIOR_Q = (0.75, 3.0)
LENGTHSCALE_PRIOR_P = (2.0, 10.0)
SIGMA_PRIOR = (0.5, 6.0)


def _affine_parts(function):
    """f(0) and f'(0) of a GP mean function, in closed form.

    m(x) = sum_i w_i a^2 exp(-(x - n_i)^2 / (2 l^2)), so both follow from the
    stored (nodes, weights, amplitude, lengthscale).  Closed form rather than a
    finite difference, so the split does not carry a step-size choice of its own.
    """

    nodes, weights, amplitude, lengthscale = function._gp_params
    gauss = amplitude ** 2 * np.exp(-0.5 * (nodes / lengthscale) ** 2)
    value = float(gauss @ weights)
    slope = float((gauss * (nodes / lengthscale ** 2)) @ weights)
    return value, slope


def unpack(theta):
    """(Phi_q, Phi_p, affine parts of each, sqrt_sigma) from a physical theta."""

    theta = np.asarray(theta, dtype=float).reshape(-1)
    phi_q = make_gp_mean_function(theta[IDX_Q], NODES_Q)
    phi_p = make_gp_mean_function(theta[IDX_P], NODES_P)
    return phi_q, phi_p, _affine_parts(phi_q), _affine_parts(phi_p), theta[IDX_SIGMA]


def _van_loan(omega2, a_qq, a_pp, sigma, dt):
    """Exact one-step propagator and noise root for the affine drift.

    A = [[Phi_q'(0), 1], [-omega^2, Phi_p'(0)]].  van Loan's block exponential
    gives Q = int_0^dt Phi(s) B B^T Phi(s)^T ds for ANY A; the usual
    Q = P - Phi P Phi^T shortcut needs the stationary P, which does not exist
    when the closure is not dissipative.
    """

    m = omega2.size
    prop = np.empty((m, 2, 2))
    root = np.zeros((m, 2, 2))
    for j in range(m):
        a = np.array([[a_qq, 1.0], [-omega2[j], a_pp]])
        block = np.zeros((4, 4))
        block[:2, :2] = -a
        block[:2, 2:] = np.array([[0.0, 0.0], [0.0, sigma[j]]])
        block[2:, 2:] = a.T
        e = expm(block * dt)
        p_j = e[2:, 2:].T
        q_cov = 0.5 * ((p_j @ e[:2, 2:]) + (p_j @ e[:2, 2:]).T)
        w, vecs = np.linalg.eigh(q_cov)
        prop[j] = p_j
        root[j] = vecs @ np.diag(np.sqrt(np.clip(w, 0.0, None)))
    return prop, root


def linear_stability(theta):
    """Trace and determinant of the linearised drift, per mode.

    An invariant measure needs trace < 0 and det > 0.  Exposed so a run can
    report how many EKI proposals were non-stationary rather than only counting
    the sentinels they produced.
    """

    _, _, (_, sq), (_, sp), _ = unpack(theta)
    trace = sq + sp
    det = sq * sp + OMEGA2
    return trace, det, bool(trace < 0.0 and np.all(det > 0.0))


def simulate(theta, *, rng, t_record, t_burn=T_BURN, dt=DT_DATA, max_abs=1e6):
    """One realisation of the two-function GP closure, read out at the gauges."""

    phi_q, phi_p, (q0, sq), (p0, sp), sqrt_sigma = unpack(theta)
    sigma = np.asarray(sqrt_sigma, dtype=float) ** 2
    if np.any(sigma <= 0.0) or not np.all(np.isfinite(sigma)):
        raise ValueError("noise intensities must be positive and finite")

    prop, root = _van_loan(OMEGA2, sq, sp, sigma, dt)
    p00, p01, p10, p11 = prop[:, 0, 0], prop[:, 0, 1], prop[:, 1, 0], prop[:, 1, 1]
    l00, l01, l10, l11 = root[:, 0, 0], root[:, 0, 1], root[:, 1, 0], root[:, 1, 1]

    n_burn = int(round(t_burn / dt))
    n_steps = int(round(t_record / dt))

    # Start from the stationary law of the LINEARISED closure where it exists,
    # from zero where it does not.  The burn-in is discarded either way; this
    # only decides how long the transient lasts.
    trace, det = sq + sp, sq * sp + OMEGA2
    if trace < 0.0 and np.all(det > 0.0):
        damping = -trace
        q = rng.normal(size=M_MODES) * np.sqrt(sigma / (damping * det))
        qd = rng.normal(size=M_MODES) * np.sqrt(sigma / damping)
    else:
        q = np.zeros(M_MODES)
        qd = np.zeros(M_MODES)

    kx = np.outer(GAUGES, K_GRID)
    cq, sqx = np.cos(kx), np.sin(kx)
    eta = np.empty((n_steps, len(GAUGES)))
    v = np.empty((n_steps, len(GAUGES)))

    chunk = 4000
    q_block = np.empty((chunk, M_MODES))
    qd_block = np.empty((chunk, M_MODES))
    total = n_burn + n_steps
    written = 0

    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        z = rng.normal(size=(stop - start, M_MODES, 2))
        kept = 0
        for offset in range(stop - start):
            # nonlinear remainders, explicit; the affine parts ride the propagator
            q = q + (q0 + (phi_q(q) - q0 - sq * q)) * dt
            qd = qd + (p0 + (phi_p(qd) - p0 - sp * qd)) * dt
            z0, z1 = z[offset, :, 0], z[offset, :, 1]
            q, qd = (p00 * q + p01 * qd + l00 * z0 + l01 * z1,
                     p10 * q + p11 * qd + l10 * z0 + l11 * z1)
            if not np.all(np.isfinite(qd)) or max(np.max(np.abs(qd)),
                                                  np.max(np.abs(q))) > max_abs:
                raise FloatingPointError(
                    "the GP closure did not dissipate; the state exceeded the guard")
            step = start + offset
            if step >= n_burn:
                q_block[kept] = q
                qd_block[kept] = qd
                kept += 1
        if kept:
            qs, qds = q_block[:kept], qd_block[:kept]
            eta[written:written + kept] = qs @ cq.T - (qds / OMEGA) @ sqx.T
            v[written:written + kept] = (qs * OMEGA) @ sqx.T + qds @ cq.T
            written += kept
    return eta, v


def statistics(theta, *, rng, t_record):
    """The same 38-vector the linear wave case fits, from its own code."""

    eta, v = simulate(theta, rng=rng, t_record=t_record)
    return case.spatial_statistics(eta, v, AUTO_LAGS, CROSS_LAGS)


# --- batched forward --------------------------------------------------------
# Profiling one forward: the GP wrapper and rbf_kernel took 43% of the time, the
# per-step finite-value guard 18%, the loop body 37%.  None of that is real
# arithmetic -- the state is ten numbers, so every step is dominated by the
# fixed cost of ~45 numpy calls on tiny arrays.
#
# Inlining the kernel and checking the guard once per block removes the first
# two.  The rest is removed by evaluating a BATCH of ensemble members in one
# pass: the arrays become (batch, modes) and (batch, modes, nodes), the call
# count per step is unchanged, and the cost per member falls by roughly the
# batch size.  This is exactly compatible with the spec's common random
# numbers -- every member of an iteration is driven by the SAME noise, so one
# draw serves the whole batch, which is why batching is possible at all.


def _gp_batch_params(theta_batch, index, nodes):
    """Kernel weights and scalars for one GP block, for a whole batch."""

    block = np.asarray(theta_batch, dtype=float)[:, index]
    values = block[:, :N_NODES]
    obs_noise = block[:, N_NODES]
    amplitude = block[:, N_NODES + 1]
    lengthscale = block[:, N_NODES + 2]
    if np.any(obs_noise <= 0) or np.any(amplitude <= 0) or np.any(lengthscale <= 0):
        raise ValueError("GP hyperparameters must be strictly positive")

    diff = nodes[:, None] - nodes[None, :]
    weights = np.empty_like(values)
    for b in range(values.shape[0]):
        k_nn = (amplitude[b] ** 2 * np.exp(-0.5 * (diff / lengthscale[b]) ** 2)
                + (obs_noise[b] ** 2 + 1e-8) * np.eye(nodes.size))
        weights[b] = np.linalg.solve(k_nn, values[b])

    amp2 = (amplitude ** 2)[:, None]
    inv_l2 = (1.0 / lengthscale ** 2)[:, None]
    gauss0 = amp2 * np.exp(-0.5 * nodes[None, :] ** 2 * inv_l2)
    value0 = np.einsum("br,br->b", gauss0, weights)
    slope0 = np.einsum("br,br->b", gauss0 * nodes[None, :] * inv_l2, weights)
    return weights, amp2[:, :, None], inv_l2[:, :, None], value0, slope0


def _gp_batch_eval(x, nodes, weights, amp2, inv_l2):
    """phi(x) for x of shape (batch, modes); the same RBF formula as the shared
    algorithms.gpr kernel, inlined for speed."""

    d = x[:, :, None] - nodes[None, None, :]
    kern = amp2 * np.exp(-0.5 * d * d * inv_l2)
    return np.einsum("bmr,br->bm", kern, weights)


def simulate_batch(theta_batch, *, rng, t_record, t_burn=T_BURN, dt=DT_DATA,
                   max_abs=1e6):
    """Simulate a batch of members under ONE shared noise realisation.

    Returns (eta, v) of shape (batch, steps, gauges) and a boolean mask of the
    members that blew up.  A member that fails does not abort the batch -- its
    mask entry is set and its output is left undefined, so the caller can emit
    the sentinel for it alone.
    """

    theta_batch = np.atleast_2d(np.asarray(theta_batch, dtype=float))
    n_batch = theta_batch.shape[0]

    wq, aq, lq, q0, sq = _gp_batch_params(theta_batch, IDX_Q, NODES_Q)
    wp, ap, lp, p0, sp = _gp_batch_params(theta_batch, IDX_P, NODES_P)
    sigma = theta_batch[:, IDX_SIGMA] ** 2
    if np.any(sigma <= 0.0) or not np.all(np.isfinite(sigma)):
        raise ValueError("noise intensities must be positive and finite")

    prop = np.empty((n_batch, M_MODES, 2, 2))
    root = np.empty((n_batch, M_MODES, 2, 2))
    for b in range(n_batch):
        prop[b], root[b] = _van_loan(OMEGA2, sq[b], sp[b], sigma[b], dt)
    p00, p01 = prop[:, :, 0, 0], prop[:, :, 0, 1]
    p10, p11 = prop[:, :, 1, 0], prop[:, :, 1, 1]
    l00, l01 = root[:, :, 0, 0], root[:, :, 0, 1]
    l10, l11 = root[:, :, 1, 0], root[:, :, 1, 1]

    n_burn = int(round(t_burn / dt))
    n_steps = int(round(t_record / dt))
    trace, det = sq + sp, sq[:, None] * sp[:, None] + OMEGA2[None, :]
    stationary = (trace < 0.0) & np.all(det > 0.0, axis=1)

    # The gate, applied before any stepping: a closure with no invariant measure
    # is rejected outright rather than simulated and read out.  Doing it here
    # also skips the work.  If every member fails there is nothing to integrate.
    if STATIONARITY_GATE and not np.any(stationary):
        return (np.zeros((n_batch, n_steps, len(GAUGES))),
                np.zeros((n_batch, n_steps, len(GAUGES))),
                np.ones(n_batch, dtype=bool))

    q = np.zeros((n_batch, M_MODES))
    qd = np.zeros((n_batch, M_MODES))
    if np.any(stationary):
        damping = np.where(stationary, -trace, 1.0)[:, None]
        start_q = rng.normal(size=(n_batch, M_MODES))
        start_p = rng.normal(size=(n_batch, M_MODES))
        q = np.where(stationary[:, None],
                     start_q * np.sqrt(sigma / (damping * det)), 0.0)
        qd = np.where(stationary[:, None],
                      start_p * np.sqrt(sigma / damping), 0.0)

    kx = np.outer(GAUGES, K_GRID)
    cq, sqx = np.cos(kx), np.sin(kx)
    eta = np.empty((n_batch, n_steps, len(GAUGES)))
    v = np.empty((n_batch, n_steps, len(GAUGES)))
    failed = ~stationary if STATIONARITY_GATE else np.zeros(n_batch, dtype=bool)

    chunk = 4000
    q_block = np.empty((chunk, n_batch, M_MODES))
    qd_block = np.empty((chunk, n_batch, M_MODES))
    total = n_burn + n_steps
    written = 0
    q0c, p0c = q0[:, None], p0[:, None]
    sqc, spc = sq[:, None], sp[:, None]

    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        # ONE noise draw for the whole batch: the spec's common random numbers.
        z = rng.normal(size=(stop - start, M_MODES, 2))
        kept = 0
        for offset in range(stop - start):
            fq = _gp_batch_eval(q, NODES_Q, wq, aq, lq)
            fp = _gp_batch_eval(qd, NODES_P, wp, ap, lp)
            q = q + (fq - sqc * q) * dt
            qd = qd + (fp - spc * qd) * dt
            z0, z1 = z[offset, :, 0], z[offset, :, 1]
            q, qd = (p00 * q + p01 * qd + l00 * z0 + l01 * z1,
                     p10 * q + p11 * qd + l10 * z0 + l11 * z1)
            step = start + offset
            if step >= n_burn:
                q_block[kept] = q
                qd_block[kept] = qd
                kept += 1
        # Guard once per block, not once per step: the check was 18% of the
        # runtime and a blow-up cannot hide inside 4000 steps -- it is
        # exponential, so it is either absent or enormous by the block end.
        bad = ~np.isfinite(qd).all(axis=1) | ~np.isfinite(q).all(axis=1) \
            | (np.abs(qd).max(axis=1) > max_abs) | (np.abs(q).max(axis=1) > max_abs)
        if np.any(bad):
            failed |= bad
            q[bad] = 0.0
            qd[bad] = 0.0
        if kept:
            qs = q_block[:kept]                              # (kept, batch, M)
            qds = qd_block[:kept]
            eta[:, written:written + kept] = np.einsum(
                "kbm,gm->bkg", qs, cq) - np.einsum(
                "kbm,gm->bkg", qds / OMEGA, sqx)
            v[:, written:written + kept] = np.einsum(
                "kbm,gm->bkg", qs * OMEGA, sqx) + np.einsum(
                "kbm,gm->bkg", qds, cq)
            written += kept
    return eta, v, failed


def statistics_batch(theta_batch, *, rng, t_record):
    """The 38-vector for each member of a batch; failed members get the sentinel."""

    eta, v, failed = simulate_batch(theta_batch, rng=rng, t_record=t_record)
    out = np.empty((eta.shape[0], 38))
    for b in range(eta.shape[0]):
        if failed[b]:
            out[b] = FORWARD_SENTINEL
            continue
        stats = case.spatial_statistics(eta[b], v[b], AUTO_LAGS, CROSS_LAGS)
        out[b] = stats if np.all(np.isfinite(stats)) else FORWARD_SENTINEL
    return out


# --- coordinates ------------------------------------------------------------
def to_latent(theta):
    return encode_positive_columns(np.asarray(theta, dtype=float), POSITIVE)


def from_latent(latent):
    return decode_positive_columns(np.asarray(latent, dtype=float), POSITIVE)


def initial_ensemble(rng, n_members):
    """Node values as a decreasing ramp plus jitter; positive blocks log-uniform."""

    def block(nodes, slope_prior, jitter, length_prior):
        slope = np.exp(rng.uniform(*np.log(slope_prior), (n_members, 1)))
        values = (-slope * nodes[None, :]
                  + rng.uniform(-jitter, jitter, (n_members, nodes.size)))
        return np.hstack([
            values,
            np.exp(rng.uniform(*np.log(OBS_NOISE_PRIOR), (n_members, 1))),
            np.exp(rng.uniform(*np.log(AMPLITUDE_PRIOR), (n_members, 1))),
            np.exp(rng.uniform(*np.log(length_prior), (n_members, 1)))])

    return to_latent(np.hstack([
        block(NODES_Q, SLOPE_PRIOR_Q, NODE_JITTER_Q, LENGTHSCALE_PRIOR_Q),
        block(NODES_P, SLOPE_PRIOR_P, NODE_JITTER_P, LENGTHSCALE_PRIOR_P),
        np.exp(rng.uniform(*np.log(SIGMA_PRIOR), (n_members, M_MODES)))]))


def prior_mean():
    """Centre of the prior: the geometric-mean ramp and hyperparameters."""

    def block(nodes, slope_prior, length_prior):
        slope = np.sqrt(slope_prior[0] * slope_prior[1])
        return np.concatenate([
            -slope * nodes,
            [np.sqrt(OBS_NOISE_PRIOR[0] * OBS_NOISE_PRIOR[1]),
             np.sqrt(AMPLITUDE_PRIOR[0] * AMPLITUDE_PRIOR[1]),
             np.sqrt(length_prior[0] * length_prior[1])]])

    return np.concatenate([
        block(NODES_Q, SLOPE_PRIOR_Q, LENGTHSCALE_PRIOR_Q),
        block(NODES_P, SLOPE_PRIOR_P, LENGTHSCALE_PRIOR_P),
        np.full(M_MODES, np.sqrt(SIGMA_PRIOR[0] * SIGMA_PRIOR[1]))])


def parameter_labels():
    return ([f"Phi_q(q={NODES_Q[i]:+.1f})" for i in range(N_NODES)]
            + ["Phi_q:obs_noise", "Phi_q:amplitude", "Phi_q:lengthscale"]
            + [f"Phi_p(p={NODES_P[i]:+.1f})" for i in range(N_NODES)]
            + ["Phi_p:obs_noise", "Phi_p:amplitude", "Phi_p:lengthscale"]
            + [f"sqrt_sigma_{j + 1}" for j in range(M_MODES)])


def timestep_error(theta, *, seed=11, t_record=1000.0, refine=4, n_reps=64):
    """Discretisation bias: mean statistics at dt against dt/refine.

    Comparing single paths at two step sizes does NOT work here.  The fine run
    draws ``refine`` times as many random numbers, so seeding both the same way
    gives two different realisations and the difference measured is sampling
    noise, not discretisation.  A first attempt did exactly that and reported a
    13% "error" that moved non-monotonically with ``refine`` -- the giveaway.

    The bias is systematic and the sampling noise is not, so the comparison is
    between MEANS over ``n_reps`` independent realisations at each step size.
    The standard error of each mean is reported alongside, because a bias below
    it has not been measured, only bounded.
    """

    def batch(step):
        rows = []
        for i in range(n_reps):
            rng = np.random.default_rng(np.random.SeedSequence([seed, i]))
            eta, v = simulate(theta, rng=rng, t_record=t_record, dt=step)
            keep = int(round(DT_DATA / step))
            rows.append(case.spatial_statistics(eta[::keep], v[::keep],
                                                AUTO_LAGS, CROSS_LAGS))
        return np.asarray(rows)

    coarse = batch(DT_DATA)
    fine = batch(DT_DATA / refine)
    mean_c, mean_f = coarse.mean(axis=0), fine.mean(axis=0)
    scale = np.maximum(np.abs(mean_f), 1e-12)
    bias = np.abs(mean_c - mean_f) / scale
    stderr = (np.sqrt(coarse.var(axis=0, ddof=1) / n_reps
                      + fine.var(axis=0, ddof=1) / n_reps) / scale)
    return {"bias": bias, "stderr": stderr,
            "bias_median": float(np.median(bias)),
            "bias_max": float(bias.max()),
            "stderr_median": float(np.median(stderr)),
            "resolved": bool(np.median(bias) > 2.0 * np.median(stderr)),
            "n_reps": n_reps, "refine": refine, "t_record": t_record}
