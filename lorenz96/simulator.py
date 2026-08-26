"""Two-scale Lorenz-96 truth system and closed GP-closure model (numba simulators).

"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from numba import njit

# Make the shared `algorithms` package importable when this module is loaded
# from the case folder (the entry script run_spec.py does the same).
_CODE_RP_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_RP_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_RP_ROOT))

from algorithms.gpr import make_gp_mean_from_theta  # noqa: E402

# ---- truth-system constants shared by both Lorenz-96 cases in the paper ----
K_SLOW = 36     # number of slow variables X_k
J_FAST = 10     # fast variables Y_{k,j} per slow variable
F_FORCE = 10.0  # constant forcing
B_FAST = 10.0   # fast-variable advection coefficient

# ---- implementation choices not specified in the paper ----
DT_FULL = 0.005  # RK4 step for the truth system and Euler-Maruyama closure
STORE_FULL = 4   # store every 4th step (sampling cadence of all statistics)


# ----------------------------------------------------------------------
# Reference (pure numpy) right-hand side and RK4 step.  These are the
# readable definition of the dynamics; the numba kernels below were verified
# against them to one ULP per step.
# ----------------------------------------------------------------------


# Slow-variable Lorenz 96 tendency without the fast-variable coupling.
def slow_lorenz96_tendency(x: np.ndarray, F: float) -> np.ndarray:
    return -np.roll(x, 1) * (np.roll(x, 2) - np.roll(x, -1)) - x + F


# Right-hand side of the full two-scale Lorenz 96 system, Eq. (2.8).
# The paper uses one symbol ``h`` for coupling; this implementation names the
# two roles separately: ``h_slow`` multiplies the slow-equation closure term,
# ``h_fast`` multiplies the slow forcing of the fast variables.  Case (a) has
# h_slow = h_fast = 1.
def two_scale_lorenz96_rhs(
    x: np.ndarray,
    y_flat: np.ndarray,
    *,
    K: int,
    J: int,
    F: float,
    h_slow: float,
    h_fast: float,
    c: float,
    b: float,
) -> tuple[np.ndarray, np.ndarray]:
    y = y_flat.reshape(K, J)
    y_bar = np.mean(y, axis=1)
    dx = slow_lorenz96_tendency(x, F) - h_slow * c * y_bar
    forcing_from_x = (h_fast / J) * np.repeat(x, J)
    dy = c * (
        -b * np.roll(y_flat, -1) * (np.roll(y_flat, -2) - np.roll(y_flat, 1))
        - y_flat
        + forcing_from_x
    )
    return dx, dy


# One RK4 step of the full two-scale system (reference implementation).
def rk4_two_scale_step(
    x: np.ndarray,
    y: np.ndarray,
    dt: float,
    *,
    K: int,
    J: int,
    F: float,
    h_slow: float,
    h_fast: float,
    c: float,
    b: float,
) -> tuple[np.ndarray, np.ndarray]:
    arguments = dict(K=K, J=J, F=F, h_slow=h_slow, h_fast=h_fast, c=c, b=b)
    k1x, k1y = two_scale_lorenz96_rhs(x, y, **arguments)
    k2x, k2y = two_scale_lorenz96_rhs(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, **arguments)
    k3x, k3y = two_scale_lorenz96_rhs(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, **arguments)
    k4x, k4y = two_scale_lorenz96_rhs(x + dt * k3x, y + dt * k3y, **arguments)
    x_next = x + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
    y_next = y + (dt / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
    return x_next, y_next


# ----------------------------------------------------------------------
# Numba kernels
# ----------------------------------------------------------------------


# Numba core of the two-scale RK4 loop (2026-08-10 speedup: the pure-Python
# loop ran ~7k steps/s and dominated runtimes; this runs ~61k).  Arithmetic
# matches two_scale_lorenz96_rhs/rk4_two_scale_step to one ULP per step
# (verified); the y-mean reduction order differs from numpy's pairwise
# summation, so chaotic trajectories decorrelate from the reference path over
# ~5-10 time units while remaining statistically identical.  Returns the
# number of rows stored, or -1 on blow-up.
@njit(cache=True, nogil=True)
def _two_scale_core(
    x, y_flat, dt, n_steps, burn_steps, store_every,
    J, F, h_slow, h_fast, c, b, want_residual, x_out, closure_out,
):
    K = x.shape[0]
    index = 0
    n_store = x_out.shape[0]
    for step in range(n_steps):
        x1, y1 = _two_scale_rhs_nb(x, y_flat, J, F, h_slow, h_fast, c, b)
        x2, y2 = _two_scale_rhs_nb(x + 0.5 * dt * x1, y_flat + 0.5 * dt * y1, J, F, h_slow, h_fast, c, b)
        x3, y3 = _two_scale_rhs_nb(x + 0.5 * dt * x2, y_flat + 0.5 * dt * y2, J, F, h_slow, h_fast, c, b)
        x4, y4 = _two_scale_rhs_nb(x + dt * x3, y_flat + dt * y3, J, F, h_slow, h_fast, c, b)
        x = x + (dt / 6.0) * (x1 + 2.0 * x2 + 2.0 * x3 + x4)
        y_flat = y_flat + (dt / 6.0) * (y1 + 2.0 * y2 + 2.0 * y3 + y4)
        ok = True
        for k in range(K):
            if not np.isfinite(x[k]) or abs(x[k]) > 1e6:
                ok = False
                break
        if ok:
            for j in range(y_flat.shape[0]):
                if not np.isfinite(y_flat[j]):
                    ok = False
                    break
        if not ok:
            return -1
        if step >= burn_steps and (step - burn_steps) % store_every == 0:
            for k in range(K):
                x_out[index, k] = x[k]
            if want_residual:
                for k in range(K):
                    acc = 0.0
                    for j in range(J):
                        acc += y_flat[k * J + j]
                    y_bar_k = acc / J
                    closure_out[index, k] = -h_slow * c * y_bar_k + (h_slow * h_fast * c / J) * x[k]
            index += 1
            if index >= n_store:
                break
    return index


@njit(cache=True, nogil=True, inline="always")
def _two_scale_rhs_nb(x, y_flat, J, F, h_slow, h_fast, c, b):
    K = x.shape[0]
    y_bar = np.empty(K)
    for k in range(K):
        acc = 0.0
        for j in range(J):
            acc += y_flat[k * J + j]
        y_bar[k] = acc / J
    dx = -np.roll(x, 1) * (np.roll(x, 2) - np.roll(x, -1)) - x + F - h_slow * c * y_bar
    forcing_from_x = np.empty(K * J)
    for k in range(K):
        fx = (h_fast / J) * x[k]
        for j in range(J):
            forcing_from_x[k * J + j] = fx
    dy = c * (
        -b * np.roll(y_flat, -1) * (np.roll(y_flat, -2) - np.roll(y_flat, 1))
        - y_flat
        + forcing_from_x
    )
    return dx, dy


# Simulate the full two-scale truth system; returns slow variables (and
# optionally the closure residual for closure-scatter diagnostics).
def simulate_two_scale_lorenz96(
    *,
    K: int = K_SLOW,
    J: int = J_FAST,
    F: float = F_FORCE,
    h_slow: float = 1.0,
    h_fast: float = 1.0,
    c: float = 10.0,
    b: float = B_FAST,
    dt: float = DT_FULL,
    t_total: float = 60.0,
    burn_in: float = 20.0,
    rng: np.random.Generator | None = None,
    store_every: int = STORE_FULL,
    return_closure_residual: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    if rng is None:
        rng = np.random.default_rng()
    # Initial condition convention: both scales drawn U(0, 1) from the caller's
    # generator, so a seed fully determines the record.
    x = rng.uniform(0.0, 1.0, size=K)
    y = rng.uniform(0.0, 1.0, size=K * J)

    n_steps = int(round((t_total + burn_in) / dt))
    burn_steps = int(round(burn_in / dt))
    n_store = max((n_steps - burn_steps) // store_every, 1)
    x_out = np.empty((n_store, K), dtype=float)
    closure_out = np.empty((n_store if return_closure_residual else 1, K), dtype=float)
    index = _two_scale_core(
        x, y, float(dt), n_steps, burn_steps, int(store_every),
        J, float(F), float(h_slow), float(h_fast), float(c), float(b),
        bool(return_closure_residual), x_out, closure_out,
    )
    if index < 0:
        raise FloatingPointError("Two-scale Lorenz 96 became unstable")
    return x_out[:index], closure_out[:index] if return_closure_residual else None


# GP kernel-regression weights for the fast simulator.  Delegates to the
# shared algorithms.gpr construction; the returned triple is what the numba
# loop inlines: m(x) = sum_r a^2 exp(-((x - node_r)/l)^2 / 2) w_r.
def gp_weights_from_theta(
    theta_gp: np.ndarray, nodes: np.ndarray
) -> tuple[np.ndarray, float, float]:
    mean = make_gp_mean_from_theta(theta_gp, nodes)
    # 1-D means carry the (nodes, weights, amplitude, lengthscale) contract.
    _, weights, amplitude, lengthscale = mean._gp_params
    return weights, amplitude, lengthscale


# Numba-accelerated Euler-Maruyama loop for the closed Lorenz 96 GP model,
# Eq. (2.11): dX_k = (slow tendency - lambda X_k + m(X_k)) dt + sqrt(sigma) dW.
@njit(cache=True, nogil=True)
def _simulate_closed_lorenz96_gp_loop(
    x0,
    noise,
    nodes,
    weights,
    amplitude,
    lengthscale,
    sigma,
    linear_coefficient,
    F,
    dt,
    n_steps,
    burn_steps,
    store_every,
):
    K = x0.size
    n_store = max((n_steps - burn_steps) // store_every, 1)
    out = np.empty((n_store, K))
    x = x0.copy()
    x_next = np.empty(K)
    index = 0
    amplitude_squared = amplitude * amplitude
    noise_scale = np.sqrt(sigma * dt)
    for step in range(n_steps):
        for k in range(K):
            xm1 = x[(k - 1) % K]
            xm2 = x[(k - 2) % K]
            xp1 = x[(k + 1) % K]
            slow_tendency = -xm1 * (xm2 - xp1) - x[k] + F
            gp_value = 0.0
            for r in range(nodes.size):
                scaled_distance = (x[k] - nodes[r]) / lengthscale
                kernel_value = amplitude_squared * np.exp(-0.5 * scaled_distance * scaled_distance)
                gp_value += kernel_value * weights[r]
            drift = slow_tendency - linear_coefficient * x[k] + gp_value
            x_next[k] = x[k] + drift * dt
            if sigma > 0.0:
                x_next[k] += noise_scale * noise[step, k]
        for k in range(K):
            x[k] = x_next[k]
        if step >= burn_steps and (step - burn_steps) % store_every == 0:
            for k in range(K):
                out[index, k] = x[k]
            index += 1
            if index >= n_store:
                break
    return out[:index]


# Euler-Maruyama simulation of the slow-variable closure (EKI forward runs and
# long evaluation runs).  ``sigma`` is the noise VARIANCE (the driver squares
# the evolved sqrt(sigma) parameter before calling); sigma = 0 gives the ODE
# closure on the same code path with no noise array allocated.
def simulate_closed_lorenz96_gp_fast(
    *,
    theta_gp: np.ndarray,
    sigma: float,
    linear_coefficient: float,
    nodes: np.ndarray,
    K: int = K_SLOW,
    F: float = F_FORCE,
    dt: float = 0.01,
    t_total: float = 40.0,
    burn_in: float = 10.0,
    rng: np.random.Generator | None = None,
    store_every: int = 1,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    if not np.isfinite(sigma) or sigma < 0.0:
        raise ValueError("sigma must be finite and nonnegative")
    if not np.isfinite(linear_coefficient) or linear_coefficient <= 0.0:
        raise ValueError("linear_coefficient must be finite and strictly positive")
    n_steps = int(round((t_total + burn_in) / dt))
    burn_steps = int(round(burn_in / dt))
    x0 = rng.uniform(0.0, 1.0, size=K)
    noise = rng.normal(size=(n_steps, K)) if sigma > 0.0 else np.zeros((1, 1), dtype=float)
    weights, amplitude, lengthscale = gp_weights_from_theta(theta_gp, nodes)
    out = _simulate_closed_lorenz96_gp_loop(
        x0,
        noise,
        np.asarray(nodes, dtype=float),
        weights,
        amplitude,
        lengthscale,
        float(sigma),
        float(linear_coefficient),
        float(F),
        float(dt),
        int(n_steps),
        int(burn_steps),
        int(store_every),
    )
    if not np.all(np.isfinite(out)) or np.max(np.abs(out)) > 1e6:
        raise FloatingPointError("Closed Lorenz 96 became unstable")
    return out


# ----------------------------------------------------------------------
# Truth-trajectory disk cache
# ----------------------------------------------------------------------


def cached_truth_trajectory(
    cache_path: Path | str,
    *,
    t_total: float,
    burn_in: float,
    seed: int,
    K: int = K_SLOW,
    J: int = J_FAST,
    F: float = F_FORCE,
    h_slow: float = 1.0,
    h_fast: float = 1.0,
    c: float = 10.0,
    b: float = B_FAST,
    dt: float = DT_FULL,
    store_every: int = STORE_FULL,
) -> np.ndarray:
    """Load (or simulate and cache) a long truth trajectory of slow variables.

    Long truth runs (e.g. t_total = 1000 for invariant-measure comparisons)
    are the slowest single simulations in the case; this cache lets them be
    reused across analyses.  The cache is valid only if EVERY stored parameter
    (including the seed) matches the request; otherwise the trajectory is
    recomputed and the file overwritten.  Returns the (time, K) slow-variable
    record after burn-in.
    """

    cache_path = Path(cache_path)
    parameters = dict(
        K=int(K), J=int(J), F=float(F), h_slow=float(h_slow), h_fast=float(h_fast),
        c=float(c), b=float(b), dt=float(dt), store_every=int(store_every),
        t_total=float(t_total), burn_in=float(burn_in), seed=int(seed),
    )
    if cache_path.exists():
        cached = np.load(cache_path)
        if all(
            name in cached and float(cached[name]) == float(value)
            for name, value in parameters.items()
        ):
            return cached["truth_eval"]
    slow, _ = simulate_two_scale_lorenz96(
        K=K, J=J, F=F, h_slow=h_slow, h_fast=h_fast, c=c, b=b,
        dt=dt, t_total=t_total, burn_in=burn_in,
        rng=np.random.default_rng(int(seed)), store_every=store_every,
        return_closure_residual=False,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, truth_eval=slow, **parameters)
    return slow
