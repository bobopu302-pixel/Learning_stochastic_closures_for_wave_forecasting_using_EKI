"""Euler-Maruyama simulator for the noisy Lorenz 63 system (simulation library only).

Origin: 1. Reproduce_papers/Lorenz63/code/run_lorenz63.py
Changes vs origin:
- extracted ONLY the simulation library (lorenz63_drift, _sim_core,
  simulate_lorenz63, _simulate_lorenz63_python and their constants); the
  legacy experiment driver (main, EKI fits, bundle/figure assembly) and its
  lorenz_plots / covariance / protocol / run_manifest imports were dropped --
  the 2026-08-23 spec driver in run_spec.py replaces them;
- comments/docstrings polished; simulation numerics untouched.

Model: additive-noise Lorenz 63

    dx1 = alpha (x2 - x1) dt                     + sqrt(sigma) dW1
    dx2 = (x1 (rho - x3) - g_L(x2)) dt           + sqrt(sigma) dW2
    dx3 = (x1 x2 - beta x3) dt                   + sqrt(sigma) dW3

with g_L(x2) = x2 (the true closure) or a learned GP conditional mean.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numba import njit


# Deterministic Lorenz 63 drift; g_func replaces x2 in the second equation.
# Used only by the pure-Python fallback loop for arbitrary g_func callables.
def lorenz63_drift(
    x: np.ndarray,
    *,
    alpha: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    g_func: Callable[[float], float] | None = None,
) -> np.ndarray:

    x1, x2, x3 = x
    g_value = x2 if g_func is None else g_func(float(x2))
    return np.asarray(
        [
            alpha * (x2 - x1),
            x1 * (rho - x3) - g_value,
            x1 * x2 - beta * x3,
        ],
        dtype=float,
    )


# Numba core of the Euler-Maruyama loop (the pure-Python loop ran ~40k steps/s,
# far too slow for the spec's T = 1500 forward windows).
# g_mode 0 = linear g(x2)=x2; g_mode 1 = GP kernel-regression mean
# m(x2) = sum_j amplitude^2 exp(-((x2-node_j)/lengthscale)^2/2) w_j.
# Arithmetic and noise-stream order match the original Python loop exactly
# (drift from the pre-update state, then additive noise, row-major normals).
# Returns the number of stored rows, or -1 on blow-up.
@njit(cache=True)
def _sim_core(
    x0, alpha, rho, beta, g_mode, nodes, weights, amplitude, lengthscale,
    dt, noise_scale, n_steps, burn_steps, store_every, noise, out,
):
    x1, x2, x3 = x0[0], x0[1], x0[2]
    n_store = out.shape[0]
    index = 0
    for step in range(n_steps):
        if g_mode == 0:
            g_value = x2
        else:
            g_value = 0.0
            for j in range(nodes.size):
                d = (x2 - nodes[j]) / lengthscale
                g_value += amplitude * amplitude * np.exp(-0.5 * d * d) * weights[j]
        d1 = alpha * (x2 - x1)
        d2 = x1 * (rho - x3) - g_value
        d3 = x1 * x2 - beta * x3
        x1 = x1 + d1 * dt
        x2 = x2 + d2 * dt
        x3 = x3 + d3 * dt
        if noise_scale > 0.0:
            x1 = x1 + noise_scale * noise[step, 0]
            x2 = x2 + noise_scale * noise[step, 1]
            x3 = x3 + noise_scale * noise[step, 2]
        if not (np.isfinite(x1) and np.isfinite(x2) and np.isfinite(x3)):
            return -1
        if abs(x1) > 1e6 or abs(x2) > 1e6 or abs(x3) > 1e6:
            return -1
        if step >= burn_steps and (step - burn_steps) % store_every == 0:
            out[index, 0] = x1
            out[index, 1] = x2
            out[index, 2] = x3
            index += 1
            if index >= n_store:
                break
    return index


_NO_NODES = np.zeros(0, dtype=float)


# Euler-Maruyama simulation of the Lorenz 63 ODE/SDE; returns the post-burn-in
# trajectory of shape (n_store, 3).
# Fast paths: g_func None (linear closure) or an algorithms.gpr.make_gp_mean
# closure (its _gp_params tuple is inlined into the numba kernel).  Any other
# callable falls back to the original pure-Python loop.
def simulate_lorenz63(
    *,
    alpha: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    sigma: float = 10.0,
    dt: float = 0.01,
    t_total: float = 40.0,
    burn_in: float = 10.0,
    x0: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    store_every: int = 1,
    g_func: Callable[[float], float] | None = None,
) -> np.ndarray:

    if rng is None:
        rng = np.random.default_rng()
    if x0 is None:
        x = rng.uniform(0.0, 1.0, size=3)
    else:
        x = np.asarray(x0, dtype=float).copy()

    n_steps = int(round((t_total + burn_in) / dt))
    burn_steps = int(round(burn_in / dt))
    n_store = max((n_steps - burn_steps) // store_every, 1)
    out = np.empty((n_store, 3), dtype=float)

    if not np.isfinite(sigma) or sigma < 0.0:
        raise ValueError("sigma must be finite and nonnegative")
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be finite and strictly positive")
    # dW ~ sqrt(dt) * N(0, 1), so sqrt(sigma) dW has scale sqrt(sigma*dt).
    noise_scale = np.sqrt(float(sigma) * dt)

    if g_func is None:
        g_mode, nodes, weights, amplitude, lengthscale = 0, _NO_NODES, _NO_NODES, 1.0, 1.0
    elif hasattr(g_func, "_gp_params"):
        nodes, weights, amplitude, lengthscale = g_func._gp_params
        g_mode = 1
    else:
        return _simulate_lorenz63_python(
            x=x, alpha=alpha, rho=rho, beta=beta, sigma=sigma, dt=dt,
            n_steps=n_steps, burn_steps=burn_steps, store_every=store_every,
            rng=rng, g_func=g_func, out=out,
        )

    # Same variate order as drawing normal(size=3) once per step.
    noise = (
        rng.normal(size=(n_steps, 3)) if sigma > 0.0 else np.zeros((1, 3), dtype=float)
    )
    stored = _sim_core(
        x, float(alpha), float(rho), float(beta), g_mode,
        np.asarray(nodes, dtype=float), np.asarray(weights, dtype=float),
        float(amplitude), float(lengthscale),
        float(dt), float(noise_scale) if sigma > 0.0 else 0.0,
        n_steps, burn_steps, int(store_every), noise, out,
    )
    if stored < 0:
        raise FloatingPointError("Lorenz 63 simulation became unstable")
    return out[:stored]


# Original pure-Python loop, kept as the fallback for arbitrary g_func
# callables (not used by the production experiments).
def _simulate_lorenz63_python(
    *, x, alpha, rho, beta, sigma, dt, n_steps, burn_steps, store_every, rng, g_func, out
):
    noise_scale = np.sqrt(float(sigma) * dt)
    index = 0
    n_store = out.shape[0]
    for step in range(n_steps):
        x = x + lorenz63_drift(x, alpha=alpha, rho=rho, beta=beta, g_func=g_func) * dt
        if sigma > 0.0:
            x = x + noise_scale * rng.normal(size=3)
        if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > 1e6:
            raise FloatingPointError("Lorenz 63 simulation became unstable")
        if step >= burn_steps and (step - burn_steps) % store_every == 0:
            out[index] = x
            index += 1
            if index >= n_store:
                break
    return out[:index]
