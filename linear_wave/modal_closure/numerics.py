"""Wave physics for the linear-wave closure: dispersion relation and the exact
Ornstein-Uhlenbeck modal propagator.

Origin: 2.Linear_wave_case/modal_closure/numerics.py
Changes vs origin:
- the summary-statistic estimators (normalized_autocorr, gauge_acf, xcorr_pair,
  cross_corr, band_energy_spectrum) moved to the shared package
  algorithms.statistics (numerically identical there); this module keeps only
  what knows it is a wave;
- comments/docstrings polished.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

GRAVITY = 9.81


def dispersion(k, depth):
    """Linear dispersion relation omega^2 = g k tanh(k h)."""

    k = np.asarray(k, dtype=float)
    return GRAVITY * k * np.tanh(k * depth)


def modal_propagator(omega2, delta, sigma, dt):
    """Exact one-step propagator and noise root of the damped oscillator SDE.

    For each mode j the state (q_j, p_j) follows
    dq = p dt, dp = (-omega^2 q - delta p) dt + sqrt(sigma) dW.  Returns
    Phi = exp(A dt) and a matrix square root L with L L^T equal to the
    one-step noise covariance, computed from the stationary covariance
    P = diag(sigma/(2 delta omega^2), sigma/(2 delta)) via the exact discrete
    Lyapunov identity Q = P - Phi P Phi^T.  Modes with sigma or delta <= 0 get
    a zero noise root.
    """

    M = omega2.size
    phi = np.empty((M, 2, 2))
    noise_root = np.zeros((M, 2, 2))
    for j in range(M):
        a = np.array([[0.0, 1.0], [-omega2[j], -delta[j]]])
        phi_j = expm(a * dt)
        phi[j] = phi_j
        if sigma[j] > 0 and delta[j] > 0:
            p = np.array([[sigma[j] / (2.0 * delta[j] * omega2[j]), 0.0],
                          [0.0, sigma[j] / (2.0 * delta[j])]])
            q_cov = p - phi_j @ p @ phi_j.T
            q_cov = 0.5 * (q_cov + q_cov.T)
            w, vecs = np.linalg.eigh(q_cov)
            noise_root[j] = vecs @ np.diag(np.sqrt(np.clip(w, 0.0, None)))
    return phi, noise_root
