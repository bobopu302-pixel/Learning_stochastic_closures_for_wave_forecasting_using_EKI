"""Shared estimator primitives: the statistics every case builds its data
vector y and forward output G(theta) from.

Assembly of these primitives into a case's observation vector (block order,
lags, bands, normalisation choices) lives in the case folders; only the
case-agnostic estimators live here.
"""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------
# Time-domain correlation estimators (linear wave case: gauge records)
# ----------------------------------------------------------------------


def normalized_autocorr(x, lag_steps):
    """Normalized autocorrelation rho(lag) of one time series.

    Demeans, then divides the lagged product mean by the variance; lag 0 (or
    a lag >= len(x)) returns 1.0, and a zero-variance series returns zeros.
    Used by the linear-wave case (single-gauge ACF block via gauge_acf).
    """

    x = x - x.mean()
    denom = np.mean(x * x)
    if denom <= 0:
        return np.zeros(len(lag_steps))
    out = np.empty(len(lag_steps))
    for i, k in enumerate(lag_steps):
        k = int(k)
        out[i] = np.mean(x[:-k] * x[k:]) / denom if 0 < k < x.size else 1.0
    return out


def gauge_acf(eta_field, lag_steps):
    """Mean single-gauge autocorrelation, averaged over all gauges.

    ``eta_field`` is (time, gauges).  Used by the linear-wave case.
    """

    acf = np.zeros(len(lag_steps))
    for k in range(eta_field.shape[1]):
        acf += normalized_autocorr(eta_field[:, k], lag_steps)
    return acf / eta_field.shape[1]


def xcorr_pair(a, b, lag):
    """Normalized cross-correlation E[a(t) b(t+lag)] / (std_a std_b) at one signed lag.

    Used by the linear-wave case (gauge-pair block via cross_corr).
    """

    a = a - a.mean()
    b = b - b.mean()
    denom = a.std() * b.std()
    if denom <= 0:
        return 0.0
    if lag > 0:
        val = np.mean(a[:-lag] * b[lag:])
    elif lag < 0:
        val = np.mean(a[-lag:] * b[:lag])
    else:
        val = np.mean(a * b)
    return float(val / denom)


def cross_corr(field, offset, lags):
    """Cross-correlation averaged over all gauge pairs separated by ``offset``.

    ``field`` is (time, gauges).  Used by the linear-wave case.
    """

    K = field.shape[1]
    out = np.zeros(len(lags))
    n_pairs = K - offset
    for k in range(n_pairs):
        for i, lag in enumerate(lags):
            out[i] += xcorr_pair(field[:, k], field[:, k + offset], int(lag))
    return out / max(n_pairs, 1)


# ----------------------------------------------------------------------
# Spectral estimator (linear wave case)
# ----------------------------------------------------------------------


def band_energy_spectrum(eta_field, grid_freqs, spec_df, dt):
    """Periodogram band energy at each grid frequency.

    Gauge-averaged, one-sided, ~variance per band: for each requested
    frequency, sums the one-sided periodogram over the band of half-width
    ``spec_df / 2``.  Used by the linear-wave case (band-energy block).
    """

    n = eta_field.shape[0]
    freqs = np.fft.rfftfreq(n, d=dt)
    psd = np.zeros(freqs.size)
    for k in range(eta_field.shape[1]):
        x = eta_field[:, k] - eta_field[:, k].mean()
        power = np.abs(np.fft.rfft(x)) ** 2 / n ** 2
        # One-sided convention: double every bin that has a negative-frequency
        # mirror (all but DC, and but Nyquist when n is even).
        if n % 2 == 0:
            power[1:-1] *= 2.0
        else:
            power[1:] *= 2.0
        psd += power
    psd /= eta_field.shape[1]
    out = np.empty(len(grid_freqs))
    for j, fj in enumerate(grid_freqs):
        out[j] = psd[np.abs(freqs - fj) <= 0.5 * spec_df].sum()
    return out


# ----------------------------------------------------------------------
# Scalar ACF (vKdV nonlinear case)
# ----------------------------------------------------------------------


def demeaned_acf(series: np.ndarray, lag: int) -> float:
    """Demeaned autocorrelation of one series at one lag (``lag`` >= 1).

    Denominator is the SUM of squares dot(x, x), not the mean -- the
    normalisation cancels in the ratio for long series but differs from
    normalized_autocorr by the (constant) sample count on the numerator side;
    kept exactly as the vKdV source.  Used by the vKdV nonlinear case
    (gauge-ACF entries of the 44-dim statistics vector).
    """

    values = np.asarray(series, dtype=float)
    values = values - np.mean(values)
    denominator = float(np.dot(values, values))
    if denominator <= 0.0:
        return 0.0
    return float(
        np.dot(values[:-lag], values[lag:]) / denominator
    )


# ----------------------------------------------------------------------
# Moment vectors (Lorenz 63/96 reproduction cases)
# ----------------------------------------------------------------------


def as_2d_samples(samples: np.ndarray) -> np.ndarray:
    """Coerce trajectories to shape (time, state_dimension) for consistency."""

    x = np.asarray(samples, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"Expected samples with shape (time, dim), got {x.shape}")
    return x


def _first_second_moments(
    samples: np.ndarray,
    n_components: int | None,
    *,
    centered: bool,
) -> np.ndarray:
    x = as_2d_samples(samples)
    if n_components is not None:
        x = x[:, :n_components]

    means = x.mean(axis=0)
    products = x - means[None, :] if centered else x
    dim = x.shape[1]

    seconds = []

    # Paper figure order: first the diagonal second moments, then the
    # off-diagonal cross moments.
    for i in range(dim):
        seconds.append(np.mean(products[:, i] * products[:, i]))
    for i in range(dim):
        for j in range(i, dim):
            if i != j:
                seconds.append(np.mean(products[:, i] * products[:, j]))
    return np.concatenate([means, np.asarray(seconds)])


def centered_first_second_moments(
    samples: np.ndarray, n_components: int | None = None
) -> np.ndarray:
    """Means followed by centered variances/covariances in paper-figure order.

    Used by the Lorenz 63/96 reproduction cases (figure-faithful convention).
    """

    return _first_second_moments(samples, n_components, centered=True)


def raw_first_second_moments(samples: np.ndarray, n_components: int | None = None) -> np.ndarray:
    """Means followed by raw products E[x_i x_j] in Remark 2.1 order.

    Used by the Lorenz 63/96 reproduction cases (Remark 2.1 convention).
    """

    return _first_second_moments(samples, n_components, centered=False)


def first_second_moments(samples: np.ndarray, n_components: int | None = None) -> np.ndarray:
    """Backward-compatible alias for the figure-faithful (centered) convention.

    New code must use one of the explicit names above so a cache cannot hide
    the centered/raw distinction.
    """

    return centered_first_second_moments(samples, n_components=n_components)


# ----------------------------------------------------------------------
# Covariance from repeated statistics, and histogram density
# ----------------------------------------------------------------------


def cov_from_samples(
    stat_samples: np.ndarray,
) -> np.ndarray:
    """Full covariance Gamma estimate from repeated finite-time statistics.

    ``stat_samples`` is (runs, q); preserves correlations between different
    observed statistics (ddof=1; a single run returns the zero matrix).  Used
    by the Lorenz reproduction cases (legacy full-Gamma path; the
    spec-2026-08-23 estimator with diagnostics is algorithms.gamma.build_gamma).
    """

    s = np.asarray(stat_samples, dtype=float)
    if s.ndim != 2:
        raise ValueError(f"Expected statistic samples with shape (runs, q), got {s.shape}")

    if s.shape[0] > 1:
        cov = np.cov(s, rowvar=False, ddof=1)
    else:
        cov = np.zeros((s.shape[1], s.shape[1]), dtype=float)

    return cov


def histogram_density(samples: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Probability-density histogram on explicit bin edges.

    Returns (bin centers, density).  Pure numpy; used by the Lorenz
    reproduction cases to build the density arrays persisted in bundle.npz.
    """

    edges = np.asarray(edges, dtype=float)
    density, _ = np.histogram(
        np.asarray(samples, dtype=float).reshape(-1), bins=edges, density=True
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, density
