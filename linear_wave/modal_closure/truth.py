"""Random-phase truth generator: a 100-component broadband linear wave field.
"""

from numbers import Integral

import numpy as np
from scipy.optimize import brentq

GRAVITY = 9.81

# Time samples evaluated per block inside get_linear_data.  Purely a memory
# knob; it does not affect the returned values.
TIME_BLOCK = 4000


def _as_1d_float_array(name, values):
    """Convert scalars and array-like inputs to a 1D float array."""
    array = np.atleast_1d(np.asarray(values, dtype=float))
    if array.ndim != 1:
        raise ValueError(f"{name} must be a scalar or a 1D array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite numbers.")
    return array


def _solve_wavenumber(omega, depth, gravity=GRAVITY):
    """Solve omega^2 = g k tanh(k h) for positive k."""
    if omega <= 0:
        raise ValueError("frequencies must be positive.")

    def dispersion_residual(k):
        return gravity * k * np.tanh(k * depth) - omega**2

    lower = np.finfo(float).tiny
    upper = max(omega**2 / gravity, omega / np.sqrt(gravity * depth), 1.0 / depth)
    while dispersion_residual(upper) <= 0:
        upper *= 2.0

    return brentq(dispersion_residual, lower, upper)


def get_linear_data(
    gauge_locations, frequencies, amplitudes, output_times, depth, runs=1, random_seed=None
):
    """Surface elevation eta and its exact time derivative v at the gauges.

    Each run draws fresh uniform initial phases from ``random_seed`` (one
    shared generator across the ``runs``, consumed in order) and superposes
    right-travelling plane waves with the given fixed amplitudes; frequencies
    map to wavenumbers through the finite-depth dispersion relation.  Returns
    two lists of (time, gauge) arrays, one entry per run.
    """

    gauge_locations = _as_1d_float_array("gauge_locations", gauge_locations)
    frequencies = _as_1d_float_array("frequencies", frequencies)
    amplitudes = _as_1d_float_array("amplitudes", amplitudes)
    output_times = _as_1d_float_array("output_times", output_times)
    try:
        depth = float(depth)
    except (TypeError, ValueError) as exc:
        raise ValueError("depth must be a positive finite number.") from exc

    if len(frequencies) != len(amplitudes):
        raise ValueError("frequencies and amplitudes must have the same length.")
    if np.any(frequencies <= 0):
        raise ValueError("frequencies must be positive.")
    if depth <= 0 or not np.isfinite(depth):
        raise ValueError("depth must be a positive finite number.")
    if not isinstance(runs, Integral) or isinstance(runs, bool) or runs < 1:
        raise ValueError("runs must be a positive integer.")

    omega = 2 * np.pi * frequencies
    k_values = np.array([_solve_wavenumber(omega_i, depth) for omega_i in omega])
    rng = np.random.default_rng(random_seed)

    def run_one_seed():
        initial_phases = rng.uniform(0, 2 * np.pi, len(frequencies))
        n_times = output_times.size
        eta_array = np.empty((n_times, gauge_locations.size))
        deta_dt_array = np.empty_like(eta_array)
        # The full (time, gauge, component) phase tensor is the memory
        # bottleneck: at a 4000 s record it is 1.3 GB as complex128, which
        # multiplies by the number of worker processes.  Blocking the time axis
        # keeps the peak flat without changing the arithmetic -- the reduction
        # over components is still over the same 100 terms in the same order,
        # so results are bit-identical to the unblocked expression.
        space_phase = (
            initial_phases[None, :] + k_values[None, :] * gauge_locations[:, None]
        )
        for start in range(0, n_times, TIME_BLOCK):
            stop = min(start + TIME_BLOCK, n_times)
            phase = (
                space_phase[None, :, :]
                - omega[None, None, :] * output_times[start:stop, None, None]
            )
            waves = amplitudes[None, None, :] * np.exp(1j * phase)
            eta_array[start:stop] = np.real(np.sum(waves, axis=2))
            deta_dt_array[start:stop] = np.imag(
                np.sum(omega[None, None, :] * waves, axis=2)
            )
        return eta_array, deta_dt_array

    eta_arrays = [None for _ in range(runs)]
    deta_dt_arrays = [None for _ in range(runs)]
    for i in range(runs):
        eta_arrays[i], deta_dt_arrays[i] = run_one_seed()
    return eta_arrays, deta_dt_arrays
