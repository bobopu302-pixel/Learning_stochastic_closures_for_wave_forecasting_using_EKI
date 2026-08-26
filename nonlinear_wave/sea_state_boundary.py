"""Exact prescribed finite-depth TMA boundary for the coastal vKdV PDE.

Origin: 3. KDV_nonlinear_case/sea_state_boundary.py
Changes vs origin: comments/docstrings only (this provenance header added).

The module generates one complete random-phase sea-surface record from a
finite-depth TMA spectrum.  The random seed is fixed, so the prescribed
boundary is a deterministic and reproducible function in every PDE run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class SeaStateParameters:
    """Parameters of the prescribed finite-depth random-phase sea state."""

    significant_wave_height_m: float = 0.3
    peak_period_s: float = 15.0
    peak_enhancement_gamma: float = 3.3
    water_depth_m: float = 20.0
    gravity_m_s2: float = 9.81
    frequency_min_hz: float = 0.03
    high_frequency_taper_start_hz: float = 0.085
    frequency_max_hz: float = 0.105
    synthesis_period_s: float = 1800.0
    ramp_duration_s: float = 150.0
    random_seed: int = 20260718


def tma_finite_depth_factor(
    frequency_hz: np.ndarray,
    depth_m: float,
    gravity_m_s2: float = 9.81,
) -> np.ndarray:
    """Return the standard piecewise TMA finite-depth transformation."""

    omega_h = 2.0 * np.pi * np.asarray(frequency_hz) * np.sqrt(
        depth_m / gravity_m_s2
    )
    factor = np.ones_like(omega_h)
    low = omega_h <= 1.0
    middle = (omega_h > 1.0) & (omega_h < 2.0)
    factor[low] = 0.5 * omega_h[low] ** 2
    factor[middle] = 1.0 - 0.5 * (2.0 - omega_h[middle]) ** 2
    return factor


def _upper_cosine_taper(
    frequency_hz: np.ndarray,
    start_hz: float,
    stop_hz: float,
) -> np.ndarray:
    """Apply the prescribed smooth high-frequency truncation."""

    taper = np.ones_like(frequency_hz)
    taper[frequency_hz >= stop_hz] = 0.0
    transition = (frequency_hz > start_hz) & (frequency_hz < stop_hz)
    phase = (frequency_hz[transition] - start_hz) / (stop_hz - start_hz)
    taper[transition] = 0.5 * (1.0 + np.cos(np.pi * phase))
    return taper


def build_tma_spectrum(
    parameters: SeaStateParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Return harmonic frequencies and the normalised discrete TMA spectrum."""

    p = parameters
    delta_f = 1.0 / p.synthesis_period_s
    first = int(np.ceil(p.frequency_min_hz / delta_f))
    last = int(np.floor(p.frequency_max_hz / delta_f))
    frequencies = delta_f * np.arange(first, last + 1, dtype=float)
    fp = 1.0 / p.peak_period_s
    sigma = np.where(frequencies <= fp, 0.07, 0.09)
    peak_shape = np.exp(
        -0.5 * ((frequencies - fp) / (sigma * fp)) ** 2
    )
    # Alpha is omitted because the final discrete spectrum is explicitly
    # rescaled to the requested significant wave height.
    jonswap_shape = (
        frequencies ** -5.0
        * np.exp(-1.25 * (fp / frequencies) ** 4)
        * p.peak_enhancement_gamma**peak_shape
    )
    finite_depth = tma_finite_depth_factor(
        frequencies,
        p.water_depth_m,
        p.gravity_m_s2,
    )
    taper = _upper_cosine_taper(
        frequencies,
        p.high_frequency_taper_start_hz,
        p.frequency_max_hz,
    )
    spectrum = jonswap_shape * finite_depth * taper
    target_m0 = (p.significant_wave_height_m / 4.0) ** 2
    spectrum *= target_m0 / (np.sum(spectrum) * delta_f)
    return frequencies, spectrum


class TMASeaState:
    """One fixed random-phase, finite-depth TMA boundary realisation."""

    def __init__(self, parameters: SeaStateParameters) -> None:
        self.parameters = parameters
        self.frequencies_hz, self.spectrum_m2_hz = build_tma_spectrum(
            parameters
        )
        self.delta_f_hz = 1.0 / parameters.synthesis_period_s
        self.amplitudes_m = np.sqrt(
            2.0 * self.spectrum_m2_hz * self.delta_f_hz
        )
        rng = np.random.default_rng(parameters.random_seed)
        self.phases_rad = rng.uniform(
            0.0,
            2.0 * np.pi,
            self.frequencies_hz.size,
        )

    def stationary_signal_m(
        self,
        time_s: float | np.ndarray,
    ) -> float | np.ndarray:
        """Evaluate the complete unramped boundary record in metres."""

        time = np.asarray(time_s, dtype=float)
        phases = (
            2.0 * np.pi * time[..., None] * self.frequencies_hz
            + self.phases_rad
        )
        values = np.sum(self.amplitudes_m * np.cos(phases), axis=-1)
        return float(values) if values.ndim == 0 else values

    def ramp(self, time_s: float | np.ndarray) -> float | np.ndarray:
        """Return the half-cosine start-up ramp."""

        time = np.asarray(time_s, dtype=float)
        ramp = np.ones_like(time)
        ramp[time <= 0.0] = 0.0
        active = (time > 0.0) & (
            time < self.parameters.ramp_duration_s
        )
        ramp[active] = 0.5 * (
            1.0
            - np.cos(
                np.pi
                * time[active]
                / self.parameters.ramp_duration_s
            )
        )
        return float(ramp) if ramp.ndim == 0 else ramp

    def exact_m(
        self,
        time_s: float | np.ndarray,
    ) -> float | np.ndarray:
        """Evaluate the complete prescribed boundary after start-up ramping."""

        values = np.asarray(self.ramp(time_s)) * np.asarray(
            self.stationary_signal_m(time_s)
        )
        return float(values) if values.ndim == 0 else values

    # Backward-compatible exact-boundary name used by the frozen PDE solver.
    truth_m = exact_m

    def metadata(self) -> dict[str, object]:
        """Return reproducibility metadata for the exact boundary."""

        discrete_m0 = float(
            np.sum(self.spectrum_m2_hz) * self.delta_f_hz
        )
        return {
            "boundary_kind": "exact_prescribed_tma",
            "sea_state": asdict(self.parameters),
            "frequency_spacing_hz": self.delta_f_hz,
            "mode_count": int(self.frequencies_hz.size),
            "discrete_m0_m2": discrete_m0,
            "discrete_Hs_m": float(4.0 * np.sqrt(discrete_m0)),
        }


class DimensionlessExactBoundarySignal:
    """Adapt the dimensional exact TMA record to ``eta/a_ref`` at time ``T``."""

    name = "exact"
    label = "Exact prescribed TMA boundary"

    def __init__(
        self,
        sea_state: TMASeaState,
        time_ref_s: float,
        amplitude_ref_m: float,
    ) -> None:
        self.sea_state = sea_state
        self.time_ref_s = float(time_ref_s)
        self.amplitude_ref_m = float(amplitude_ref_m)

    def stationary_m(
        self,
        time_s: float | np.ndarray,
    ) -> float | np.ndarray:
        return self.sea_state.stationary_signal_m(time_s)

    def dimensional_m(
        self,
        time_s: float | np.ndarray,
    ) -> float | np.ndarray:
        return self.sea_state.exact_m(time_s)

    def __call__(
        self,
        time_dimensionless: float | np.ndarray,
    ) -> float | np.ndarray:
        time_s = np.asarray(time_dimensionless, dtype=float) * self.time_ref_s
        values = (
            np.asarray(self.dimensional_m(time_s)) / self.amplitude_ref_m
        )
        return float(values) if values.ndim == 0 else values

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "label": self.label,
            "boundary_kind": "exact_prescribed_tma",
        }


def make_exact_tma_boundary(
    sea_state_parameters: SeaStateParameters,
    time_ref_s: float,
    amplitude_ref_m: float,
) -> tuple[TMASeaState, DimensionlessExactBoundarySignal]:
    """Construct the only boundary signal admitted by the PDE workflow."""

    sea_state = TMASeaState(sea_state_parameters)
    signal = DimensionlessExactBoundarySignal(
        sea_state,
        time_ref_s,
        amplitude_ref_m,
    )
    return sea_state, signal
