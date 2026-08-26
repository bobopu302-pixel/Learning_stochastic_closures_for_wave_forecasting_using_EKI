"""Three-trace modal incident lifting for the C6/C4--CN KdV discretisation.

Origin: 3. KDV_nonlinear_case/high_order_incident_lifting.py
Changes vs origin: comments/docstrings only (this provenance header added).

The physical boundary supplies one real, mean-zero, periodic stationary record
``g(T)`` at the offshore node.  Its positive-frequency Fourier coefficients
are extracted once.  For every retained angular frequency ``omega``, the
long-wave phase increment ``theta`` is obtained from the fully discrete
dispersion relation

    (2/dt) tan(omega*dt/2) = Omega_sd(theta/dy),

where ``Omega_sd`` is the C6-D1/C4-D3 semidiscrete symbol.  The three numerical
traces are then generated from the same coefficients with phase multipliers
``exp(-i*q*theta)``, q=0,1,2.

The half-cosine start-up is delayed mode by mode to avoid an acausal onset.
That construction is exactly phase matched for the stationary carriers after
spin-up; it is not an exact Z-domain lifting of the ramp sidebands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import brentq

from high_order_matched_dabc import (
    fourth_order_cn_group_velocity,
    fourth_order_semidiscrete_omega,
)


def _as_positive_frequency_array(values: np.ndarray | float) -> np.ndarray:
    frequencies = np.atleast_1d(np.asarray(values, dtype=float))
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("angular frequencies must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
        raise ValueError("angular frequencies must be finite and positive")
    return frequencies


def half_cosine_ramp(
    time: np.ndarray | float,
    duration: float,
) -> np.ndarray | float:
    """Return a causal half-cosine ramp, equal to one after ``duration``."""

    if duration <= 0.0:
        raise ValueError("ramp duration must be positive")
    values = np.asarray(time, dtype=float)
    ramp = np.ones_like(values)
    ramp[values <= 0.0] = 0.0
    active = (values > 0.0) & (values < duration)
    ramp[active] = 0.5 * (1.0 - np.cos(np.pi * values[active] / duration))
    return float(ramp) if ramp.ndim == 0 else ramp


def discrete_long_branch_phase_increments(
    angular_frequency: np.ndarray | float,
    advection: float,
    dispersion: float,
    dy: float,
    dt: float,
    *,
    d1_order: int = 6,
    branch_samples: int = 60001,
) -> tuple[np.ndarray, dict[str, float]]:
    """Invert the C6/C4--CN dispersion relation on its first long-wave branch."""

    omega = _as_positive_frequency_array(angular_frequency)
    if advection <= 0.0 or dispersion <= 0.0 or dy <= 0.0 or dt <= 0.0:
        raise ValueError("advection, dispersion, dy and dt must be positive")
    if d1_order not in (4, 6):
        raise ValueError("d1_order must be 4 or 6")
    if branch_samples < 1001:
        raise ValueError("branch_samples must be at least 1001")
    if np.any(omega * dt >= np.pi):
        raise ValueError("a forcing frequency reaches the CN temporal Nyquist limit")

    theta_grid = np.linspace(0.0, np.pi, int(branch_samples))
    spatial_frequency = np.asarray(
        fourth_order_semidiscrete_omega(
            theta_grid / dy,
            advection,
            dispersion,
            dy,
            d1_order=d1_order,
        )
    )
    differences = np.diff(spatial_frequency)
    turning_candidates = np.flatnonzero(
        (differences[:-1] > 0.0) & (differences[1:] <= 0.0)
    )
    if turning_candidates.size:
        turning_index = int(turning_candidates[0] + 1)
    else:
        turning_index = int(np.argmax(spatial_frequency))
    if turning_index <= 0:
        raise FloatingPointError("no increasing long-wave branch was found")

    upper_theta = float(theta_grid[turning_index])
    branch_maximum = float(spatial_frequency[turning_index])
    targets = (2.0 / dt) * np.tan(0.5 * omega * dt)
    if np.any(targets >= branch_maximum):
        first = int(np.flatnonzero(targets >= branch_maximum)[0])
        raise ValueError(
            "forcing frequency lies beyond the discrete long-wave branch at "
            f"index {first}: target={targets[first]:.6g}, max={branch_maximum:.6g}"
        )

    def symbol(theta: float) -> float:
        return float(
            fourth_order_semidiscrete_omega(
                theta / dy,
                advection,
                dispersion,
                dy,
                d1_order=d1_order,
            )
        )

    increments = np.array(
        [
            brentq(
                lambda theta, target=target: symbol(theta) - target,
                0.0,
                upper_theta,
                xtol=5.0e-15,
                rtol=4.0 * np.finfo(float).eps,
            )
            for target in targets
        ]
    )
    reconstructed = np.asarray(
        fourth_order_semidiscrete_omega(
            increments / dy,
            advection,
            dispersion,
            dy,
            d1_order=d1_order,
        )
    )
    residual = reconstructed - targets
    group_velocity = np.asarray(
        fourth_order_cn_group_velocity(
            increments / dy,
            advection,
            dispersion,
            dy,
            dt,
            d1_order=d1_order,
        )
    )
    if np.any(group_velocity <= 0.0):
        raise FloatingPointError("a selected phase increment has non-positive group velocity")
    metadata = {
        "branch_turning_phase_rad": upper_theta,
        "branch_maximum_semidiscrete_frequency": branch_maximum,
        "maximum_target_to_branch_ratio": float(np.max(targets) / branch_maximum),
        "maximum_absolute_dispersion_residual": float(np.max(np.abs(residual))),
        "maximum_relative_dispersion_residual": float(
            np.max(np.abs(residual) / np.maximum(np.abs(targets), 1.0))
        ),
        "minimum_CN_group_velocity": float(np.min(group_velocity)),
        "maximum_CN_group_velocity": float(np.max(group_velocity)),
    }
    return increments, metadata


def continuous_long_branch_phase_increments(
    angular_frequency: np.ndarray | float,
    advection: float,
    dispersion: float,
    dy: float,
) -> np.ndarray:
    """Return ``k*dy`` on the continuous KdV long-wave branch."""

    omega = _as_positive_frequency_array(angular_frequency)
    if advection <= 0.0 or dispersion <= 0.0 or dy <= 0.0:
        raise ValueError("advection, dispersion and dy must be positive")
    critical = np.sqrt(advection / (3.0 * dispersion))
    maximum = advection * critical - dispersion * critical**3
    if np.any(omega >= maximum):
        raise ValueError("forcing frequency lies beyond the continuous KdV branch")
    wavenumber = np.array(
        [
            brentq(
                lambda k, target=target: advection * k - dispersion * k**3 - target,
                0.0,
                critical,
            )
            for target in omega
        ]
    )
    return wavenumber * dy


@dataclass(frozen=True)
class PeriodicBoundarySpectrum:
    """Positive-frequency complex amplitudes extracted from one boundary record."""

    angular_frequency: np.ndarray
    complex_amplitude: np.ndarray
    period: float
    sample_dt: float
    removed_mean: float
    reconstruction_relative_l2: float
    retained_variance_fraction: float

    def __post_init__(self) -> None:
        omega = np.asarray(self.angular_frequency, dtype=float)
        amplitude = np.asarray(self.complex_amplitude, dtype=complex)
        if omega.ndim != 1 or omega.size == 0:
            raise ValueError("angular_frequency must be a non-empty vector")
        if amplitude.shape != omega.shape:
            raise ValueError("complex_amplitude must match angular_frequency")
        if not np.all(np.isfinite(omega)) or np.any(omega <= 0.0):
            raise ValueError("angular frequencies must be finite and positive")
        if np.any(np.diff(omega) <= 0.0):
            raise ValueError("angular frequencies must be strictly increasing")
        if not np.all(np.isfinite(amplitude)):
            raise ValueError("complex amplitudes must be finite")
        if not np.isfinite(self.period) or self.period <= 0.0:
            raise ValueError("period must be finite and positive")
        if not np.isfinite(self.sample_dt) or self.sample_dt <= 0.0:
            raise ValueError("sample_dt must be finite and positive")
        diagnostics = (
            self.removed_mean,
            self.reconstruction_relative_l2,
            self.retained_variance_fraction,
        )
        if not np.all(np.isfinite(diagnostics)):
            raise ValueError("spectrum diagnostics must be finite")
        object.__setattr__(self, "angular_frequency", omega)
        object.__setattr__(self, "complex_amplitude", amplitude)

    @classmethod
    def from_samples(
        cls,
        samples: np.ndarray,
        sample_dt: float,
        minimum_angular_frequency: float,
        maximum_angular_frequency: float,
    ) -> "PeriodicBoundarySpectrum":
        """Extract a band-limited periodic spectrum from endpoint-excluded samples."""

        values = np.asarray(samples, dtype=float)
        if values.ndim != 1 or values.size < 16:
            raise ValueError("samples must be a one-dimensional periodic record")
        if not np.all(np.isfinite(values)):
            raise ValueError("samples contain non-finite values")
        if sample_dt <= 0.0:
            raise ValueError("sample_dt must be positive")
        if not 0.0 < minimum_angular_frequency < maximum_angular_frequency:
            raise ValueError("invalid retained angular-frequency band")

        removed_mean = float(np.mean(values))
        centred = values - removed_mean
        transform = np.fft.rfft(centred) / values.size
        omega = 2.0 * np.pi * np.fft.rfftfreq(values.size, d=sample_dt)
        bin_width = 2.0 * np.pi / (values.size * sample_dt)
        selected = (
            (omega >= minimum_angular_frequency - 0.25 * bin_width)
            & (omega <= maximum_angular_frequency + 0.25 * bin_width)
            & (omega > 0.0)
        )
        if not np.any(selected):
            raise ValueError("no Fourier bins lie in the retained band")
        selected_indices = np.flatnonzero(selected)
        if values.size % 2 == 0 and selected_indices[-1] == values.size // 2:
            raise ValueError("the retained band must not include the Nyquist bin")
        selected_omega = omega[selected]
        complex_amplitude = 2.0 * transform[selected]

        sample_times = sample_dt * np.arange(values.size)
        reconstructed = np.sum(
            np.real(
                complex_amplitude[None, :]
                * np.exp(1.0j * sample_times[:, None] * selected_omega[None, :])
            ),
            axis=1,
        )
        denominator = max(float(np.linalg.norm(centred)), np.finfo(float).tiny)
        relative_error = float(np.linalg.norm(reconstructed - centred) / denominator)
        retained_fraction = float(
            1.0
            - np.linalg.norm(reconstructed - centred) ** 2 / denominator**2
        )
        return cls(
            angular_frequency=selected_omega,
            complex_amplitude=complex_amplitude,
            period=float(values.size * sample_dt),
            sample_dt=float(sample_dt),
            removed_mean=removed_mean,
            reconstruction_relative_l2=relative_error,
            retained_variance_fraction=retained_fraction,
        )

    def stationary_values(self, time: np.ndarray | float) -> np.ndarray | float:
        """Evaluate the retained stationary boundary modes at arbitrary times."""

        values = np.asarray(time, dtype=float)
        output = np.sum(
            np.real(
                self.complex_amplitude
                * np.exp(1.0j * values[..., None] * self.angular_frequency)
            ),
            axis=-1,
        )
        return float(output) if output.ndim == 0 else output


@dataclass(frozen=True)
class ModalThreeTraceLifting:
    """Three causal modal traces derived from one periodic boundary spectrum."""

    spectrum: PeriodicBoundarySpectrum
    phase_increment: np.ndarray
    ramp_delay_per_cell: np.ndarray
    ramp_duration: float
    method: str

    def __post_init__(self) -> None:
        increments = np.asarray(self.phase_increment, dtype=float)
        if increments.shape != self.spectrum.angular_frequency.shape:
            raise ValueError("phase increments must match the retained spectrum")
        if not np.all(np.isfinite(increments)) or np.any(increments < 0.0):
            raise ValueError("phase increments must be finite and non-negative")
        if self.ramp_duration <= 0.0:
            raise ValueError("ramp_duration must be positive")
        delays = np.asarray(self.ramp_delay_per_cell, dtype=float)
        if delays.shape != self.spectrum.angular_frequency.shape:
            raise ValueError("ramp delays must match the retained spectrum")
        if not np.all(np.isfinite(delays)) or np.any(delays < 0.0):
            raise ValueError("ramp delays must be finite and non-negative")
        object.__setattr__(self, "phase_increment", increments)
        object.__setattr__(self, "ramp_delay_per_cell", delays)

    @property
    def phase_delay(self) -> np.ndarray:
        return self.phase_increment / self.spectrum.angular_frequency

    def stationary_trace(
        self,
        time: np.ndarray | float,
        offset: int,
    ) -> np.ndarray | float:
        """Evaluate the stationary q-th trace without the start-up ramp."""

        if offset not in (0, 1, 2):
            raise ValueError("offset must be 0, 1 or 2")
        values = np.asarray(time, dtype=float)
        phase = (
            values[..., None] * self.spectrum.angular_frequency
            - float(offset) * self.phase_increment
        )
        output = np.sum(
            np.real(self.spectrum.complex_amplitude * np.exp(1.0j * phase)),
            axis=-1,
        )
        return float(output) if output.ndim == 0 else output

    def trace(
        self,
        time: np.ndarray | float,
        offset: int,
    ) -> np.ndarray | float:
        """Evaluate the q-th trace with a mode-wise causal ramp delay."""

        if offset not in (0, 1, 2):
            raise ValueError("offset must be 0, 1 or 2")
        values = np.asarray(time, dtype=float)
        delayed_time = (
            values[..., None] - float(offset) * self.ramp_delay_per_cell
        )
        ramp = np.asarray(half_cosine_ramp(delayed_time, self.ramp_duration))
        phase = (
            values[..., None] * self.spectrum.angular_frequency
            - float(offset) * self.phase_increment
        )
        output = np.sum(
            ramp
            * np.real(self.spectrum.complex_amplitude * np.exp(1.0j * phase)),
            axis=-1,
        )
        return float(output) if output.ndim == 0 else output

    def callables(
        self,
        boundary_signal: Callable[[float], float] | None = None,
    ) -> tuple[Callable[[float], float], ...]:
        """Return ``(g0,g1,g2)`` callables accepted by the high-order solver.

        When ``boundary_signal`` is provided, it is retained verbatim as
        ``g0``; only ``g1`` and ``g2`` are reconstructed from the extracted
        spectrum. This avoids silently altering the supplied physical record.
        """

        first = (
            (lambda time: float(self.trace(time, 0)))
            if boundary_signal is None
            else (lambda time: float(boundary_signal(time)))
        )
        return (
            first,
            lambda time: float(self.trace(time, 1)),
            lambda time: float(self.trace(time, 2)),
        )

    def stationary_field(
        self,
        times: np.ndarray,
        coordinates: np.ndarray,
        dy: float,
    ) -> np.ndarray:
        """Return the exact stationary discrete travelling-wave field."""

        sample_times = np.atleast_1d(np.asarray(times, dtype=float))
        y = np.atleast_1d(np.asarray(coordinates, dtype=float))
        if sample_times.ndim != 1 or y.ndim != 1 or dy <= 0.0:
            raise ValueError("times and coordinates must be one-dimensional; dy>0")
        wavenumber = self.phase_increment / dy
        output = np.empty((sample_times.size, y.size))
        spatial_phase = y[:, None] * wavenumber[None, :]
        spatial_factor = np.exp(-1.0j * spatial_phase)
        for index, time in enumerate(sample_times):
            temporal = (
                self.spectrum.complex_amplitude
                * np.exp(1.0j * self.spectrum.angular_frequency * time)
            )
            output[index] = np.sum(
                np.real(temporal[None, :] * spatial_factor),
                axis=1,
            )
        return output


def build_modal_three_trace_lifting(
    spectrum: PeriodicBoundarySpectrum,
    method: str,
    advection: float,
    dispersion: float,
    dy: float,
    dt: float,
    ramp_duration: float,
    *,
    d1_order: int = 6,
    ramp_delay_mode: str = "group",
) -> tuple[ModalThreeTraceLifting, dict[str, float | str]]:
    """Build duplicated, shallow, continuous or fully discrete trace phases."""

    omega = spectrum.angular_frequency
    metadata: dict[str, float | str] = {"method": method}
    if method == "duplicated":
        phase_increment = np.zeros_like(omega)
        group_delay = np.zeros_like(omega)
    elif method == "shallow":
        phase_increment = omega * dy / advection
        group_delay = np.full_like(omega, dy / advection)
    elif method == "continuous_kdv":
        phase_increment = continuous_long_branch_phase_increments(
            omega, advection, dispersion, dy
        )
        continuous_k = phase_increment / dy
        continuous_group_velocity = advection - 3.0 * dispersion * continuous_k**2
        if np.any(continuous_group_velocity <= 0.0):
            raise FloatingPointError("continuous KdV branch has non-positive group velocity")
        group_delay = dy / continuous_group_velocity
    elif method == "discrete_c6c4":
        phase_increment, branch_metadata = discrete_long_branch_phase_increments(
            omega,
            advection,
            dispersion,
            dy,
            dt,
            d1_order=d1_order,
        )
        metadata.update(branch_metadata)
        numerical_group_velocity = np.asarray(
            fourth_order_cn_group_velocity(
                phase_increment / dy,
                advection,
                dispersion,
                dy,
                dt,
                d1_order=d1_order,
            )
        )
        group_delay = dy / numerical_group_velocity
    else:
        raise ValueError(f"unknown lifting method: {method}")
    if ramp_delay_mode == "group":
        ramp_delay = group_delay
    elif ramp_delay_mode == "phase":
        ramp_delay = phase_increment / omega
    else:
        raise ValueError("ramp_delay_mode must be 'group' or 'phase'")
    metadata.update(
        {
            "ramp_delay_mode": ramp_delay_mode,
            "minimum_phase_increment_rad": float(np.min(phase_increment)),
            "maximum_phase_increment_rad": float(np.max(phase_increment)),
            "maximum_two_node_phase_shift_rad": float(
                2.0 * np.max(phase_increment)
            ),
            "minimum_ramp_delay_per_cell_dimensionless": float(
                np.min(ramp_delay)
            ),
            "maximum_ramp_delay_per_cell_dimensionless": float(
                np.max(ramp_delay)
            ),
            "maximum_group_minus_phase_delay_per_cell_dimensionless": float(
                np.max(np.abs(group_delay - phase_increment / omega))
            ),
        }
    )
    return (
        ModalThreeTraceLifting(
            spectrum=spectrum,
            phase_increment=phase_increment,
            ramp_delay_per_cell=ramp_delay,
            ramp_duration=ramp_duration,
            method=method,
        ),
        metadata,
    )
