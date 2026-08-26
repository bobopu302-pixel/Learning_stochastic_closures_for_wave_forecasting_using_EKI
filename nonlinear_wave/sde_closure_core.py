"""Stochastic extension of the frozen implicit-midpoint vKdV solver.

Origin: 3. KDV_nonlinear_case/sde_closure_core.py
Changes vs origin: comments/docstrings only (one Chinese design-note
reference translated; this provenance header added).

This module adds additive noise to the *unchanged* coarse-grid deterministic
solver chain (`CoastalHighOrderImplicitMidpointDABCSolver`) without editing
any frozen production file.  The stochastic step follows the conservative
Crank--Nicolson convention of Debussche & Printems (1999): one frozen
Gaussian increment per accepted time step enters the right-hand side of the
same bordered linear solve, so the increment is filtered by
``(I - dt/2 L)^{-1}`` exactly like the deterministic drift.  The increment is
frozen during fixed-point iterations and included in the true bordered
equation residual, so the convergence gates of the deterministic solver
remain meaningful.

Two noise constructions are provided (design-notes Section 4):

* ``ModalNoise`` -- scheme A, the linear-wave-case-style additive modal
  forcing sum_j sqrt(sigma_j) [cos(k_j y) dW_j^c + sin(k_j y) dW_j^s] with a
  fixed frequency grid mapped to wavenumbers through the long-wave vKdV
  dispersion branch at reference depth.  Amplitudes are parameterised by
  sqrt(sigma_j) exactly as in the linear case.
* ``GridWhiteNoise`` -- scheme B, the de Bouard--Debussche
  phi(x) d^2W/(dx dt) forcing discretised per grid cell as
  phi(y_j) sqrt(dt/dy) xi_{j,n} (spatially white at the coarse grid scale).

Both act on the *surface* variable and are mapped to the Green-normalised
state through S = d^{1/4}; both vanish on the three incident rows, on the
three DABC rows, and inside the numerical guard via a smooth physical-region
taper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from high_order_implicit_midpoint_candidate import (
    CoastalHighOrderImplicitMidpointDABCSolver,
)
from transparent_boundary_vkdv import march_convolution_system


def smooth_physical_taper(
    y_m: np.ndarray,
    *,
    start_m: float = 40.0,
    ramp_m: float = 200.0,
    end_m: float = 4000.0,
) -> np.ndarray:
    """Return a C1 half-cosine window supported on the physical region.

    Zero before ``start_m`` (covering the incident rows), rises to one over
    ``ramp_m``, falls back to zero over ``ramp_m`` before ``end_m``, and is
    exactly zero beyond ``end_m`` (the numerical guard and DABC rows).
    """

    values = np.asarray(y_m, dtype=float)
    if end_m - start_m <= 2.0 * ramp_m:
        raise ValueError("taper support is too short for the requested ramps")
    taper = np.zeros_like(values)
    rising = (values >= start_m) & (values < start_m + ramp_m)
    taper[rising] = 0.5 * (
        1.0 - np.cos(np.pi * (values[rising] - start_m) / ramp_m)
    )
    flat = (values >= start_m + ramp_m) & (values <= end_m - ramp_m)
    taper[flat] = 1.0
    falling = (values > end_m - ramp_m) & (values <= end_m)
    taper[falling] = 0.5 * (
        1.0 - np.cos(np.pi * (end_m - values[falling]) / ramp_m)
    )
    return taper


def long_branch_wavenumber(
    angular_frequency_nd: float,
    mu: float,
) -> float:
    """Solve omega = k - (mu/6) k^3 on the long-wave propagating branch.

    The branch maximum sits at k* = sqrt(2/mu); requested frequencies must
    stay below the branch peak omega(k*) = (2/3) k* like the incident
    lifting does.
    """

    omega = float(angular_frequency_nd)
    if omega <= 0.0:
        raise ValueError("angular frequency must be positive")
    k_star = np.sqrt(2.0 / mu)
    omega_max = k_star - (mu / 6.0) * k_star**3
    if omega >= omega_max:
        raise ValueError(
            f"frequency {omega:.4f} exceeds the long-wave branch peak "
            f"{omega_max:.4f}"
        )
    roots = np.roots([-(mu / 6.0), 0.0, 1.0, -omega])
    physical = [
        float(np.real(root))
        for root in roots
        if abs(np.imag(root)) < 1.0e-12
        and 0.0 < np.real(root) < k_star
    ]
    if not physical:
        raise RuntimeError("no propagating-branch root found")
    return min(physical)


def terrain_weight(
    depth_ratio: np.ndarray,
    exponent: float,
) -> np.ndarray:
    """Bathymetry-only noise envelope w = (d_min/d)^q, unit maximum.

    Uses no fine-run information: the premise is that coarse-model error
    production grows where the local wavelength shortens (points per
    wavelength ~ sqrt(h)/dx), so shallower water gets more noise.  q = 0 is
    uniform; larger q concentrates the envelope on the shallow shelf.  The
    exponent is an EKI-learnable structural parameter.
    """

    d = np.asarray(depth_ratio, dtype=float)
    if np.any(d <= 0.0):
        raise ValueError("depth ratio must be positive")
    if exponent < 0.0:
        raise ValueError("terrain exponent must be non-negative")
    weight = (float(np.min(d)) / d) ** float(exponent)
    return weight / float(np.max(weight))


def _moving_average(values: np.ndarray, window_points: int) -> np.ndarray:
    if window_points <= 1:
        return np.asarray(values, dtype=float)
    kernel = np.ones(int(window_points)) / float(window_points)
    padded = np.pad(
        np.asarray(values, dtype=float),
        (window_points // 2, window_points - 1 - window_points // 2),
        mode="edge",
    )
    return np.convolve(padded, kernel, mode="valid")


def error_production_weight(
    y_grid_m: np.ndarray,
    depth_grid_m: np.ndarray,
    y_error_m: np.ndarray,
    error_std_m: np.ndarray,
    *,
    gravity_m_s2: float = 9.81,
    smoothing_window_m: float = 150.0,
) -> np.ndarray:
    """Data-driven noise envelope from the measured coarse-model error.

    For one-way transport at speed c the stationary error variance obeys
    d(sigma_err^2)/dy ~ q(y)/c(y), so the local error-production rate is
    q(y) = c(y) d(sigma_err^2)/dy.  The returned weight is
    sqrt(max(q, 0)) normalised to unit maximum: noise is injected where the
    coarse model actually generates error (the under-resolved slope), not
    uniformly.  Negative production (error decay) maps to zero weight.
    """

    y_grid = np.asarray(y_grid_m, dtype=float)
    variance_source = np.asarray(error_std_m, dtype=float) ** 2
    variance = np.interp(y_grid, np.asarray(y_error_m, dtype=float),
                         variance_source)
    spacing = float(np.mean(np.diff(y_grid)))
    window_points = max(1, int(round(smoothing_window_m / spacing)))
    variance = _moving_average(variance, window_points)
    celerity = np.sqrt(gravity_m_s2 * np.asarray(depth_grid_m, dtype=float))
    production = np.gradient(variance, y_grid) * celerity
    weight = np.sqrt(np.clip(production, 0.0, None))
    weight = _moving_average(weight, window_points)
    peak = float(np.max(weight))
    if peak <= 0.0:
        raise ValueError("error production weight vanished everywhere")
    return weight / peak


@dataclass(frozen=True)
class ModalNoiseParameters:
    """Scheme-A modal noise definition (linear-wave-case convention).

    ``correlation_time_nd`` adds temporal colour: each Wiener dimension is
    replaced by a unit-variance OU state with the given dimensionless
    correlation time, refreshed as z <- rho z + sqrt(1-rho^2) xi with
    rho = exp(-dt/tau).  The per-step marginal increment variance is kept
    identical to the white case, so tau = 0 degenerates bitwise to the
    original scheme and the amplitude convention is unchanged; temporal
    accumulation on resonant modes grows with tau and is recalibrated by
    the pilot/EKI.
    """

    frequencies_hz: tuple[float, ...]
    sqrt_sigma: tuple[float, ...]
    correlation_time_nd: float = 0.0
    # Variance-preserving colour convention: normalise each mode pair's OU
    # forcing so its spectral density AT THAT MODE'S OWN FREQUENCY equals
    # the white value for every tau.  This decouples tau from the injected
    # resonant power (the fixed per-step-variance convention couples them:
    # larger tau then amplifies resonant power by ~2 tau/dt, which is why a
    # calibration can reject tau for power reasons alone).
    variance_preserving: bool = False
    taper_start_m: float = 40.0
    taper_ramp_m: float = 200.0
    taper_end_m: float = 4000.0


class ModalNoise:
    """Additive modal Q-Wiener increments in the normalised state.

    Column pairs are S(y) * taper(y) * sqrt(sigma_j) * {cos(k_j y), sin(k_j y)}
    so each retained mode injects pointwise surface variance sigma_j dt on the
    taper plateau (cos^2 + sin^2 = 1).  sqrt(sigma_j) is dimensionless
    (surface units u = eta/a0 per sqrt of dimensionless time).
    """

    kind = "modal_scheme_A"

    def __init__(
        self,
        parameters: ModalNoiseParameters,
        y_nd: np.ndarray,
        lambda_ref_m: float,
        time_ref_s: float,
        mu: float,
        surface_to_green: np.ndarray,
        dt_nd: float,
        rng: np.random.Generator,
        spatial_weight: np.ndarray | None = None,
    ) -> None:
        frequencies = np.asarray(parameters.frequencies_hz, dtype=float)
        sqrt_sigma = np.asarray(parameters.sqrt_sigma, dtype=float)
        if frequencies.size != sqrt_sigma.size:
            raise ValueError("frequencies and sqrt_sigma must align")
        if np.any(sqrt_sigma < 0.0):
            raise ValueError("sqrt_sigma must be non-negative")
        self.parameters = parameters
        self.rng = rng
        self.dt_nd = float(dt_nd)
        y_m = np.asarray(y_nd, dtype=float) * lambda_ref_m
        taper = smooth_physical_taper(
            y_m,
            start_m=parameters.taper_start_m,
            ramp_m=parameters.taper_ramp_m,
            end_m=parameters.taper_end_m,
        )
        if spatial_weight is not None:
            weight = np.asarray(spatial_weight, dtype=float)
            if weight.shape != taper.shape:
                raise ValueError("spatial_weight must match the grid")
            if np.any(weight < 0.0):
                raise ValueError("spatial_weight must be non-negative")
            taper = taper * weight
        scale = np.asarray(surface_to_green, dtype=float) * taper
        self.wavenumbers_nd = np.asarray(
            [
                long_branch_wavenumber(
                    2.0 * np.pi * frequency * time_ref_s, mu
                )
                for frequency in frequencies
            ]
        )
        columns = []
        for k_nd, amplitude in zip(self.wavenumbers_nd, sqrt_sigma):
            phase = k_nd * np.asarray(y_nd, dtype=float)
            columns.append(scale * amplitude * np.cos(phase))
            columns.append(scale * amplitude * np.sin(phase))
        self.basis = np.column_stack(columns)
        self.taper = taper
        tau = float(parameters.correlation_time_nd)
        if tau < 0.0:
            raise ValueError("correlation_time_nd must be >= 0")
        if tau > 0.0:
            self.rho = float(np.exp(-self.dt_nd / tau))
            self.state = self.rng.standard_normal(self.basis.shape[1])
            if parameters.variance_preserving:
                omega_nd = (
                    2.0
                    * np.pi
                    * np.asarray(frequencies, dtype=float)
                    * time_ref_s
                )
                lorentzian = (
                    1.0
                    - 2.0 * self.rho * np.cos(omega_nd * self.dt_nd)
                    + self.rho**2
                ) / (1.0 - self.rho**2)
                per_mode = np.sqrt(lorentzian)
                self.vp_scale = np.repeat(per_mode, 2)
            else:
                self.vp_scale = np.ones(self.basis.shape[1])
        else:
            self.rho = 0.0
            self.state = None
            self.vp_scale = np.ones(self.basis.shape[1])

    def __call__(self, step: int) -> np.ndarray:
        if self.state is None:
            xi = self.rng.standard_normal(self.basis.shape[1])
            return self.basis @ xi * np.sqrt(self.dt_nd)
        increment = self.basis @ (self.state * self.vp_scale) * np.sqrt(
            self.dt_nd
        )
        xi = self.rng.standard_normal(self.basis.shape[1])
        self.state = self.rho * self.state + np.sqrt(
            1.0 - self.rho**2
        ) * xi
        return increment

    def metadata(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "frequencies_hz": list(self.parameters.frequencies_hz),
            "sqrt_sigma_nd": list(self.parameters.sqrt_sigma),
            "wavenumbers_nd": self.wavenumbers_nd.tolist(),
            "correlation_time_nd": self.parameters.correlation_time_nd,
            "taper_start_m": self.parameters.taper_start_m,
            "taper_ramp_m": self.parameters.taper_ramp_m,
            "taper_end_m": self.parameters.taper_end_m,
            "wiener_dimension": int(self.basis.shape[1]),
        }


@dataclass(frozen=True)
class GridWhiteNoiseParameters:
    """Scheme-B phi(x) d^2W/(dx dt) noise definition.

    ``correlation_length_cells`` is the Gaussian spatial-correlation length
    of the forcing in grid cells.  Zero gives the mathematically rough pure
    space-time white noise; de Bouard--Debussche well-posedness requires a
    Hilbert--Schmidt (spatially smoothed) covariance, and on the
    non-dissipative implicit-midpoint scheme the pure white case pumps
    non-radiating grid-scale wavenumbers and blows up (documented negative
    control), so a small positive correlation length is the usable form.
    """

    phi_amplitude: float
    correlation_length_cells: float = 3.0
    # Temporal OU colour; 0 = white in time (original scheme).  Same
    # per-step marginal-variance convention as ModalNoiseParameters.
    correlation_time_nd: float = 0.0
    # Variance-preserving colour convention: normalise the OU forcing so
    # its spectral density at the reference band-centre frequency equals
    # the white value for every tau (see ModalNoiseParameters).  The
    # reference is dimensionless: 2*pi*0.0675 Hz*t0 = 5.244.
    variance_preserving: bool = False
    variance_reference_omega_nd: float = 5.244
    taper_start_m: float = 40.0
    taper_ramp_m: float = 200.0
    taper_end_m: float = 4000.0


class GridWhiteNoise:
    """de Bouard--Debussche forcing, white at the coarse grid scale.

    The dimensionless surface increment per step and cell is
    phi(y_j) sqrt(dt/dy) xi_{j,n}; the envelope phi = phi_amplitude * taper.
    The 1/sqrt(dy) factor is the standard cylindrical-Wiener projection onto
    grid indicator functions (Debussche & Printems 1999), which makes the
    injected energy density grid-independent while the pointwise variance
    grows as the grid refines -- the coarse grid provides the spatial cutoff.
    """

    kind = "grid_white_scheme_B"

    def __init__(
        self,
        parameters: GridWhiteNoiseParameters,
        y_nd: np.ndarray,
        lambda_ref_m: float,
        surface_to_green: np.ndarray,
        dt_nd: float,
        rng: np.random.Generator,
        spatial_weight: np.ndarray | None = None,
    ) -> None:
        if parameters.phi_amplitude < 0.0:
            raise ValueError("phi_amplitude must be non-negative")
        self.parameters = parameters
        self.rng = rng
        self.dt_nd = float(dt_nd)
        y_values = np.asarray(y_nd, dtype=float)
        self.dy_nd = float(y_values[1] - y_values[0])
        y_m = y_values * lambda_ref_m
        taper = smooth_physical_taper(
            y_m,
            start_m=parameters.taper_start_m,
            ramp_m=parameters.taper_ramp_m,
            end_m=parameters.taper_end_m,
        )
        if spatial_weight is not None:
            weight = np.asarray(spatial_weight, dtype=float)
            if weight.shape != taper.shape:
                raise ValueError("spatial_weight must match the grid")
            if np.any(weight < 0.0):
                raise ValueError("spatial_weight must be non-negative")
            taper = taper * weight
        self.envelope = (
            np.asarray(surface_to_green, dtype=float)
            * parameters.phi_amplitude
            * taper
        )
        self.taper = taper
        length = float(parameters.correlation_length_cells)
        if length < 0.0:
            raise ValueError("correlation_length_cells must be >= 0")
        if length == 0.0:
            self.kernel = None
        else:
            half_width = max(1, int(np.ceil(4.0 * length)))
            offsets = np.arange(-half_width, half_width + 1, dtype=float)
            kernel = np.exp(-0.5 * (offsets / length) ** 2)
            # Normalise so the smoothed field keeps unit pointwise variance;
            # phi_amplitude therefore has the same meaning for every
            # correlation length while the injected spectrum narrows.
            self.kernel = kernel / np.sqrt(np.sum(kernel**2))
        tau = float(parameters.correlation_time_nd)
        if tau < 0.0:
            raise ValueError("correlation_time_nd must be >= 0")
        if tau > 0.0:
            self.rho = float(np.exp(-self.dt_nd / tau))
            self.state = self._innovation()
            if parameters.variance_preserving:
                omega = float(parameters.variance_reference_omega_nd)
                self.vp_scale = float(
                    np.sqrt(
                        (
                            1.0
                            - 2.0 * self.rho * np.cos(omega * self.dt_nd)
                            + self.rho**2
                        )
                        / (1.0 - self.rho**2)
                    )
                )
            else:
                self.vp_scale = 1.0
        else:
            self.rho = 0.0
            self.state = None
            self.vp_scale = 1.0

    def _innovation(self) -> np.ndarray:
        xi = self.rng.standard_normal(self.envelope.size)
        if self.kernel is not None:
            xi = np.convolve(xi, self.kernel, mode="same")
        return xi

    def __call__(self, step: int) -> np.ndarray:
        if self.state is None:
            xi = self._innovation()
            return self.envelope * xi * np.sqrt(self.dt_nd / self.dy_nd)
        increment = (
            self.envelope
            * self.state
            * (self.vp_scale * np.sqrt(self.dt_nd / self.dy_nd))
        )
        self.state = self.rho * self.state + np.sqrt(
            1.0 - self.rho**2
        ) * self._innovation()
        return increment

    def metadata(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "phi_amplitude_nd": self.parameters.phi_amplitude,
            "correlation_length_cells": (
                self.parameters.correlation_length_cells
            ),
            "correlation_time_nd": self.parameters.correlation_time_nd,
            "dy_nd": self.dy_nd,
            "taper_start_m": self.parameters.taper_start_m,
            "taper_ramp_m": self.parameters.taper_ramp_m,
            "taper_end_m": self.parameters.taper_end_m,
            "wiener_dimension": int(self.envelope.size),
            "pointwise_variance_convention": (
                "unit-variance smoothed field; increment = envelope * "
                "smoothed_xi * sqrt(dt/dy)"
            ),
        }


class StochasticImplicitMidpointDABCSolver(
    CoastalHighOrderImplicitMidpointDABCSolver
):
    """Implicit-midpoint marcher with an additive per-step noise increment.

    ``run_stochastic`` reproduces the parent ``run`` step for step; the only
    change is a frozen additive increment in the bordered right-hand side and
    in the true equation residual.  With ``noise_increment=None`` (or a
    generator returning zeros) the trajectory is bitwise identical to the
    deterministic parent.

    ``damping_factor`` implements Rayleigh damping -lambda*v by first-order
    splitting: after the implicit-midpoint step is accepted, the state is
    multiplied by exp(-lambda*dt*mask(y)).  The mask must be zero on the
    three incident rows and throughout the guard/DABC region, so the
    convolution histories and boundary constraints (which only read those
    rows) remain exactly consistent with the undamped discrete recurrences.
    """

    def run_stochastic(
        self,
        initial_surface: np.ndarray,
        output_stride: int,
        boundary_traces: tuple[
            Callable[[float], float],
            Callable[[float], float],
            Callable[[float], float],
        ]
        | None = None,
        *,
        noise_increment: Callable[[int], np.ndarray] | None = None,
        damping_factor: np.ndarray | None = None,
        initial_outflow_relative_tolerance: float = 1.0e-10,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if output_stride < 1:
            raise ValueError("output_stride must be positive")
        if boundary_traces is None:
            boundary_traces = (
                lambda _time: 0.0,
                lambda _time: 0.0,
                lambda _time: 0.0,
            )
        if len(boundary_traces) != 3:
            raise ValueError("three incident surface traces are required")
        if damping_factor is not None:
            damping_factor = np.asarray(damping_factor, dtype=float)
            if damping_factor.shape != self.y.shape:
                raise ValueError("damping_factor must match the grid")
            if np.any(damping_factor <= 0.0) or np.any(damping_factor > 1.0):
                raise ValueError("damping_factor entries must lie in (0, 1]")
            boundary_rows_ok = (
                np.allclose(damping_factor[:3], 1.0)
                and np.allclose(damping_factor[-6:], 1.0)
            )
            if not boundary_rows_ok:
                raise ValueError(
                    "damping_factor must be exactly one on the incident and "
                    "DABC rows"
                )

        surface_initial = np.asarray(initial_surface, dtype=float).copy()
        if surface_initial.shape != self.y.shape:
            raise ValueError("initial_surface must match y")
        if not np.all(np.isfinite(surface_initial)):
            raise ValueError("initial_surface contains non-finite values")
        field_scale = max(
            float(np.max(np.abs(surface_initial))), np.finfo(float).tiny
        )
        tail_ratio = float(np.max(np.abs(surface_initial[-6:]))) / field_scale
        if tail_ratio > initial_outflow_relative_tolerance:
            raise ValueError(
                "the homogeneous linear DABC requires zero/compatible "
                "exterior initial data"
            )

        initial_trace_values = tuple(
            float(trace(0.0)) for trace in boundary_traces
        )
        for row, trace_value in enumerate(initial_trace_values):
            surface_initial[row] = trace_value
        current = self.to_normalized(surface_initial)
        holder = [current]
        self.fixed_point_iteration_counts = []
        self.maximum_fixed_point_update = 0.0
        self.maximum_fixed_point_scaled_update = 0.0
        self.maximum_equation_residual = 0.0
        self.maximum_scaled_equation_residual = 0.0

        times = [0.0]
        surface_outputs = [surface_initial.copy()]
        normalized_outputs = [current.copy()]
        initial_left_residuals = tuple(
            float(surface_initial[row] - trace_value)
            for row, trace_value in enumerate(initial_trace_values)
        )
        initial_right_residuals = tuple(
            float(constraint @ current) for constraint in self.constraints
        )
        residuals = [initial_left_residuals + initial_right_residuals]

        kernel_list: list[np.ndarray] = []
        kernel_sources: list[int] = []
        for shift in range(3):
            kernel_list.extend(
                (
                    self.kernels.root_sum,
                    self.kernels.root_pair_sum,
                    self.kernels.root_product,
                )
            )
            kernel_sources.extend((shift, shift + 1, shift + 2))

        def solve_step(step: int, histories: np.ndarray) -> np.ndarray:
            previous = holder[0]
            base_rhs = np.asarray(self.right_matrix @ previous).ravel()
            if noise_increment is not None:
                noise_vector = np.asarray(
                    noise_increment(step), dtype=float
                )
                if noise_vector.shape != previous.shape:
                    raise ValueError("noise increment must match the state")
                base_rhs = base_rhs + noise_vector
            time_value = step * self.dt
            trace_values = tuple(
                float(trace(time_value)) for trace in boundary_traces
            )
            boundary_rhs = np.empty(3)
            for shift in range(3):
                h1, h2, h3 = histories[3 * shift : 3 * shift + 3]
                boundary_rhs[shift] = h1 - h2 + h3

            def solve_with_nonlinearity(
                nonlinearity: np.ndarray,
            ) -> np.ndarray:
                rhs = base_rhs + self.dt * nonlinearity
                for row, trace_value in enumerate(trace_values):
                    rhs[row] = self.surface_to_green[row] * trace_value
                for shift in range(3):
                    rhs[self.n - 1 - shift] = boundary_rhs[shift]
                return self.lu.solve(rhs)

            def true_bordered_equation_residual(
                state: np.ndarray,
                nonlinearity: np.ndarray,
            ) -> tuple[float, float]:
                rhs = base_rhs + self.dt * nonlinearity
                lhs = np.asarray(self.left_matrix @ state).ravel()
                for row, trace_value in enumerate(trace_values):
                    rhs[row] = self.surface_to_green[row] * trace_value
                    lhs[row] = state[row]
                for shift, constraint in enumerate(self.constraints):
                    row = self.n - 1 - shift
                    rhs[row] = boundary_rhs[shift]
                    lhs[row] = float(constraint @ state)
                residual = float(np.max(np.abs(lhs - rhs)))
                scale = max(
                    float(np.max(np.abs(lhs))),
                    float(np.max(np.abs(rhs))),
                    np.finfo(float).tiny,
                )
                return residual, residual / scale

            nonlinear_previous = self.nonlinear(previous)
            iterate = solve_with_nonlinearity(nonlinear_previous)
            if not np.all(np.isfinite(iterate)):
                raise FloatingPointError(
                    f"non-finite stochastic predictor at step {step}"
                )
            converged = False
            final_update = float("inf")
            final_scaled_update = float("inf")
            final_equation_residual = float("inf")
            final_scaled_equation_residual = float("inf")
            final_nonlinearity = nonlinear_previous
            for iteration_count in range(
                1, self.fixed_point_maximum_iterations + 1
            ):
                midpoint = 0.5 * (previous + iterate)
                midpoint_nonlinearity = self.nonlinear(midpoint)
                updated = solve_with_nonlinearity(midpoint_nonlinearity)
                if not np.all(np.isfinite(updated)):
                    amplitude = float(
                        np.max(np.abs(self.to_surface(iterate)))
                    )
                    raise FloatingPointError(
                        "non-finite stochastic iterate at "
                        f"step {step}, iteration {iteration_count}; "
                        f"previous iterate max|u|={amplitude:.6g}"
                    )
                final_update = float(np.max(np.abs(updated - iterate)))
                scale = max(float(np.max(np.abs(updated))), 1.0)
                final_scaled_update = final_update / scale
                update_threshold = (
                    self.fixed_point_absolute_tolerance
                    + self.fixed_point_relative_tolerance * scale
                )
                true_midpoint_nonlinearity = self.nonlinear(
                    0.5 * (previous + updated)
                )
                (
                    final_equation_residual,
                    final_scaled_equation_residual,
                ) = true_bordered_equation_residual(
                    updated, true_midpoint_nonlinearity
                )
                iterate = updated
                if (
                    final_update <= update_threshold
                    and final_equation_residual
                    <= self.equation_absolute_tolerance
                    and final_scaled_equation_residual
                    <= self.equation_relative_tolerance
                ):
                    converged = True
                    final_nonlinearity = true_midpoint_nonlinearity
                    break
            if not converged:
                amplitude = float(np.max(np.abs(self.to_surface(iterate))))
                raise FloatingPointError(
                    "stochastic fixed point did not converge at "
                    f"step {step}; update={final_update:.3e}, "
                    f"equation residual={final_equation_residual:.3e}, "
                    f"max|u|={amplitude:.6g}"
                )

            self.fixed_point_iteration_counts.append(iteration_count)
            self.maximum_fixed_point_update = max(
                self.maximum_fixed_point_update, final_update
            )
            self.maximum_fixed_point_scaled_update = max(
                self.maximum_fixed_point_scaled_update, final_scaled_update
            )
            self.maximum_equation_residual = max(
                self.maximum_equation_residual, final_equation_residual
            )
            self.maximum_scaled_equation_residual = max(
                self.maximum_scaled_equation_residual,
                final_scaled_equation_residual,
            )
            new_state = iterate
            if damping_factor is not None:
                new_state = new_state * damping_factor
            new_surface = self.to_surface(new_state)
            holder[0] = new_state

            if step % output_stride == 0 or step == self.n_steps:
                times.append(time_value)
                surface_outputs.append(new_surface.copy())
                normalized_outputs.append(new_state.copy())
                left_residuals = tuple(
                    float(new_surface[row] - trace_value)
                    for row, trace_value in enumerate(trace_values)
                )
                right_residuals = tuple(
                    float(constraint @ new_state - boundary_rhs[index])
                    for index, constraint in enumerate(self.constraints)
                )
                residuals.append(left_residuals + right_residuals)
            return new_surface[-2:-7:-1].copy()

        march_convolution_system(
            self.n_steps,
            kernel_list,
            kernel_sources,
            surface_initial[-2:-7:-1].copy(),
            solve_step,
        )
        return (
            np.asarray(times),
            np.asarray(surface_outputs),
            np.asarray(normalized_outputs),
            np.asarray(residuals),
        )
