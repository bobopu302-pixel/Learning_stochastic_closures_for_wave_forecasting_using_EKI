"""Open-boundary solvers and verification tests for the coastal vKdV study.

Origin: 3. KDV_nonlinear_case/transparent_boundary_vkdv.py
Changes vs origin (numerics untouched):
* matplotlib imports and the figure blocks at the end of the four
  validation runners (run_constant_depth_validation,
  run_discrete_abc_validation, run_coastal_pulse_validation,
  run_coastal_truth) deleted (release ships no plotting); every
  metrics.json / .npy / .csv data save is preserved;
* the fine_profile collection in run_discrete_abc_validation deleted
  (it only fed the deleted figure).

The production solver assembled from :mod:`pde_core` is deliberately left
unchanged.  This module develops the open-boundary replacement in verifiable
stages.  The first stage solves the constant-coefficient linearised KdV

    u_t + U1*u_y + U2*u_yyy = 0,       U2 > 0,

on a finite interval.  A prescribed elevation is imposed at the inflow
``y=0`` and the two conditions at ``y=L`` are convolution-quadrature
discretisations of the continuous transparent boundary conditions

    u - L^{-1}(1/lambda**2) * u_yy = 0,
    u_y - L^{-1}(1/lambda) * u_yy = 0,

where ``lambda`` is the unique root with negative real part of

    U2*lambda**3 + U1*lambda + s = 0.

The boundary therefore represents the semi-infinite constant-depth exterior;
it does not set the wave to zero or add a damping layer.  The continuous
operator is retained as a reference test.  The production coastal calculation
uses the fully discrete artificial boundary conditions matched to the
centred Crank--Nicolson (C--CN) discretisation of Besse et al.; its nonlinear
term remains the original second-order directional-upwind discretisation.

References
----------
C. Besse, M. Ehrhardt and I. Lacroix-Violet (2016), ``Discrete
artificial boundary conditions for the linearized Korteweg--de Vries
equation``, Numer. Methods Partial Differential Equations 32, 145--172.

The convolution-quadrature implementation follows the operational-calculus
idea of C. Lubich.  It discretises the *continuous* KdV transparent operator;
it is not presented as the fully discrete Z-transform boundary of Besse et al.
That distinction is retained in filenames and diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.sparse import csc_matrix, diags, eye, lil_matrix
from scipy.sparse.linalg import splu

from pde_core import (
    CoastalParameters,
    IMEXBDF2Solver,
    computational_grid,
    cubic_coastal_depth,
    nonperiodic_derivative_matrices,
)
from sea_state_boundary import SeaStateParameters, make_exact_tma_boundary


@dataclass(frozen=True)
class LinearTBCParameters:
    """Parameters for the constant-depth transparent-boundary benchmark."""

    length: float = 25.0
    n_x: int = 501
    dt: float = 0.005
    final_time: float = 45.0
    advection: float = 1.0
    dispersion: float = 0.02
    amplitude: float = 0.1
    packet_centre: float = 7.0
    packet_width: float = 2.0
    carrier_wavenumber: float = 2.0
    output_stride: int = 20

    @property
    def n_steps(self) -> int:
        steps = int(round(self.final_time / self.dt))
        if not np.isclose(steps * self.dt, self.final_time):
            raise ValueError("final_time must be an integer multiple of dt")
        return steps

    @property
    def group_velocity(self) -> float:
        return self.advection - 3.0 * self.dispersion * self.carrier_wavenumber**2


def _negative_cubic_root(
    laplace_points: np.ndarray,
    advection: float,
    dispersion: float,
) -> np.ndarray:
    """Return the root with negative real part at each Laplace point.

    For Re(s)>0 and positive dispersion the root-separation theorem gives
    exactly one such root.  Explicitly checking all three roots is slower than
    Cardano's formula but avoids branch-switching errors in the CQ kernel.
    """

    if dispersion <= 0.0:
        raise ValueError("dispersion must be positive in the shoreward coordinate")
    if advection <= 0.0:
        raise ValueError("advection must be positive in the shoreward coordinate")
    points = np.asarray(laplace_points, dtype=complex)
    p = advection / dispersion
    q = points / dispersion
    discriminant = (0.5 * q) ** 2 + (p / 3.0) ** 3
    square_root = np.sqrt(discriminant)
    first = -0.5 * q + square_root
    second = -0.5 * q - square_root
    # Use the larger Cardano factor, then enforce u*v=-p/3.  This avoids
    # catastrophic cancellation and gives all three roots without branch
    # tracking around the CQ contour.
    factor = np.where(np.abs(first) >= np.abs(second), first, second)
    u = np.exp(np.log(factor) / 3.0)
    v = -p / (3.0 * u)
    omega = np.exp(2.0j * np.pi / 3.0)
    roots = np.stack(
        (u + v, omega * u + omega**2 * v, omega**2 * u + omega * v)
    )
    indices = np.argmin(np.real(roots), axis=0)
    result = np.take_along_axis(roots, indices[None, ...], axis=0)[0]
    residual = dispersion * result**3 + advection * result + points
    scale = np.abs(points) + np.abs(advection * result) + np.abs(dispersion * result**3)
    relative_residual = np.max(np.abs(residual) / np.maximum(scale, 1.0))
    if relative_residual > 5.0e-10:
        raise FloatingPointError(
            f"cubic root residual is too large ({relative_residual:.3e})"
        )
    return result


def bdf2_convolution_quadrature_weights(
    kernel: Callable[[np.ndarray], np.ndarray],
    dt: float,
    n_steps: int,
) -> np.ndarray:
    """Compute BDF2 convolution-quadrature weights by a Cauchy FFT.

    If ``K`` is the Laplace transform of a causal kernel, the weights are the
    Taylor coefficients of ``K(delta(zeta)/dt)`` with
    ``delta(zeta)=3/2-2*zeta+zeta**2/2``.  The FFT circle radius balances
    aliasing against roundoff amplification.
    """

    if dt <= 0.0 or n_steps < 1:
        raise ValueError("dt must be positive and n_steps must be at least one")
    transform_size = 1 << int(np.ceil(np.log2(2 * (n_steps + 1))))
    radius = np.exp(np.log(np.finfo(float).eps) / (2.0 * transform_size))
    angles = -2.0j * np.pi * np.arange(transform_size) / transform_size
    zeta = radius * np.exp(angles)
    delta = 1.5 - 2.0 * zeta + 0.5 * zeta**2
    samples = np.asarray(kernel(delta / dt), dtype=complex)
    scaled_weights = np.fft.ifft(samples)
    weights = scaled_weights[: n_steps + 1] / radius ** np.arange(n_steps + 1)
    imaginary_ratio = float(
        np.max(np.abs(np.imag(weights)))
        / max(np.max(np.abs(np.real(weights))), np.finfo(float).tiny)
    )
    if imaginary_ratio > 2.0e-7:
        raise FloatingPointError(
            f"CQ weights are not numerically real (relative imaginary part {imaginary_ratio:.3e})"
        )
    return np.real(weights)


@dataclass(frozen=True)
class TransparentKernels:
    """BDF2-CQ weights for the two right/outflow KdV conditions."""

    inverse_lambda: np.ndarray
    inverse_lambda_squared: np.ndarray

    @classmethod
    def build(
        cls,
        advection: float,
        dispersion: float,
        dt: float,
        n_steps: int,
    ) -> "TransparentKernels":
        def roots(points: np.ndarray) -> np.ndarray:
            return _negative_cubic_root(points, advection, dispersion)

        inverse_lambda = bdf2_convolution_quadrature_weights(
            lambda points: 1.0 / roots(points), dt, n_steps
        )
        inverse_lambda_squared = bdf2_convolution_quadrature_weights(
            lambda points: 1.0 / roots(points) ** 2, dt, n_steps
        )
        return cls(inverse_lambda, inverse_lambda_squared)


def march_two_cq_convolutions(
    n_steps: int,
    first_kernel: np.ndarray,
    second_kernel: np.ndarray,
    initial_trace: float,
    solve_step: Callable[[int, float, float], float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """March two causal histories exactly with a CDQ/FFT online convolution.

    ``solve_step(n, h1, h2)`` is called in increasing time order and returns
    the new boundary trace ``q[n]``.  The supplied histories are

        h_m[n] = sum_{j=1}^n kernel_m[j] * q[n-j].

    A divide-and-conquer convolution (CDQ) ensures every known block only
    contributes to later time levels.  This reduces the long-run cost from
    quadratic to roughly ``O(N log(N)^2)`` without approximating the kernels.
    """

    if len(first_kernel) < n_steps + 1 or len(second_kernel) < n_steps + 1:
        raise ValueError("CQ kernels are shorter than the requested march")
    traces = np.empty(n_steps + 1, dtype=float)
    traces[0] = float(initial_trace)
    first_history = np.zeros(n_steps + 1, dtype=float)
    second_history = np.zeros(n_steps + 1, dtype=float)

    def add_cross(left: int, middle: int, right: int) -> None:
        source = traces[left:middle]
        width = right - left
        first = first_kernel[:width]
        second = second_kernel[:width]
        if width <= 64:
            first_convolution = np.convolve(source, first)
            second_convolution = np.convolve(source, second)
        else:
            convolution_length = source.size + width - 1
            transform_size = next_fast_len(convolution_length)
            source_transform = rfft(source, transform_size)
            first_convolution = irfft(
                source_transform * rfft(first, transform_size), transform_size
            )[:convolution_length]
            second_convolution = irfft(
                source_transform * rfft(second, transform_size), transform_size
            )[:convolution_length]
        start = middle - left
        stop = right - left
        first_history[middle:right] += first_convolution[start:stop]
        second_history[middle:right] += second_convolution[start:stop]

    def recurse(left: int, right: int) -> None:
        if right - left == 1:
            if left > 0:
                traces[left] = solve_step(
                    left, first_history[left], second_history[left]
                )
            return
        middle = (left + right) // 2
        recurse(left, middle)
        add_cross(left, middle, right)
        recurse(middle, right)

    recurse(0, n_steps + 1)
    return traces, first_history, second_history


def march_convolution_system(
    n_steps: int,
    kernels: list[np.ndarray],
    kernel_sources: list[int],
    initial_traces: np.ndarray,
    solve_step: Callable[[int, np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """CDQ/FFT march for several kernel/source convolution histories."""

    if len(kernels) != len(kernel_sources):
        raise ValueError("each kernel must identify one source trace")
    if any(len(kernel) < n_steps + 1 for kernel in kernels):
        raise ValueError("a convolution kernel is too short")
    initial = np.asarray(initial_traces, dtype=float)
    traces = np.empty((initial.size, n_steps + 1), dtype=float)
    traces[:, 0] = initial
    histories = np.zeros((len(kernels), n_steps + 1), dtype=float)

    def add_cross(left: int, middle: int, right: int) -> None:
        width = right - left
        transform_size = None
        source_transforms: dict[int, np.ndarray] = {}
        if width > 64:
            convolution_length = (middle - left) + width - 1
            transform_size = next_fast_len(convolution_length)
        for history_index, (kernel, source_index) in enumerate(
            zip(kernels, kernel_sources)
        ):
            source = traces[source_index, left:middle]
            truncated_kernel = kernel[:width]
            if width <= 64:
                convolution = np.convolve(source, truncated_kernel)
            else:
                assert transform_size is not None
                if source_index not in source_transforms:
                    source_transforms[source_index] = rfft(source, transform_size)
                convolution = irfft(
                    source_transforms[source_index]
                    * rfft(truncated_kernel, transform_size),
                    transform_size,
                )[: source.size + width - 1]
            histories[history_index, middle:right] += convolution[
                middle - left : right - left
            ]

    def recurse(left: int, right: int) -> None:
        if right - left == 1:
            if left > 0:
                new_traces = np.asarray(solve_step(left, histories[:, left]), dtype=float)
                if new_traces.shape != initial.shape:
                    raise RuntimeError("convolution-system trace shape changed")
                traces[:, left] = new_traces
            return
        middle = (left + right) // 2
        recurse(left, middle)
        add_cross(left, middle, right)
        recurse(middle, right)

    recurse(0, n_steps + 1)
    return traces, histories


def inverse_z_transform_weights(
    samples: np.ndarray,
    radius: float,
    n_steps: int,
) -> np.ndarray:
    """Invert samples of K(z) taken at z^{-1}=rho*exp(-i theta)."""

    values = np.asarray(samples, dtype=complex)
    scaled = np.fft.ifft(values)
    weights = scaled[: n_steps + 1] / radius ** np.arange(n_steps + 1)
    imaginary_ratio = float(
        np.max(np.abs(np.imag(weights)))
        / max(np.max(np.abs(np.real(weights))), np.finfo(float).tiny)
    )
    if imaginary_ratio > 5.0e-7:
        raise FloatingPointError(
            f"inverse-Z weights are not numerically real ({imaginary_ratio:.3e})"
        )
    return np.real(weights)


@dataclass(frozen=True)
class CenteredCNDiscreteKernels:
    """Four right-boundary kernels for the Besse C-CN discrete TBC."""

    root_sum: np.ndarray
    root_sum_squared: np.ndarray
    root_product: np.ndarray
    root_product_squared: np.ndarray

    @classmethod
    def build(
        cls,
        advection: float,
        dispersion: float,
        dx: float,
        dt: float,
        n_steps: int,
    ) -> "CenteredCNDiscreteKernels":
        if advection <= 0.0 or dispersion <= 0.0:
            raise ValueError("C-CN DABC requires positive advection and dispersion")
        transform_size = 1 << int(np.ceil(np.log2(2 * (n_steps + 1))))
        radius = np.exp(np.log(np.finfo(float).eps) / (2.0 * transform_size))
        zeta = radius * np.exp(
            -2.0j * np.pi * np.arange(transform_size) / transform_size
        )
        ratio = (1.0 - zeta) / (1.0 + zeta)
        coefficient_a = advection * dx**2 / dispersion
        coefficient_b = 4.0 * dx**3 / (dispersion * dt) * ratio
        inside_sum = np.empty(transform_size, dtype=complex)
        inside_product = np.empty(transform_size, dtype=complex)

        # Batched companion matrices are substantially faster than calling
        # numpy.roots for every point on a long inverse-Z contour.
        chunk_size = 8192
        c3 = -(2.0 - coefficient_a)
        c1 = 2.0 - coefficient_a
        for start in range(0, transform_size, chunk_size):
            stop = min(start + chunk_size, transform_size)
            count = stop - start
            companion = np.zeros((count, 4, 4), dtype=complex)
            companion[:, 1, 0] = 1.0
            companion[:, 2, 1] = 1.0
            companion[:, 3, 2] = 1.0
            companion[:, 0, 3] = 1.0  # -c0, with c0=-1
            companion[:, 1, 3] = -c1
            companion[:, 2, 3] = -coefficient_b[start:stop]
            companion[:, 3, 3] = -c3
            roots = np.linalg.eigvals(companion)
            order = np.argsort(np.abs(roots), axis=1)
            selected = np.take_along_axis(roots, order[:, :2], axis=1)
            if np.any(np.abs(selected) >= 1.0):
                raise FloatingPointError("C-CN root separation failed inside the unit disk")
            if np.any(
                np.abs(np.take_along_axis(roots, order[:, 2:], axis=1)) <= 1.0
            ):
                raise FloatingPointError("C-CN root separation failed outside the unit disk")
            inside_sum[start:stop] = selected[:, 0] + selected[:, 1]
            inside_product[start:stop] = selected[:, 0] * selected[:, 1]

        return cls(
            inverse_z_transform_weights(inside_sum, radius, n_steps),
            inverse_z_transform_weights(inside_sum**2, radius, n_steps),
            inverse_z_transform_weights(inside_product, radius, n_steps),
            inverse_z_transform_weights(inside_product**2, radius, n_steps),
        )


def centered_cn_derivative_matrices(
    n: int,
    dx: float,
) -> tuple[csc_matrix, csc_matrix]:
    """Return the three-point D1 and five-point D3 used by the C-CN DABC."""

    if n < 8:
        raise ValueError("at least eight points are required")
    d1 = lil_matrix((n, n), dtype=float)
    d3 = lil_matrix((n, n), dtype=float)
    for row in range(2, n - 2):
        d1[row, row - 1] = -0.5 / dx
        d1[row, row + 1] = 0.5 / dx
        d3[row, row - 2] = -0.5 / dx**3
        d3[row, row - 1] = 1.0 / dx**3
        d3[row, row + 1] = -1.0 / dx**3
        d3[row, row + 2] = 0.5 / dx**3
    return d1.tocsc(), d3.tocsc()


class LinearKdVConvolutionTBCSolver:
    """BDF2 solver with a Dirichlet inflow and two CQ transparent outflow rows."""

    def __init__(
        self,
        y: np.ndarray,
        advection: float,
        dispersion: float,
        dt: float,
        n_steps: int,
    ) -> None:
        self.y = np.asarray(y, dtype=float)
        self.advection = float(advection)
        self.dispersion = float(dispersion)
        self.dt = float(dt)
        self.n_steps = int(n_steps)
        self.n = self.y.size
        if self.n < 16:
            raise ValueError("at least 16 grid points are required")
        self.dy = float(self.y[1] - self.y[0])
        if not np.allclose(np.diff(self.y), self.dy):
            raise ValueError("the CQ-TBC solver currently requires a uniform grid")
        self.d1, self.d2, self.d3 = nonperiodic_derivative_matrices(self.n, self.dy)
        self.linear = (-self.advection * self.d1 - self.dispersion * self.d3).tocsc()
        self.kernels = TransparentKernels.build(
            self.advection, self.dispersion, self.dt, self.n_steps
        )

        d1_right = self.d1.getrow(self.n - 1).toarray().ravel()
        d2_right = self.d2.getrow(self.n - 1).toarray().ravel()
        value_row = np.zeros(self.n)
        value_row[-1] = 1.0
        self.value_constraint = (
            value_row - self.kernels.inverse_lambda_squared[0] * d2_right
        )
        self.slope_constraint = (
            d1_right - self.kernels.inverse_lambda[0] * d2_right
        )
        self.boundary_rows = (0, self.n - 2, self.n - 1)

        identity = eye(self.n, format="csc")
        self.start_lu = self._factor_with_boundary(identity - self.dt * self.linear)
        self.bdf_lu = self._factor_with_boundary(
            identity - (2.0 / 3.0) * self.dt * self.linear
        )

    def _factor_with_boundary(self, matrix: csc_matrix) -> object:
        bordered = matrix.tolil()
        bordered[0, :] = 0.0
        bordered[0, 0] = 1.0
        bordered[-2, :] = self.value_constraint
        bordered[-1, :] = self.slope_constraint
        return splu(bordered.tocsc())

    def _history_rhs(
        self,
        step: int,
        second_derivative_history: list[float],
    ) -> tuple[float, float]:
        past = np.asarray(second_derivative_history[::-1], dtype=float)
        if past.size != step:
            raise RuntimeError("transparent-boundary history length mismatch")
        slope_history = float(
            np.dot(self.kernels.inverse_lambda[1 : step + 1], past)
        )
        value_history = float(
            np.dot(self.kernels.inverse_lambda_squared[1 : step + 1], past)
        )
        return value_history, slope_history

    def _set_boundary_rhs(
        self,
        rhs: np.ndarray,
        step: int,
        boundary_signal: Callable[[float], float],
        second_derivative_history: list[float],
    ) -> None:
        value_history, slope_history = self._history_rhs(
            step, second_derivative_history
        )
        rhs[0] = float(boundary_signal(step * self.dt))
        rhs[-2] = value_history
        rhs[-1] = slope_history

    def boundary_residuals(
        self,
        field: np.ndarray,
        step: int,
        boundary_signal: Callable[[float], float],
        second_derivative_history: list[float],
    ) -> tuple[float, float, float]:
        value_history, slope_history = self._history_rhs(
            step, second_derivative_history[:-1]
        )
        return (
            float(field[0] - boundary_signal(step * self.dt)),
            float(self.value_constraint @ field - value_history),
            float(self.slope_constraint @ field - slope_history),
        )

    def run(
        self,
        u0: np.ndarray,
        output_stride: int,
        boundary_signal: Callable[[float], float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Advance the field and return output times, fields and TBC residuals."""

        if output_stride < 1:
            raise ValueError("output_stride must be positive")
        signal = (lambda _time: 0.0) if boundary_signal is None else boundary_signal
        initial = np.asarray(u0, dtype=float).copy()
        if initial.shape != (self.n,):
            raise ValueError("u0 has the wrong shape")
        initial[0] = float(signal(0.0))

        times = [0.0]
        fields = [initial.copy()]
        residuals = [(initial[0] - signal(0.0), np.nan, np.nan)]
        states = [initial.copy(), initial.copy()]

        def solve_step(
            step: int,
            slope_history: float,
            value_history: float,
        ) -> float:
            if step == 1:
                rhs = states[1].copy()
                lu = self.start_lu
            else:
                rhs = (4.0 / 3.0) * states[1] - (1.0 / 3.0) * states[0]
                lu = self.bdf_lu
            rhs[0] = float(signal(step * self.dt))
            rhs[-2] = value_history
            rhs[-1] = slope_history
            new_state = lu.solve(rhs)
            if not np.all(np.isfinite(new_state)):
                raise FloatingPointError(
                    f"non-finite state at transparent-boundary step {step}"
                )
            states[0], states[1] = states[1], new_state
            if step % output_stride == 0 or step == self.n_steps:
                times.append(step * self.dt)
                fields.append(new_state.copy())
                residuals.append(
                    (
                        float(new_state[0] - signal(step * self.dt)),
                        float(self.value_constraint @ new_state - value_history),
                        float(self.slope_constraint @ new_state - slope_history),
                    )
                )
            return float((self.d2.getrow(-1) @ new_state).item())

        initial_trace = float((self.d2.getrow(-1) @ initial).item())
        march_two_cq_convolutions(
            self.n_steps,
            self.kernels.inverse_lambda,
            self.kernels.inverse_lambda_squared,
            initial_trace,
            solve_step,
        )

        return np.asarray(times), np.asarray(fields), np.asarray(residuals)


class CenteredCNDABCSolver:
    """Linear C-CN scheme with its matched right discrete artificial boundary."""

    def __init__(
        self,
        y: np.ndarray,
        advection: float,
        dispersion: float,
        dt: float,
        n_steps: int,
    ) -> None:
        self.y = np.asarray(y, dtype=float)
        self.n = self.y.size
        self.dx = float(self.y[1] - self.y[0])
        self.advection = float(advection)
        self.dispersion = float(dispersion)
        self.dt = float(dt)
        self.n_steps = int(n_steps)
        self.d1, self.d3 = centered_cn_derivative_matrices(self.n, self.dx)
        self.linear = (-self.advection * self.d1 - self.dispersion * self.d3).tocsc()
        self.kernels = CenteredCNDiscreteKernels.build(
            self.advection,
            self.dispersion,
            self.dx,
            self.dt,
            self.n_steps,
        )
        identity = eye(self.n, format="csc")
        self.left_matrix = (identity - 0.5 * self.dt * self.linear).tocsc()
        self.right_matrix = (identity + 0.5 * self.dt * self.linear).tocsc()

        self.first_constraint = np.zeros(self.n)
        self.first_constraint[-1] = 1.0
        self.first_constraint[-2] = -self.kernels.root_sum[0]
        self.first_constraint[-3] = self.kernels.root_product[0]
        self.second_constraint = np.zeros(self.n)
        self.second_constraint[-1] = 1.0
        self.second_constraint[-2] = -2.0 * self.kernels.root_sum[0]
        self.second_constraint[-3] = self.kernels.root_sum_squared[0]
        self.second_constraint[-5] = -self.kernels.root_product_squared[0]
        bordered = self.left_matrix.tolil()
        bordered[0, :] = 0.0
        bordered[0, 0] = 1.0
        bordered[1, :] = 0.0
        bordered[1, 1] = 1.0
        bordered[-2, :] = self.first_constraint
        bordered[-1, :] = self.second_constraint
        self.lu = splu(bordered.tocsc())

    def run(
        self,
        initial: np.ndarray,
        output_stride: int,
        boundary_signal: Callable[[float], float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if output_stride < 1:
            raise ValueError("output_stride must be positive")
        signal = (lambda _time: 0.0) if boundary_signal is None else boundary_signal
        current = np.asarray(initial, dtype=float).copy()
        delay = self.dx / self.advection
        current[0] = float(signal(0.0))
        current[1] = 0.0
        states = [current]
        times = [0.0]
        residuals = [(0.0, np.nan, np.nan)]

        def delayed_signal(time: float) -> float:
            return 0.0 if time <= delay else float(signal(time - delay))

        holder = [current]

        def solve_step(step: int, histories: np.ndarray) -> np.ndarray:
            previous = holder[0]
            rhs = np.asarray(self.right_matrix @ previous).ravel()
            time = step * self.dt
            rhs[0] = float(signal(time))
            rhs[1] = delayed_signal(time)
            h1, h2, h3, h4 = histories
            rhs[-2] = h1 - h3
            rhs[-1] = 2.0 * h1 - h2 + h4
            new_state = self.lu.solve(rhs)
            if not np.all(np.isfinite(new_state)):
                raise FloatingPointError(f"non-finite C-CN DABC state at step {step}")
            holder[0] = new_state
            if step % output_stride == 0 or step == self.n_steps:
                times.append(time)
                states.append(new_state.copy())
                residuals.append(
                    (
                        float(new_state[0] - signal(time)),
                        float(self.first_constraint @ new_state - (h1 - h3)),
                        float(
                            self.second_constraint @ new_state
                            - (2.0 * h1 - h2 + h4)
                        ),
                    )
                )
            return np.array((new_state[-2], new_state[-3], new_state[-5]))

        march_convolution_system(
            self.n_steps,
            [
                self.kernels.root_sum,
                self.kernels.root_sum_squared,
                self.kernels.root_product,
                self.kernels.root_product_squared,
            ],
            [0, 1, 1, 2],
            np.array((current[-2], current[-3], current[-5])),
            solve_step,
        )
        return np.asarray(times), np.asarray(states), np.asarray(residuals)


class _ExperimentalCoastalCQTBCSolver:
    """Unused continuous-CQ coastal prototype retained for method comparison.

    Production and all reported coastal results use
    :class:`CoastalCNAB2DABCSolver`, whose boundary is matched to its interior
    discretisation.  This prototype is intentionally private and is never
    selected by the command-line interface.

    The coordinate ``y=L-x`` increases from the 15 m offshore inflow toward
    the 5 m nearshore outflow.  This makes the propagation sign positive and
    puts the two transparent conditions at the right boundary, matching the
    root convention used in the continuous KdV TBC derivation.

    The variable-depth Green-normalised interior and the original second-order
    directional-upwind nonlinear term are the same as in the baseline solver.
    Only the two zero-derivative outflow rows are replaced.  The transparent
    operator is linearised about the constant 5 m exterior shelf; this is an
    explicit modelling approximation and is measured by pulse tests below.
    """

    def __init__(
        self,
        y: np.ndarray,
        depth_ratio: np.ndarray,
        epsilon: float,
        mu: float,
        dt: float,
        n_steps: int,
    ) -> None:
        self.y = np.asarray(y, dtype=float)
        self.depth_ratio = np.asarray(depth_ratio, dtype=float)
        if self.y.shape != self.depth_ratio.shape or np.any(self.depth_ratio <= 0.0):
            raise ValueError("y and positive depth_ratio must have matching shapes")
        self.n = self.y.size
        self.dy = float(self.y[1] - self.y[0])
        if not np.allclose(np.diff(self.y), self.dy):
            raise ValueError("the coastal CQ-TBC solver requires a uniform grid")
        self.epsilon = float(epsilon)
        self.mu = float(mu)
        self.dt = float(dt)
        self.n_steps = int(n_steps)
        self.d1, self.d2, self.d3 = nonperiodic_derivative_matrices(self.n, self.dy)

        self.root_depth = np.sqrt(self.depth_ratio)
        self.green_to_surface = self.depth_ratio ** (-0.25)
        self.surface_to_green = 1.0 / self.green_to_surface
        self.gamma = 1.5 * self.epsilon * self.depth_ratio ** (-0.5)
        self.delta = (self.mu / 6.0) * self.depth_ratio ** 2.5

        p_matrix = diags(self.root_depth, format="csc")
        r_matrix = diags(self.green_to_surface, format="csc")
        s_matrix = diags(self.surface_to_green, format="csc")
        b_matrix = diags(self.delta, format="csc")
        shoaling_advection = 0.5 * (p_matrix @ self.d1 + self.d1 @ p_matrix)
        surface_linear = -(shoaling_advection + b_matrix @ self.d3)
        self.linear = (s_matrix @ surface_linear @ r_matrix).tocsc()

        self.outflow_advection = float(self.root_depth[-1])
        self.outflow_dispersion = float(self.delta[-1])
        shelf_nodes = min(8, self.n)
        if not np.allclose(
            self.depth_ratio[-shelf_nodes:], self.depth_ratio[-1], rtol=0.0, atol=1.0e-13
        ):
            raise ValueError("the last seven-point stencil must lie on a constant-depth shelf")
        self.kernels = TransparentKernels.build(
            self.outflow_advection,
            self.outflow_dispersion,
            self.dt,
            self.n_steps,
        )

        d1_surface = (self.d1.getrow(-1) @ r_matrix).toarray().ravel()
        d2_surface = (self.d2.getrow(-1) @ r_matrix).toarray().ravel()
        surface_value = np.zeros(self.n)
        surface_value[-1] = self.green_to_surface[-1]
        self.value_constraint = (
            surface_value - self.kernels.inverse_lambda_squared[0] * d2_surface
        )
        self.slope_constraint = (
            d1_surface - self.kernels.inverse_lambda[0] * d2_surface
        )
        self._d2_surface_right = d2_surface

        identity = eye(self.n, format="csc")
        self.start_lu = self._factor_with_boundary(identity - self.dt * self.linear)
        self.bdf_lu = self._factor_with_boundary(
            identity - (2.0 / 3.0) * self.dt * self.linear
        )

    def _factor_with_boundary(self, matrix: csc_matrix) -> object:
        bordered = matrix.tolil()
        bordered[0, :] = 0.0
        bordered[0, 0] = 1.0
        bordered[-2, :] = self.value_constraint
        bordered[-1, :] = self.slope_constraint
        return splu(bordered.tocsc())

    def to_normalized(self, surface: np.ndarray) -> np.ndarray:
        scale = self.surface_to_green if surface.ndim == 1 else self.surface_to_green[None, :]
        return scale * surface

    def to_surface(self, normalized: np.ndarray) -> np.ndarray:
        scale = self.green_to_surface if normalized.ndim == 1 else self.green_to_surface[None, :]
        return scale * normalized

    def nonlinear(self, normalized: np.ndarray) -> np.ndarray:
        """Original second-order directional-upwind nonlinearity in y."""

        surface = self.to_surface(normalized)
        backward = np.empty_like(surface)
        forward = np.empty_like(surface)
        backward[0] = (surface[1] - surface[0]) / self.dy
        backward[1] = backward[0]
        backward[2:] = (
            3.0 * surface[2:] - 4.0 * surface[1:-1] + surface[:-2]
        ) / (2.0 * self.dy)
        forward[-1] = (surface[-1] - surface[-2]) / self.dy
        forward[-2] = forward[-1]
        forward[:-2] = (
            -3.0 * surface[:-2] + 4.0 * surface[1:-1] - surface[2:]
        ) / (2.0 * self.dy)
        characteristic_speed = self.gamma * surface
        derivative = np.where(characteristic_speed >= 0.0, backward, forward)
        return self.surface_to_green * (-characteristic_speed * derivative)

    def run(
        self,
        initial_surface: np.ndarray,
        output_stride: int,
        boundary_signal: Callable[[float], float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return times, shoreward surface/normalised fields and TBC residuals."""

        if output_stride < 1:
            raise ValueError("output_stride must be positive")
        signal = (lambda _time: 0.0) if boundary_signal is None else boundary_signal
        initial = self.to_normalized(np.asarray(initial_surface, dtype=float))
        if initial.shape != (self.n,):
            raise ValueError("initial_surface has the wrong shape")
        initial[0] = self.surface_to_green[0] * float(signal(0.0))

        times = [0.0]
        surface_outputs = [self.to_surface(initial)]
        normalized_outputs = [initial.copy()]
        residuals = [(surface_outputs[0][0] - signal(0.0), np.nan, np.nan)]
        states = [initial.copy(), initial.copy()]
        nonlinear_states = [self.nonlinear(initial), self.nonlinear(initial)]

        def solve_step(
            step: int,
            slope_history: float,
            value_history: float,
        ) -> float:
            if step == 1:
                rhs = states[1] + self.dt * nonlinear_states[1]
                lu = self.start_lu
            else:
                rhs = (
                    (4.0 / 3.0) * states[1]
                    - (1.0 / 3.0) * states[0]
                    + (2.0 / 3.0)
                    * self.dt
                    * (2.0 * nonlinear_states[1] - nonlinear_states[0])
                )
                lu = self.bdf_lu
            rhs = np.asarray(rhs, dtype=float)
            rhs[0] = self.surface_to_green[0] * float(signal(step * self.dt))
            rhs[-2] = value_history
            rhs[-1] = slope_history
            new_state = lu.solve(rhs)
            if not np.all(np.isfinite(new_state)):
                previous_amplitude = float(np.max(np.abs(self.to_surface(states[1]))))
                raise FloatingPointError(
                    f"non-finite coastal CQ-TBC state at step {step}; "
                    f"previous max|u|={previous_amplitude:.6g}"
                )
            new_nonlinear = self.nonlinear(new_state)
            states[0], states[1] = states[1], new_state
            nonlinear_states[0], nonlinear_states[1] = nonlinear_states[1], new_nonlinear
            if step % output_stride == 0 or step == self.n_steps:
                surface = self.to_surface(new_state)
                times.append(step * self.dt)
                surface_outputs.append(surface)
                normalized_outputs.append(new_state.copy())
                residuals.append(
                    (
                        float(surface[0] - signal(step * self.dt)),
                        float(self.value_constraint @ new_state - value_history),
                        float(self.slope_constraint @ new_state - slope_history),
                    )
                )
            return float(self._d2_surface_right @ new_state)

        march_two_cq_convolutions(
            self.n_steps,
            self.kernels.inverse_lambda,
            self.kernels.inverse_lambda_squared,
            float(self._d2_surface_right @ initial),
            solve_step,
        )
        return (
            np.asarray(times),
            np.asarray(surface_outputs),
            np.asarray(normalized_outputs),
            np.asarray(residuals),
        )


class CoastalCNAB2DABCSolver:
    """Variable-depth CNAB2 vKdV with the matched C-CN discrete outflow."""

    def __init__(
        self,
        y: np.ndarray,
        depth_ratio: np.ndarray,
        epsilon: float,
        mu: float,
        dt: float,
        n_steps: int,
    ) -> None:
        self.y = np.asarray(y, dtype=float)
        self.depth_ratio = np.asarray(depth_ratio, dtype=float)
        if self.y.shape != self.depth_ratio.shape or np.any(self.depth_ratio <= 0.0):
            raise ValueError("y and positive depth_ratio must have matching shapes")
        self.n = self.y.size
        self.dy = float(self.y[1] - self.y[0])
        self.epsilon = float(epsilon)
        self.mu = float(mu)
        self.dt = float(dt)
        self.n_steps = int(n_steps)
        self.d1, self.d3 = centered_cn_derivative_matrices(self.n, self.dy)

        self.root_depth = np.sqrt(self.depth_ratio)
        self.green_to_surface = self.depth_ratio ** (-0.25)
        self.surface_to_green = 1.0 / self.green_to_surface
        self.gamma = 1.5 * self.epsilon * self.depth_ratio ** (-0.5)
        self.delta = (self.mu / 6.0) * self.depth_ratio ** 2.5
        p_matrix = diags(self.root_depth, format="csc")
        r_matrix = diags(self.green_to_surface, format="csc")
        s_matrix = diags(self.surface_to_green, format="csc")
        b_matrix = diags(self.delta, format="csc")
        shoaling_advection = 0.5 * (p_matrix @ self.d1 + self.d1 @ p_matrix)
        surface_linear = -(shoaling_advection + b_matrix @ self.d3)
        self.linear = (s_matrix @ surface_linear @ r_matrix).tocsc()

        self.outflow_advection = float(self.root_depth[-1])
        self.outflow_dispersion = float(self.delta[-1])
        if not np.allclose(self.depth_ratio[-8:], self.depth_ratio[-1], atol=1.0e-13):
            raise ValueError("the DABC outflow stencil must be on a constant shelf")
        self.kernels = CenteredCNDiscreteKernels.build(
            self.outflow_advection,
            self.outflow_dispersion,
            self.dy,
            self.dt,
            self.n_steps,
        )

        identity = eye(self.n, format="csc")
        self.left_matrix = (identity - 0.5 * self.dt * self.linear).tocsc()
        self.right_matrix = (identity + 0.5 * self.dt * self.linear).tocsc()
        self.first_constraint = np.zeros(self.n)
        self.first_constraint[-1] = self.green_to_surface[-1]
        self.first_constraint[-2] = (
            -self.kernels.root_sum[0] * self.green_to_surface[-2]
        )
        self.first_constraint[-3] = (
            self.kernels.root_product[0] * self.green_to_surface[-3]
        )
        self.second_constraint = np.zeros(self.n)
        self.second_constraint[-1] = self.green_to_surface[-1]
        self.second_constraint[-2] = (
            -2.0 * self.kernels.root_sum[0] * self.green_to_surface[-2]
        )
        self.second_constraint[-3] = (
            self.kernels.root_sum_squared[0] * self.green_to_surface[-3]
        )
        self.second_constraint[-5] = (
            -self.kernels.root_product_squared[0] * self.green_to_surface[-5]
        )
        bordered = self.left_matrix.tolil()
        bordered[0, :] = 0.0
        bordered[0, 0] = 1.0
        bordered[1, :] = 0.0
        bordered[1, 1] = 1.0
        bordered[-2, :] = self.first_constraint
        bordered[-1, :] = self.second_constraint
        self.lu = splu(bordered.tocsc())

    def to_normalized(self, surface: np.ndarray) -> np.ndarray:
        scale = self.surface_to_green if surface.ndim == 1 else self.surface_to_green[None, :]
        return scale * surface

    def to_surface(self, normalized: np.ndarray) -> np.ndarray:
        scale = self.green_to_surface if normalized.ndim == 1 else self.green_to_surface[None, :]
        return scale * normalized

    def nonlinear(self, normalized: np.ndarray) -> np.ndarray:
        surface = self.to_surface(normalized)
        backward = np.empty_like(surface)
        forward = np.empty_like(surface)
        backward[0] = (surface[1] - surface[0]) / self.dy
        backward[1] = backward[0]
        backward[2:] = (
            3.0 * surface[2:] - 4.0 * surface[1:-1] + surface[:-2]
        ) / (2.0 * self.dy)
        forward[-1] = (surface[-1] - surface[-2]) / self.dy
        forward[-2] = forward[-1]
        forward[:-2] = (
            -3.0 * surface[:-2] + 4.0 * surface[1:-1] - surface[2:]
        ) / (2.0 * self.dy)
        speed = self.gamma * surface
        derivative = np.where(speed >= 0.0, backward, forward)
        return self.surface_to_green * (-speed * derivative)

    def run(
        self,
        initial_surface: np.ndarray,
        output_stride: int,
        boundary_signal: Callable[[float], float] | None = None,
        adjacent_boundary_signal: Callable[[float], float] | None = None,
        step_diagnostic: Callable[
            [int, np.ndarray, np.ndarray, np.ndarray], None
        ]
        | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """March the coastal problem with two offshore incident traces.

        ``boundary_signal`` prescribes the surface at ``y=0``.  When
        ``adjacent_boundary_signal`` is omitted, the value at ``y=dy`` uses
        the legacy nondispersive delay ``dy/sqrt(d_offshore)``.  Supplying the
        adjacent trace permits a frequency-by-frequency dispersive lifting
        while preserving backward compatibility with all existing runs.  The
        optional ``step_diagnostic`` receives ``(step, previous, new,
        explicit_rhs)`` at every completed step and does not alter the solve.
        """
        if output_stride < 1:
            raise ValueError("output_stride must be positive")
        signal = (lambda _time: 0.0) if boundary_signal is None else boundary_signal
        current = self.to_normalized(np.asarray(initial_surface, dtype=float))
        delay = self.dy / float(self.root_depth[0])

        def delayed_signal(time: float) -> float:
            return 0.0 if time <= delay else float(signal(time - delay))

        second_signal = (
            delayed_signal
            if adjacent_boundary_signal is None
            else adjacent_boundary_signal
        )

        current[0] = self.surface_to_green[0] * float(signal(0.0))
        current[1] = self.surface_to_green[1] * float(second_signal(0.0))
        nonlinear_current = self.nonlinear(current)
        holder = [current, nonlinear_current, nonlinear_current]
        times = [0.0]
        surface_outputs = [self.to_surface(current)]
        normalized_outputs = [current.copy()]
        residuals = [(surface_outputs[0][0] - signal(0.0), np.nan, np.nan)]

        def solve_step(step: int, histories: np.ndarray) -> np.ndarray:
            previous, nonlinearity, older_nonlinearity = holder
            if step == 1:
                explicit = nonlinearity
            else:
                explicit = 1.5 * nonlinearity - 0.5 * older_nonlinearity
            rhs = np.asarray(self.right_matrix @ previous).ravel() + self.dt * explicit
            time = step * self.dt
            rhs[0] = self.surface_to_green[0] * float(signal(time))
            rhs[1] = self.surface_to_green[1] * float(second_signal(time))
            h1, h2, h3, h4 = histories
            rhs[-2] = h1 - h3
            rhs[-1] = 2.0 * h1 - h2 + h4
            new_state = self.lu.solve(rhs)
            if not np.all(np.isfinite(new_state)):
                previous_amplitude = float(np.max(np.abs(self.to_surface(previous))))
                raise FloatingPointError(
                    f"non-finite coastal C-CN DABC state at step {step}; "
                    f"previous max|u|={previous_amplitude:.6g}"
                )
            if step_diagnostic is not None:
                step_diagnostic(step, previous, new_state, explicit)
            new_nonlinearity = self.nonlinear(new_state)
            holder[0] = new_state
            holder[2] = nonlinearity
            holder[1] = new_nonlinearity
            surface = self.to_surface(new_state)
            if step % output_stride == 0 or step == self.n_steps:
                times.append(time)
                surface_outputs.append(surface)
                normalized_outputs.append(new_state.copy())
                residuals.append(
                    (
                        float(surface[0] - signal(time)),
                        float(self.first_constraint @ new_state - (h1 - h3)),
                        float(
                            self.second_constraint @ new_state
                            - (2.0 * h1 - h2 + h4)
                        ),
                    )
                )
            return np.array((surface[-2], surface[-3], surface[-5]))

        initial_surface_field = self.to_surface(current)
        march_convolution_system(
            self.n_steps,
            [
                self.kernels.root_sum,
                self.kernels.root_sum_squared,
                self.kernels.root_product,
                self.kernels.root_product_squared,
            ],
            [0, 1, 1, 2],
            np.array(
                (
                    initial_surface_field[-2],
                    initial_surface_field[-3],
                    initial_surface_field[-5],
                )
            ),
            solve_step,
        )
        return (
            np.asarray(times),
            np.asarray(surface_outputs),
            np.asarray(normalized_outputs),
            np.asarray(residuals),
        )


def gaussian_wave_packet(y: np.ndarray, p: LinearTBCParameters) -> np.ndarray:
    shifted = np.asarray(y) - p.packet_centre
    envelope = np.exp(-(shifted / p.packet_width) ** 2)
    return p.amplitude * envelope * np.cos(p.carrier_wavenumber * shifted)


def whole_line_spectral_reference(
    coordinates: np.ndarray,
    times: np.ndarray,
    p: LinearTBCParameters,
) -> np.ndarray:
    """Return a large-period Fourier reference restricted to the test interval."""

    reference_left = -80.0
    reference_right = 100.0
    reference_points = 1 << 15
    reference_grid = np.linspace(
        reference_left, reference_right, reference_points, endpoint=False
    )
    spacing = (reference_right - reference_left) / reference_points
    wavenumbers = 2.0 * np.pi * np.fft.fftfreq(reference_points, d=spacing)
    initial = gaussian_wave_packet(reference_grid, p)
    transformed = np.fft.fft(initial)
    frequency = p.advection * wavenumbers - p.dispersion * wavenumbers**3
    output = np.empty((times.size, coordinates.size))
    for row, time in enumerate(times):
        field = np.fft.ifft(transformed * np.exp(-1.0j * frequency * time)).real
        output[row] = np.interp(coordinates, reference_grid, field)
    return output


def _trapz_squared(field: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    return np.trapezoid(np.asarray(field) ** 2, coordinates, axis=-1)


def run_constant_depth_validation(output_directory: Path) -> dict[str, float | bool]:
    """Run the first transparent-boundary benchmark and save reproducible outputs."""

    p = LinearTBCParameters()
    if p.group_velocity <= 0.0:
        raise ValueError("the selected packet must travel toward the transparent boundary")
    output_directory.mkdir(parents=True, exist_ok=True)
    raw_directory = output_directory / "raw_data"
    raw_directory.mkdir(parents=True, exist_ok=True)

    y = np.linspace(0.0, p.length, p.n_x)
    initial = gaussian_wave_packet(y, p)
    solver = LinearKdVConvolutionTBCSolver(
        y, p.advection, p.dispersion, p.dt, p.n_steps
    )
    times, numerical, residuals = solver.run(initial, p.output_stride)
    reference = whole_line_spectral_reference(y, times, p)

    initial_energy = float(_trapz_squared(initial, y))
    error_energy = _trapz_squared(numerical - reference, y) / initial_energy
    numerical_energy = _trapz_squared(numerical, y) / initial_energy
    reference_energy = _trapz_squared(reference, y) / initial_energy
    relative_l2 = np.sqrt(
        _trapz_squared(numerical - reference, y)
        / np.maximum(_trapz_squared(reference, y), 1.0e-30)
    )
    exit_time = (p.length - p.packet_centre + 4.0 * p.packet_width) / p.group_velocity
    late = times >= exit_time
    reflected_energy = float(np.max(error_energy[late])) if np.any(late) else float("nan")
    metrics = {
        "group_velocity": p.group_velocity,
        "estimated_exit_time": exit_time,
        "maximum_error_energy_fraction_after_exit": reflected_energy,
        "final_error_energy_fraction": float(error_energy[-1]),
        "final_numerical_energy_fraction": float(numerical_energy[-1]),
        "final_reference_energy_fraction": float(reference_energy[-1]),
        "maximum_absolute_boundary_residual": float(np.nanmax(np.abs(residuals))),
        "maximum_relative_l2_before_reference_exit": float(
            np.max(relative_l2[~late]) if np.any(~late) else np.nan
        ),
        "passed_reflection_gate_1e-4": bool(reflected_energy < 1.0e-4),
    }

    np.save(raw_directory / "y.npy", y)
    np.save(raw_directory / "times.npy", times)
    np.save(raw_directory / "u_numerical.npy", numerical)
    np.save(raw_directory / "u_reference.npy", reference)
    np.save(raw_directory / "boundary_residuals.npy", residuals)
    np.save(raw_directory / "kernel_inverse_lambda.npy", solver.kernels.inverse_lambda)
    np.save(
        raw_directory / "kernel_inverse_lambda_squared.npy",
        solver.kernels.inverse_lambda_squared,
    )
    with (output_directory / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump({"parameters": asdict(p), "metrics": metrics}, stream, indent=2)

    with (output_directory / "energy_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "time",
                "numerical_energy_fraction",
                "reference_energy_fraction",
                "error_energy_fraction",
                "relative_l2",
            ]
        )
        writer.writerows(
            zip(times, numerical_energy, reference_energy, error_energy, relative_l2)
        )

    return metrics


def run_discrete_abc_validation(output_directory: Path) -> dict[str, float | bool]:
    """Validate the matched C-CN DABC at the 5 m exterior coefficients."""

    output_directory.mkdir(parents=True, exist_ok=True)
    raw_directory = output_directory / "raw_data"
    raw_directory.mkdir(parents=True, exist_ok=True)
    length = 4000.0 / 150.0
    advection = np.sqrt(5.0 / 15.0)
    dispersion = (0.01 / 6.0) * (5.0 / 15.0) ** 2.5
    grid_cases = ((385, 0.016), (769, 0.008), (1537, 0.004))
    convergence_rows: list[dict[str, float]] = []

    for n_x, dt in grid_cases:
        p = replace(
            LinearTBCParameters(),
            length=length,
            n_x=n_x,
            dt=dt,
            final_time=30.0,
            advection=advection,
            dispersion=dispersion,
            amplitude=0.1,
            packet_centre=5.0,
            packet_width=1.5,
            carrier_wavenumber=1.0,
            output_stride=int(round(30.0 / dt)),
        )
        y = np.linspace(0.0, length, n_x)
        initial = gaussian_wave_packet(y, p)
        initial[:2] = 0.0
        solver = CenteredCNDABCSolver(y, advection, dispersion, dt, p.n_steps)
        times, fields, residuals = solver.run(initial, p.output_stride)
        reference = whole_line_spectral_reference(y, times, p)
        difference = fields[-1] - reference[-1]
        e2 = float(
            np.sqrt(_trapz_squared(difference, y) / _trapz_squared(reference[-1], y))
        )
        einf = float(np.max(np.abs(difference)) / np.max(np.abs(reference[-1])))
        convergence_rows.append(
            {
                "N_x": float(n_x),
                "dx": float(y[1] - y[0]),
                "dt": dt,
                "relative_E2": e2,
                "relative_Einf": einf,
            }
        )

    spacings = np.array([row["dx"] for row in convergence_rows])
    e2_values = np.array([row["relative_E2"] for row in convergence_rows])
    einf_values = np.array([row["relative_Einf"] for row in convergence_rows])
    e2_order = float(np.polyfit(np.log(spacings), np.log(e2_values), 1)[0])
    einf_order = float(np.polyfit(np.log(spacings), np.log(einf_values), 1)[0])

    p_exit = replace(
        LinearTBCParameters(),
        length=length,
        n_x=1537,
        dt=0.004,
        final_time=60.0,
        advection=advection,
        dispersion=dispersion,
        amplitude=0.1,
        packet_centre=5.0,
        packet_width=1.5,
        carrier_wavenumber=1.0,
        output_stride=50,
    )
    y_exit = np.linspace(0.0, length, p_exit.n_x)
    initial_exit = gaussian_wave_packet(y_exit, p_exit)
    initial_exit[:2] = 0.0
    exit_solver = CenteredCNDABCSolver(
        y_exit, advection, dispersion, p_exit.dt, p_exit.n_steps
    )
    exit_times, exit_fields, exit_residuals = exit_solver.run(
        initial_exit, p_exit.output_stride
    )
    exit_reference = whole_line_spectral_reference(y_exit, exit_times, p_exit)
    initial_energy = float(_trapz_squared(initial_exit, y_exit))
    numerical_energy = _trapz_squared(exit_fields, y_exit) / initial_energy
    reference_energy = _trapz_squared(exit_reference, y_exit) / initial_energy
    error_energy = _trapz_squared(exit_fields - exit_reference, y_exit) / initial_energy
    late = exit_times >= 45.0
    late_reflection = float(np.max(error_energy[late]))

    metrics = {
        "coupled_refinement_order_E2": e2_order,
        "coupled_refinement_order_Einf": einf_order,
        "fine_grid_relative_E2_at_T30": float(e2_values[-1]),
        "fine_grid_relative_Einf_at_T30": float(einf_values[-1]),
        "maximum_error_energy_fraction_after_T45": late_reflection,
        "final_numerical_energy_fraction": float(numerical_energy[-1]),
        "maximum_absolute_dabc_residual": float(np.nanmax(np.abs(exit_residuals))),
        "passed_second_order_gate": bool(1.8 < e2_order < 2.2),
        "passed_reflection_gate_1e-4": bool(late_reflection < 1.0e-4),
    }

    with (output_directory / "convergence.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=convergence_rows[0].keys())
        writer.writeheader()
        writer.writerows(convergence_rows)
    with (output_directory / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
    np.save(raw_directory / "y.npy", y_exit)
    np.save(raw_directory / "times.npy", exit_times)
    np.save(raw_directory / "u_numerical.npy", exit_fields)
    np.save(raw_directory / "u_reference.npy", exit_reference)
    np.save(raw_directory / "dabc_residuals.npy", exit_residuals)

    return metrics


def coastal_shoreward_grid(
    p: CoastalParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return x/y coordinates and depth in both orientations."""

    x, x_m = computational_grid(p)
    depth_m_x = cubic_coastal_depth(x_m, p)
    depth_ratio_x = depth_m_x / p.h_ref_m
    y = x[-1] - x[::-1]
    return x, x_m, depth_m_x, y, depth_ratio_x[::-1]


def run_coastal_pulse_validation(output_directory: Path) -> dict[str, float | bool]:
    """Compare the old zero-derivative outflow with the new CQ-TBC outflow."""

    p = replace(
        CoastalParameters(),
        n_x=1537,
        dt=0.004,
        final_time=60.0,
        output_stride=50,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    raw_directory = output_directory / "raw_data"
    raw_directory.mkdir(parents=True, exist_ok=True)
    x, x_m, depth_m_x, y, depth_y = coastal_shoreward_grid(p)
    depth_x = depth_y[::-1]

    packet_centre = 5.0
    packet_width = 1.5
    packet_wavenumber = 1.0
    shifted = y - packet_centre
    initial_y = 0.25 * np.exp(-(shifted / packet_width) ** 2) * np.cos(
        packet_wavenumber * shifted
    )
    initial_y[0] = 0.0

    open_solver = CoastalCNAB2DABCSolver(
        y, depth_y, p.epsilon, p.mu, p.dt, int(round(p.final_time / p.dt))
    )
    times, open_surface_y, open_normalized_y, residuals = open_solver.run(
        initial_y, p.output_stride
    )

    baseline_solver = IMEXBDF2Solver(
        x, depth_x, p.epsilon, p.mu, p.dt, p.propagation_sign
    )
    baseline_times, baseline_surface_x, baseline_normalized_x = baseline_solver.run(
        initial_y[::-1], p.final_time, p.output_stride
    )
    if not np.array_equal(times, baseline_times):
        raise RuntimeError("pulse-validation output times do not agree")
    baseline_surface_y = baseline_surface_x[:, ::-1]
    baseline_normalized_y = baseline_normalized_x[:, ::-1]

    initial_energy = float(np.trapezoid(open_normalized_y[0] ** 2, y))
    open_energy = np.trapezoid(open_normalized_y**2, y, axis=1) / initial_energy
    baseline_energy = np.trapezoid(baseline_normalized_y**2, y, axis=1) / initial_energy
    start_index = int(np.searchsorted(y, packet_centre))
    advective_travel = float(
        np.trapezoid(1.0 / np.sqrt(depth_y[start_index:]), y[start_index:])
    )
    exit_time = advective_travel + 4.0 * packet_width / np.sqrt(depth_y[-1])
    late = times >= exit_time
    open_late = float(np.max(open_energy[late])) if np.any(late) else float("nan")
    baseline_late = (
        float(np.max(baseline_energy[late])) if np.any(late) else float("nan")
    )
    metrics = {
        "estimated_exit_time_dimensionless": exit_time,
        "estimated_exit_time_s": exit_time * p.time_ref_s,
        "maximum_late_energy_fraction_discrete_abc": open_late,
        "maximum_late_energy_fraction_zero_derivative": baseline_late,
        "final_energy_fraction_discrete_abc": float(open_energy[-1]),
        "final_energy_fraction_zero_derivative": float(baseline_energy[-1]),
        "late_energy_improvement_factor": float(baseline_late / max(open_late, 1.0e-30)),
        "maximum_absolute_tbc_residual": float(np.nanmax(np.abs(residuals))),
        "discrete_abc_has_less_late_energy": bool(open_late < baseline_late),
    }

    np.save(raw_directory / "x_m.npy", x_m)
    np.save(raw_directory / "y_dimensionless.npy", y)
    np.save(raw_directory / "h_m.npy", depth_m_x)
    np.save(raw_directory / "times_dimensionless.npy", times)
    np.save(raw_directory / "initial_surface_y.npy", initial_y)
    np.save(raw_directory / "eta_discrete_abc_y.npy", open_surface_y)
    np.save(raw_directory / "eta_zero_derivative_y.npy", baseline_surface_y)
    np.save(raw_directory / "normalized_discrete_abc_y.npy", open_normalized_y)
    np.save(raw_directory / "normalized_zero_derivative_y.npy", baseline_normalized_y)
    np.save(raw_directory / "tbc_residuals.npy", residuals)
    with (output_directory / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump({"parameters": asdict(p), "metrics": metrics}, stream, indent=2)

    return metrics


def run_coastal_truth(
    output_directory: Path,
    quick: bool = False,
) -> dict[str, float | bool]:
    """Run the TMA truth boundary through the nonlinear coastal CQ-TBC solver."""

    p = CoastalParameters()
    if quick:
        p = replace(
            p,
            n_x=513,
            dt=0.008,
            final_time=8.0,
            output_stride=5,
            boundary_ramp_s=10.0,
            statistics_start_s=20.0,
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    raw_directory = output_directory / "raw_data"
    raw_directory.mkdir(parents=True, exist_ok=True)
    _x, x_m, depth_m_x, y, depth_y = coastal_shoreward_grid(p)

    sea_parameters = SeaStateParameters(
        significant_wave_height_m=0.3,
        peak_period_s=15.0,
        peak_enhancement_gamma=3.3,
        water_depth_m=p.h_offshore_m,
        gravity_m_s2=p.gravity,
        frequency_min_hz=0.03,
        high_frequency_taper_start_hz=0.085,
        frequency_max_hz=0.105,
        synthesis_period_s=1800.0,
        ramp_duration_s=p.boundary_ramp_s,
        random_seed=p.random_seed,
    )
    sea_state, exact_boundary = make_exact_tma_boundary(
        sea_parameters, p.time_ref_s, p.a_ref_m
    )
    solver = CoastalCNAB2DABCSolver(
        y, depth_y, p.epsilon, p.mu, p.dt, int(round(p.final_time / p.dt))
    )
    times, surface_y, normalized_y, residuals = solver.run(
        np.zeros_like(y), p.output_stride, exact_boundary
    )
    times_s = times * p.time_ref_s
    eta_x_m = p.a_ref_m * surface_y[:, ::-1]
    boundary_m = p.a_ref_m * np.asarray(exact_boundary(times))
    energy = 0.5 * np.trapezoid(normalized_y**2, y, axis=1)
    statistics_mask = times_s >= min(p.statistics_start_s, 0.75 * times_s[-1])
    if np.count_nonzero(statistics_mask) >= 3:
        slope, intercept = np.polyfit(times_s[statistics_mask], energy[statistics_mask], 1)
        mean_energy = float(np.mean(energy[statistics_mask]))
        relative_slope_per_hour = float(slope * 3600.0 / max(mean_energy, 1.0e-30))
    else:
        slope = intercept = relative_slope_per_hour = float("nan")
        mean_energy = float(np.mean(energy))
    post_indices = np.flatnonzero(statistics_mask)
    post_energy = energy[statistics_mask]
    post_peak_local = int(np.argmax(post_energy))
    post_peak_index = int(post_indices[post_peak_local])
    post_peak = float(energy[post_peak_index])
    final_quarter = times_s >= 0.75 * times_s[-1]
    final_quarter_peak = float(np.max(energy[final_quarter]))
    metrics = {
        "final_time_s": float(times_s[-1]),
        "maximum_abs_eta_m": float(np.max(np.abs(eta_x_m))),
        "maximum_abs_eta_over_local_depth": float(
            np.max(np.abs(eta_x_m) / depth_m_x[None, :])
        ),
        "mean_post_spinup_dimensionless_energy": mean_energy,
        "post_spinup_energy_slope_per_s": float(slope),
        "relative_energy_slope_per_hour": relative_slope_per_hour,
        "post_spinup_peak_energy": post_peak,
        "post_spinup_peak_time_s": float(times_s[post_peak_index]),
        "final_energy_to_post_spinup_peak": float(energy[-1] / post_peak),
        "final_quarter_peak_to_post_spinup_peak": float(final_quarter_peak / post_peak),
        "no_late_secular_energy_record": bool(final_quarter_peak < post_peak),
        "maximum_absolute_dabc_residual": float(np.nanmax(np.abs(residuals))),
        "all_fields_finite": bool(np.all(np.isfinite(eta_x_m))),
    }

    np.save(raw_directory / "x_m.npy", x_m)
    np.save(raw_directory / "y_dimensionless.npy", y)
    np.save(raw_directory / "h_m.npy", depth_m_x)
    np.save(raw_directory / "times_s.npy", times_s)
    np.save(raw_directory / "eta_truth_m.npy", eta_x_m)
    np.save(raw_directory / "boundary_truth_m.npy", boundary_m)
    np.save(raw_directory / "green_energy.npy", energy)
    np.save(raw_directory / "tbc_residuals.npy", residuals)
    np.save(raw_directory / "dabc_kernel_root_sum.npy", solver.kernels.root_sum)
    np.save(
        raw_directory / "dabc_kernel_root_sum_squared.npy",
        solver.kernels.root_sum_squared,
    )
    np.save(raw_directory / "dabc_kernel_root_product.npy", solver.kernels.root_product)
    np.save(
        raw_directory / "dabc_kernel_root_product_squared.npy",
        solver.kernels.root_product_squared,
    )
    with (raw_directory / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "purpose": "Nonlinear variable-depth KdV truth run with a matched discrete outflow",
                "coordinate_convention": "stored eta uses x=0 nearshore to x=L offshore",
                "parameters": asdict(p),
                "sea_state": sea_state.metadata(),
                "boundary": exact_boundary.metadata(),
                "transparent_exterior": {
                    "depth_m": p.nearshore_depth_m,
                    "dimensionless_advection": solver.outflow_advection,
                    "dimensionless_dispersion": solver.outflow_dispersion,
                    "method": "C-CN discrete artificial boundary from Besse et al. (2016)",
                    "history_algorithm": "exact CDQ divide-and-conquer FFT convolution",
                },
            },
            stream,
            indent=2,
        )
    with (output_directory / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--constant-validation",
        action="store_true",
        help="run the continuous-CQ reference and matched discrete-ABC benchmarks",
    )
    parser.add_argument(
        "--coastal-pulse-validation",
        action="store_true",
        help="compare the matched discrete ABC and zero-derivative coastal outflows",
    )
    parser.add_argument(
        "--coastal-production",
        action="store_true",
        help="run the 1800 s nonlinear TMA truth case with matched discrete outflow",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use a short coarse coastal-production smoke test",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "transparent_boundary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (
        args.constant_validation
        or args.coastal_pulse_validation
        or args.coastal_production
    ):
        raise SystemExit(
            "Select --constant-validation, --coastal-pulse-validation, or --coastal-production."
        )
    all_metrics: dict[str, dict[str, float | bool]] = {}
    if args.constant_validation:
        all_metrics["continuous_cq_reference"] = run_constant_depth_validation(
            args.output / "constant_depth"
        )
        all_metrics["matched_discrete_abc"] = run_discrete_abc_validation(
            args.output / "discrete_abc"
        )
    if args.coastal_pulse_validation:
        all_metrics["coastal_pulse"] = run_coastal_pulse_validation(
            args.output / "coastal_pulse"
        )
    if args.coastal_production:
        all_metrics["coastal_truth"] = run_coastal_truth(
            args.output / "coastal_truth", quick=args.quick
        )
    print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
