"""Core deterministic operators for the physical-time coastal vKdV model.
This module contains only the PDE state definition, bathymetry, non-periodic
finite-difference helpers, and the legacy IMEX-BDF2 verification solver.  The
final production calculation uses the high-order implicit-midpoint solver
assembled by :mod:`coastal_entropy_midpoint_production`.

Coordinates increase offshore.  The complete random-phase TMA record enters
at ``x=L`` and waves propagate towards ``x=0``.  The implementation is
non-periodic and uses the matched artificial-boundary construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import Callable

import numpy as np
from scipy.sparse import csc_matrix, diags, eye, lil_matrix
from scipy.sparse.linalg import splu


@dataclass(frozen=True)
class CoastalParameters:
    """Dimensional reference scales and production discretisation."""

    gravity: float = 9.81
    h_ref_m: float = 15.0
    # Characteristic amplitude Hs/2 for the production Hs=0.3 m sea state.
    a_ref_m: float = 0.15
    mu: float = 0.01
    h_offshore_m: float = 15.0
    domain_m: float = 4000.0
    transition_start_m: float = 1000.0
    transition_end_m: float = 3000.0
    nearshore_depth_m: float = 5.0
    # Preserve an 1800 s physical integration after changing h_ref to 15 m.
    final_time: float = 145.566
    dt: float = 0.002
    n_x: int = 3073
    output_stride: int = 70
    propagation_sign: int = -1
    random_seed: int = 20260718
    boundary_ramp_s: float = 150.0
    statistics_start_s: float = 500.0

    @property
    def epsilon(self) -> float:
        return self.a_ref_m / self.h_ref_m

    @property
    def kappa(self) -> float:
        return self.mu / self.epsilon

    @property
    def lambda_ref_m(self) -> float:
        return self.h_ref_m / np.sqrt(self.mu)

    @property
    def c_ref_m_s(self) -> float:
        return np.sqrt(self.gravity * self.h_ref_m)

    @property
    def time_ref_s(self) -> float:
        return self.lambda_ref_m / self.c_ref_m_s


def cubic_coastal_depth(
    x_m: np.ndarray | float,
    p: CoastalParameters,
) -> np.ndarray:
    """Return the 5 m shelf, smooth 5--15 m slope, and 15 m shelf."""

    values = np.asarray(x_m, dtype=float)
    span = p.transition_end_m - p.transition_start_m
    if span <= 0.0:
        raise ValueError("transition_end_m must exceed transition_start_m")
    s = np.clip((values - p.transition_start_m) / span, 0.0, 1.0)
    smoothstep = 3.0 * s**2 - 2.0 * s**3
    return p.nearshore_depth_m + (
        p.h_offshore_m - p.nearshore_depth_m
    ) * smoothstep


def computational_grid(
    p: CoastalParameters,
    n_x: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return dimensionless and dimensional grids on the complete domain."""

    count = p.n_x if n_x is None else int(n_x)
    if count < 16:
        raise ValueError("at least 16 grid points are required")
    x_m = np.linspace(0.0, p.domain_m, count, endpoint=True)
    return x_m / p.lambda_ref_m, x_m


def _finite_difference_weights(
    offsets: np.ndarray,
    derivative: int,
) -> np.ndarray:
    matrix = np.vstack(
        [
            offsets.astype(float) ** power / factorial(power)
            for power in range(offsets.size)
        ]
    )
    target = np.zeros(offsets.size)
    target[derivative] = 1.0
    return np.linalg.solve(matrix, target)


def nonperiodic_derivative_matrix(
    n: int,
    dx: float,
    derivative: int,
    stencil_size: int = 7,
) -> csc_matrix:
    """Build a uniform-grid derivative with one-sided edge closures."""

    if derivative < 1 or derivative >= stencil_size:
        raise ValueError("derivative order must be smaller than stencil_size")
    if stencil_size % 2 != 1 or n < stencil_size:
        raise ValueError("stencil_size must be odd and no larger than n")
    half = stencil_size // 2
    matrix = lil_matrix((n, n), dtype=float)
    for row in range(n):
        start = min(max(row - half, 0), n - stencil_size)
        columns = np.arange(start, start + stencil_size)
        offsets = columns - row
        matrix[row, columns] = (
            _finite_difference_weights(offsets, derivative) / dx**derivative
        )
    return matrix.tocsc()


def nonperiodic_derivative_matrices(
    n: int,
    dx: float,
) -> tuple[csc_matrix, csc_matrix, csc_matrix]:
    """Return non-periodic first-, second-, and third-derivative matrices."""

    return (
        nonperiodic_derivative_matrix(n, dx, 1),
        nonperiodic_derivative_matrix(n, dx, 2),
        nonperiodic_derivative_matrix(n, dx, 3),
    )


class IMEXBDF2Solver:
    """Boundary-bordered IMEX-BDF2 solver retained for PDE verification."""

    def __init__(
        self,
        x: np.ndarray,
        depth_ratio: np.ndarray,
        epsilon: float,
        mu: float,
        dt: float,
        propagation_sign: int,
    ) -> None:
        if propagation_sign not in (-1, 1):
            raise ValueError("propagation_sign must be -1 or +1")
        self.x = np.asarray(x, dtype=float)
        self.depth_ratio = np.asarray(depth_ratio, dtype=float)
        if np.any(self.depth_ratio <= 0.0):
            raise ValueError("vKdV requires strictly positive computational depth")
        self.epsilon = float(epsilon)
        self.mu = float(mu)
        self.dt = float(dt)
        self.sign = int(propagation_sign)
        self.dx = float(self.x[1] - self.x[0])
        self.n = self.x.size
        self.d1, self.d2, self.d3 = nonperiodic_derivative_matrices(
            self.n,
            self.dx,
        )

        self.root_depth = np.sqrt(self.depth_ratio)
        self.green_to_surface = self.depth_ratio ** (-0.25)
        self.surface_to_green = 1.0 / self.green_to_surface
        self.gamma = 1.5 * self.epsilon * self.depth_ratio ** (-0.5)
        self.delta = (self.mu / 6.0) * self.depth_ratio ** 2.5

        pmat = diags(self.root_depth, format="csc")
        rmat = diags(self.green_to_surface, format="csc")
        smat = diags(self.surface_to_green, format="csc")
        bmat = diags(self.delta, format="csc")
        shoaling_advection = 0.5 * (pmat @ self.d1 + self.d1 @ pmat)
        surface_linear = -self.sign * (shoaling_advection + bmat @ self.d3)
        self.linear = (smat @ surface_linear @ rmat).tocsc()

        left_d1 = (self.d1.getrow(0) @ rmat).toarray().ravel()
        left_d2 = (self.d2.getrow(0) @ rmat).toarray().ravel()
        self._left_constraint = np.vstack((left_d1, left_d2))
        self._boundary_rows = (0, 1, self.n - 1)

        identity = eye(self.n, format="csc")
        self.start_lu, self.start_rhs = self._bordered_pair(
            identity - self.dt * self.linear,
            identity,
        )
        bdf_lhs = identity - (2.0 / 3.0) * self.dt * self.linear
        bdf_lhs_lil = bdf_lhs.tolil()
        bdf_lhs_lil[0, :] = self._left_constraint[0]
        bdf_lhs_lil[1, :] = self._left_constraint[1]
        bdf_lhs_lil[-1, :] = 0.0
        bdf_lhs_lil[-1, -1] = 1.0
        self.bdf_lu = splu(bdf_lhs_lil.tocsc())

    def _bordered_pair(
        self,
        lhs: csc_matrix,
        rhs: csc_matrix,
    ) -> tuple[object, csc_matrix]:
        lhs_lil = lhs.tolil()
        rhs_lil = rhs.tolil()
        lhs_lil[0, :] = self._left_constraint[0]
        lhs_lil[1, :] = self._left_constraint[1]
        lhs_lil[-1, :] = 0.0
        lhs_lil[-1, -1] = 1.0
        for row in self._boundary_rows:
            rhs_lil[row, :] = 0.0
        return splu(lhs_lil.tocsc()), rhs_lil.tocsc()

    def to_normalized(self, surface: np.ndarray) -> np.ndarray:
        scale = (
            self.surface_to_green
            if surface.ndim == 1
            else self.surface_to_green[:, None]
        )
        return scale * surface

    def to_surface(self, normalized: np.ndarray) -> np.ndarray:
        scale = (
            self.green_to_surface
            if normalized.ndim == 1
            else self.green_to_surface[:, None]
        )
        return scale * normalized

    def nonlinear(self, normalized: np.ndarray) -> np.ndarray:
        """Return the second-order directional-upwind nonlinear drift."""

        surface = self.to_surface(normalized)
        if surface.ndim != 1:
            raise ValueError("IMEXBDF2Solver advances one field at a time")
        backward = np.empty_like(surface)
        forward = np.empty_like(surface)
        backward[0] = (surface[1] - surface[0]) / self.dx
        backward[1] = backward[0]
        backward[2:] = (
            3.0 * surface[2:]
            - 4.0 * surface[1:-1]
            + surface[:-2]
        ) / (2.0 * self.dx)
        forward[-1] = (surface[-1] - surface[-2]) / self.dx
        forward[-2] = forward[-1]
        forward[:-2] = (
            -3.0 * surface[:-2]
            + 4.0 * surface[1:-1]
            - surface[2:]
        ) / (2.0 * self.dx)
        characteristic_speed = self.sign * self.gamma * surface
        derivative = np.where(characteristic_speed >= 0.0, backward, forward)
        surface_nonlinear = -characteristic_speed * derivative
        return self.surface_to_green * surface_nonlinear

    def _set_boundary_rhs(
        self,
        rhs: np.ndarray,
        time: float,
        boundary_signal: Callable[[float], float],
    ) -> None:
        rhs[0] = 0.0
        rhs[1] = 0.0
        rhs[-1] = self.surface_to_green[-1] * float(boundary_signal(time))

    def enforce_boundary(
        self,
        normalized: np.ndarray,
        time: float,
        boundary_signal: Callable[[float], float],
    ) -> np.ndarray:
        result = np.asarray(normalized, dtype=float).copy()
        result[-1] = self.surface_to_green[-1] * float(boundary_signal(time))
        free = result[2:]
        rhs = -self._left_constraint[:, 2:] @ free
        result[:2] = np.linalg.solve(self._left_constraint[:, :2], rhs)
        return result

    def boundary_residuals(
        self,
        surface: np.ndarray,
        time: float,
        boundary_signal: Callable[[float], float],
    ) -> tuple[float, float, float]:
        return (
            float((self.d1.getrow(0) @ surface).item()),
            float((self.d2.getrow(0) @ surface).item()),
            float(surface[-1] - boundary_signal(time)),
        )

    def run(
        self,
        surface_initial: np.ndarray,
        final_time: float,
        output_stride: int,
        boundary_signal: Callable[[float], float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Advance the normalised state and return time, surface, and state."""

        signal = (
            (lambda _time: 0.0)
            if boundary_signal is None
            else boundary_signal
        )
        n_steps = int(round(final_time / self.dt))
        if not np.isclose(n_steps * self.dt, final_time):
            raise ValueError("final_time must be an integer multiple of dt")
        if output_stride < 1:
            raise ValueError("output_stride must be positive")

        normalized = self.enforce_boundary(
            self.to_normalized(np.asarray(surface_initial, dtype=float)),
            0.0,
            signal,
        )
        normalized_previous = normalized.copy()
        times = [0.0]
        normalized_fields = [normalized.copy()]
        surface_fields = [self.to_surface(normalized)]

        nonlinear_old = self.nonlinear(normalized)
        rhs = self.start_rhs @ normalized + self.dt * nonlinear_old
        self._set_boundary_rhs(rhs, self.dt, signal)
        normalized = self.start_lu.solve(rhs)
        if not np.all(np.isfinite(normalized)):
            raise FloatingPointError(
                f"non-finite state after first step at T={self.dt:.6g}"
            )
        nonlinear_now = self.nonlinear(normalized)
        if output_stride == 1:
            times.append(self.dt)
            normalized_fields.append(normalized.copy())
            surface_fields.append(self.to_surface(normalized))

        for step in range(1, n_steps):
            completed = step + 1
            new_time = completed * self.dt
            rhs = (
                (4.0 / 3.0) * normalized
                - (1.0 / 3.0) * normalized_previous
                + (2.0 / 3.0)
                * self.dt
                * (2.0 * nonlinear_now - nonlinear_old)
            )
            self._set_boundary_rhs(rhs, new_time, signal)
            normalized_new = self.bdf_lu.solve(rhs)
            if not np.all(np.isfinite(normalized_new)):
                raise FloatingPointError(
                    "non-finite state at "
                    f"T={new_time:.6g}; previous max|v|="
                    f"{float(np.max(np.abs(normalized))):.6g}"
                )
            nonlinear_old, nonlinear_now = (
                nonlinear_now,
                self.nonlinear(normalized_new),
            )
            normalized_previous, normalized = normalized, normalized_new
            if completed % output_stride == 0 or completed == n_steps:
                times.append(new_time)
                normalized_fields.append(normalized.copy())
                surface_fields.append(self.to_surface(normalized))

        return (
            np.asarray(times),
            np.asarray(surface_fields),
            np.asarray(normalized_fields),
        )
