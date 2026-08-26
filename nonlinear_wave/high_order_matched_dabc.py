"""Fourth-order centered KdV operator with its matched CN discrete boundary.
This module is deliberately isolated from the production variable-depth solver.
It implements the constant-coefficient linear equation

    u_t + a u_y + b u_yyy = 0,  a > 0, b > 0,

using Crank--Nicolson in time and explicit fourth-order centered stencils in
space.  The seven-point combined spatial stencil produces a sixth-degree
spatial characteristic polynomial after a Z transform.  Its three roots
inside the unit disk define three non-local right-boundary constraints.

The construction follows the fully discrete strategy of Besse, Ehrhardt and
Lacroix-Violet (2016), but the sixth-degree polynomial and three-root boundary
annihilator below are derived for this module's fourth-order stencil.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.sparse import csc_matrix, eye, lil_matrix
from scipy.sparse.linalg import splu

from transparent_boundary_vkdv import (
    inverse_z_transform_weights,
    march_convolution_system,
)


# Positive-offset coefficients in
#   D u_j = sum_m c_m (u_{j+m} - u_{j-m}).
# The powers of dx are applied when matrices are assembled.
D1_POSITIVE_C4 = np.array((2.0 / 3.0, -1.0 / 12.0, 0.0))
D1_POSITIVE_C6 = np.array((3.0 / 4.0, -3.0 / 20.0, 1.0 / 60.0))
D3_POSITIVE = np.array((-13.0 / 8.0, 1.0, -1.0 / 8.0))


def d1_positive_coefficients(order: int) -> np.ndarray:
    """Return positive-offset D1 coefficients for the requested order."""

    if order == 4:
        return D1_POSITIVE_C4
    if order == 6:
        return D1_POSITIVE_C6
    raise ValueError("D1 order must be 4 or 6")


def fourth_order_derivative_matrices(
    n: int,
    dx: float,
    *,
    d1_order: int = 4,
) -> tuple[csc_matrix, csc_matrix]:
    """Return explicit centered D1 and fourth-order D3 matrices.

    Rows within three points of either boundary are left empty.  They are
    replaced by three prescribed incident traces and three matched DABC rows
    by :class:`FourthOrderCNDABCSolver`.
    """

    if n < 16:
        raise ValueError("at least sixteen points are required")
    if dx <= 0.0:
        raise ValueError("dx must be positive")
    d1_positive = d1_positive_coefficients(d1_order)
    d1 = lil_matrix((n, n), dtype=float)
    d3 = lil_matrix((n, n), dtype=float)
    for row in range(3, n - 3):
        for offset, coefficient in enumerate(d1_positive, start=1):
            if coefficient != 0.0:
                d1[row, row + offset] = coefficient / dx
                d1[row, row - offset] = -coefficient / dx
        for offset, coefficient in enumerate(D3_POSITIVE, start=1):
            d3[row, row + offset] = coefficient / dx**3
            d3[row, row - offset] = -coefficient / dx**3
    return d1.tocsc(), d3.tocsc()


def fourth_order_modified_wavenumbers(
    k: np.ndarray | float,
    dx: float,
    *,
    d1_order: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Return real modified wavenumbers for D1 and minus-imaginary D3.

    With a plane wave ``exp(i k y)``, the symbols are ``i*k1`` and
    ``-i*k3`` respectively.
    """

    theta = np.asarray(k, dtype=float) * dx
    d1_positive = d1_positive_coefficients(d1_order)
    harmonics = np.arange(1, 4, dtype=float)
    sines = np.sin(theta[..., None] * harmonics)
    k1 = 2.0 * np.sum(sines * d1_positive, axis=-1) / dx
    k3 = -2.0 * np.sum(sines * D3_POSITIVE, axis=-1) / dx**3
    return k1, k3


def fourth_order_semidiscrete_omega(
    k: np.ndarray | float,
    advection: np.ndarray | float,
    dispersion: np.ndarray | float,
    dx: float,
    *,
    d1_order: int = 4,
) -> np.ndarray:
    """Return the semidiscrete frequency for ``u_t+a*u_y+b*u_yyy=0``."""

    k1, k3 = fourth_order_modified_wavenumbers(k, dx, d1_order=d1_order)
    return np.asarray(advection) * k1 - np.asarray(dispersion) * k3


def fourth_order_cn_omega(
    k: np.ndarray | float,
    advection: np.ndarray | float,
    dispersion: np.ndarray | float,
    dx: float,
    dt: float,
    *,
    d1_order: int = 4,
) -> np.ndarray:
    """Return the real fully discrete CN frequency on its principal branch."""

    omega_sd = fourth_order_semidiscrete_omega(
        k, advection, dispersion, dx, d1_order=d1_order
    )
    return (2.0 / dt) * np.arctan(0.5 * dt * omega_sd)


def fourth_order_cn_group_velocity(
    k: np.ndarray | float,
    advection: np.ndarray | float,
    dispersion: np.ndarray | float,
    dx: float,
    dt: float,
    *,
    d1_order: int = 4,
) -> np.ndarray:
    """Return d(omega_CN)/dk for the fourth-order spatial operator."""

    theta = np.asarray(k, dtype=float) * dx
    d1_positive = d1_positive_coefficients(d1_order)
    harmonics = np.arange(1, 4, dtype=float)
    cosines = np.cos(theta[..., None] * harmonics)
    d_k1 = 2.0 * np.sum(
        cosines * harmonics * d1_positive,
        axis=-1,
    )
    d_k3 = (
        (13.0 / 4.0) * np.cos(theta)
        - 4.0 * np.cos(2.0 * theta)
        + 0.75 * np.cos(3.0 * theta)
    ) / dx**2
    derivative_sd = np.asarray(advection) * d_k1 - np.asarray(dispersion) * d_k3
    omega_sd = fourth_order_semidiscrete_omega(
        k, advection, dispersion, dx, d1_order=d1_order
    )
    return derivative_sd / (1.0 + (0.5 * dt * omega_sd) ** 2)


def combined_positive_coefficients(
    advection: float,
    dispersion: float,
    dx: float,
    *,
    d1_order: int = 4,
) -> np.ndarray:
    """Return kappa_m in K u=sum kappa_m(u[j+m]-u[j-m])."""

    return (
        advection * d1_positive_coefficients(d1_order) / dx
        + dispersion * D3_POSITIVE / dx**3
    )


def characteristic_polynomial_coefficients(
    advection: float,
    dispersion: float,
    dx: float,
    dt: float,
    zeta: np.ndarray | complex,
    *,
    d1_order: int = 4,
) -> np.ndarray:
    """Return descending coefficients of the sixth-degree CN spatial polynomial.

    Here ``zeta=z^{-1}``, with ``|zeta|<1`` on the inverse-Z contour.  If
    ``kappa_m`` are the positive-offset coefficients of ``a*D1+b*D3``, the
    characteristic equation after multiplication by ``r**3`` is

      k3*r^6+k2*r^5+k1*r^4+lambda*r^3-k1*r^2-k2*r-k3=0,

    where ``lambda=2/dt*(1-zeta)/(1+zeta)``.
    """

    if advection <= 0.0 or dispersion <= 0.0 or dx <= 0.0 or dt <= 0.0:
        raise ValueError("positive advection, dispersion, dx and dt are required")
    kappa1, kappa2, kappa3 = combined_positive_coefficients(
        advection, dispersion, dx, d1_order=d1_order
    )
    coefficient_scale = max(abs(kappa1), abs(kappa2), abs(kappa3))
    if abs(kappa3) <= 1.0e-8 * coefficient_scale:
        raise FloatingPointError(
            "the seven-point characteristic polynomial is degenerate or poorly "
            "scaled; use D1 order 4 or change the grid"
        )
    zeta_array = np.asarray(zeta, dtype=complex)
    lam = (2.0 / dt) * (1.0 - zeta_array) / (1.0 + zeta_array)
    coefficients = np.empty(zeta_array.shape + (7,), dtype=complex)
    coefficients[..., 0] = kappa3
    coefficients[..., 1] = kappa2
    coefficients[..., 2] = kappa1
    coefficients[..., 3] = lam
    coefficients[..., 4] = -kappa1
    coefficients[..., 5] = -kappa2
    coefficients[..., 6] = -kappa3
    return coefficients


@dataclass(frozen=True)
class FourthOrderCNDiscreteKernels:
    """Three elementary-symmetric kernels for the three stable CN roots."""

    root_sum: np.ndarray
    root_pair_sum: np.ndarray
    root_product: np.ndarray
    transform_size: int
    radius: float
    minimum_inside_gap: float
    minimum_outside_gap: float
    maximum_normalized_polynomial_residual: float
    d1_order: int

    @classmethod
    def build(
        cls,
        advection: float,
        dispersion: float,
        dx: float,
        dt: float,
        n_steps: int,
        *,
        transform_size: int | None = None,
        d1_order: int = 4,
    ) -> "FourthOrderCNDiscreteKernels":
        """Build inverse-Z kernels and audit the 3/3 root separation."""

        if n_steps < 1:
            raise ValueError("n_steps must be positive")
        minimum_size = 1 << int(np.ceil(np.log2(2 * (n_steps + 1))))
        if transform_size is None:
            transform_size = minimum_size
        if transform_size < minimum_size or transform_size & (transform_size - 1):
            raise ValueError("transform_size must be a power of two at least 2*(n_steps+1)")

        radius = float(
            np.exp(np.log(np.finfo(float).eps) / (2.0 * transform_size))
        )
        zeta = radius * np.exp(
            -2.0j * np.pi * np.arange(transform_size) / transform_size
        )
        coefficients = characteristic_polynomial_coefficients(
            advection, dispersion, dx, dt, zeta, d1_order=d1_order
        )
        monic = coefficients / coefficients[:, :1]
        root_sum = np.empty(transform_size, dtype=complex)
        root_pair_sum = np.empty(transform_size, dtype=complex)
        root_product = np.empty(transform_size, dtype=complex)
        minimum_inside_gap = np.inf
        minimum_outside_gap = np.inf
        maximum_residual = 0.0

        chunk_size = 4096
        for start in range(0, transform_size, chunk_size):
            stop = min(start + chunk_size, transform_size)
            count = stop - start
            companion = np.zeros((count, 6, 6), dtype=complex)
            companion[:, 1, 0] = 1.0
            companion[:, 2, 1] = 1.0
            companion[:, 3, 2] = 1.0
            companion[:, 4, 3] = 1.0
            companion[:, 5, 4] = 1.0
            # For x^6+c5*x^5+...+c0, the last column is
            # (-c0,-c1,...,-c5)^T.
            companion[:, :, 5] = -monic[start:stop, :0:-1]
            roots = np.linalg.eigvals(companion)
            moduli = np.abs(roots)
            inside_counts = np.sum(moduli < 1.0, axis=1)
            if np.any(inside_counts != 3):
                first = int(np.flatnonzero(inside_counts != 3)[0] + start)
                raise FloatingPointError(
                    f"fourth-order CN root separation failed at contour index {first}"
                )
            order = np.argsort(moduli, axis=1)
            stable = np.take_along_axis(roots, order[:, :3], axis=1)
            unstable = np.take_along_axis(roots, order[:, 3:], axis=1)
            minimum_inside_gap = min(
                minimum_inside_gap, float(np.min(1.0 - np.abs(stable)))
            )
            minimum_outside_gap = min(
                minimum_outside_gap, float(np.min(np.abs(unstable) - 1.0))
            )

            values = np.zeros_like(roots)
            scales = np.zeros_like(moduli)
            for coefficient in monic[start:stop].T:
                values = values * roots + coefficient[:, None]
                scales = scales * moduli + np.abs(coefficient)[:, None]
            normalized = np.abs(values) / np.maximum(scales, np.finfo(float).tiny)
            maximum_residual = max(maximum_residual, float(np.max(normalized)))

            stable_sum = np.sum(stable, axis=1)
            root_sum[start:stop] = stable_sum
            root_pair_sum[start:stop] = 0.5 * (
                stable_sum**2 - np.sum(stable**2, axis=1)
            )
            root_product[start:stop] = np.prod(stable, axis=1)

        return cls(
            root_sum=inverse_z_transform_weights(root_sum, radius, n_steps),
            root_pair_sum=inverse_z_transform_weights(
                root_pair_sum, radius, n_steps
            ),
            root_product=inverse_z_transform_weights(
                root_product, radius, n_steps
            ),
            transform_size=int(transform_size),
            radius=radius,
            minimum_inside_gap=float(minimum_inside_gap),
            minimum_outside_gap=float(minimum_outside_gap),
            maximum_normalized_polynomial_residual=float(maximum_residual),
            d1_order=int(d1_order),
        )


class FourthOrderCNDABCSolver:
    """Linear fourth-order C4--CN solver with a matched right DABC.

    Three values at the left boundary are prescribed.  At the right boundary,
    three consecutive shifts of the stable-root annihilator are imposed:

      u[J-q] - e1*u[J-q-1] + e2*u[J-q-2] - e3*u[J-q-3] = 0,

    for ``q=0,1,2``.  Products denote discrete time convolutions.
    """

    def __init__(
        self,
        y: np.ndarray,
        advection: float,
        dispersion: float,
        dt: float,
        n_steps: int,
        *,
        kernel_transform_size: int | None = None,
        d1_order: int = 4,
    ) -> None:
        self.y = np.asarray(y, dtype=float)
        if self.y.ndim != 1 or self.y.size < 16:
            raise ValueError("y must be a one-dimensional grid with at least 16 points")
        self.n = self.y.size
        self.dx = float(self.y[1] - self.y[0])
        if not np.allclose(np.diff(self.y), self.dx):
            raise ValueError("the fourth-order DABC requires a uniform grid")
        self.advection = float(advection)
        self.dispersion = float(dispersion)
        self.dt = float(dt)
        self.n_steps = int(n_steps)
        self.d1_order = int(d1_order)
        self.d1, self.d3 = fourth_order_derivative_matrices(
            self.n, self.dx, d1_order=self.d1_order
        )
        self.linear = (
            -self.advection * self.d1 - self.dispersion * self.d3
        ).tocsc()
        self.kernels = FourthOrderCNDiscreteKernels.build(
            self.advection,
            self.dispersion,
            self.dx,
            self.dt,
            self.n_steps,
            transform_size=kernel_transform_size,
            d1_order=self.d1_order,
        )

        identity = eye(self.n, format="csc")
        self.left_matrix = (identity - 0.5 * self.dt * self.linear).tocsc()
        self.right_matrix = (identity + 0.5 * self.dt * self.linear).tocsc()
        self.constraints: list[np.ndarray] = []
        for shift in range(3):
            row = np.zeros(self.n)
            anchor = self.n - 1 - shift
            row[anchor] = 1.0
            row[anchor - 1] = -self.kernels.root_sum[0]
            row[anchor - 2] = self.kernels.root_pair_sum[0]
            row[anchor - 3] = -self.kernels.root_product[0]
            self.constraints.append(row)

        bordered = self.left_matrix.tolil()
        for row in range(3):
            bordered[row, :] = 0.0
            bordered[row, row] = 1.0
        # Put q=2,1,0 on rows J-2,J-1,J.  In the transformed problem this
        # makes the zero-lag block for (u[J-2],u[J-1],u[J]) unit triangular.
        for shift, constraint in enumerate(self.constraints):
            bordered[self.n - 1 - shift, :] = constraint
        self.lu = splu(bordered.tocsc())

    def run(
        self,
        initial: np.ndarray,
        output_stride: int,
        boundary_traces: tuple[
            Callable[[float], float],
            Callable[[float], float],
            Callable[[float], float],
        ]
        | None = None,
        *,
        initial_outflow_relative_tolerance: float = 1.0e-10,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """March the finite-domain problem and return times, fields, residuals.

        The homogeneous Z-transform DABC is derived for zero initial data in
        the exterior half-line. Consequently, the rightmost six grid values
        must be negligible relative to the initial field unless a compatible
        non-zero exterior initial-data correction has first been derived.
        """

        if output_stride < 1:
            raise ValueError("output_stride must be positive")
        if boundary_traces is None:
            boundary_traces = (
                lambda _time: 0.0,
                lambda _time: 0.0,
                lambda _time: 0.0,
            )
        if len(boundary_traces) != 3:
            raise ValueError("three incident boundary traces are required")
        current = np.asarray(initial, dtype=float).copy()
        if current.shape != self.y.shape:
            raise ValueError("initial field must match y")
        if initial_outflow_relative_tolerance <= 0.0:
            raise ValueError("initial_outflow_relative_tolerance must be positive")
        field_scale = max(float(np.max(np.abs(current))), np.finfo(float).tiny)
        outflow_tail_ratio = float(np.max(np.abs(current[-6:]))) / field_scale
        if outflow_tail_ratio > initial_outflow_relative_tolerance:
            raise ValueError(
                "the homogeneous DABC requires zero/compatible exterior initial "
                "data; the rightmost six initial values are not negligible "
                f"(relative tail {outflow_tail_ratio:.3e} > "
                f"{initial_outflow_relative_tolerance:.3e})"
            )
        for row, trace in enumerate(boundary_traces):
            current[row] = float(trace(0.0))

        times = [0.0]
        fields = [current.copy()]
        residuals = [(0.0, 0.0, 0.0)]
        holder = [current]

        # Histories store u[J-1],...,u[J-5].  The three shifted
        # annihilators use source triples (0,1,2), (1,2,3), (2,3,4).
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
            rhs = np.asarray(self.right_matrix @ previous).ravel()
            time = step * self.dt
            for row, trace in enumerate(boundary_traces):
                rhs[row] = float(trace(time))
            boundary_rhs = np.empty(3)
            for shift in range(3):
                h1, h2, h3 = histories[3 * shift : 3 * shift + 3]
                boundary_rhs[shift] = h1 - h2 + h3
                rhs[self.n - 1 - shift] = boundary_rhs[shift]
            new_state = self.lu.solve(rhs)
            if not np.all(np.isfinite(new_state)):
                raise FloatingPointError(
                    f"non-finite fourth-order C4--CN state at step {step}"
                )
            holder[0] = new_state
            if step % output_stride == 0 or step == self.n_steps:
                times.append(time)
                fields.append(new_state.copy())
                residuals.append(
                    tuple(
                        float(constraint @ new_state - boundary_rhs[index])
                        for index, constraint in enumerate(self.constraints)
                    )
                )
            return new_state[-2:-7:-1].copy()

        march_convolution_system(
            self.n_steps,
            kernel_list,
            kernel_sources,
            current[-2:-7:-1].copy(),
            solve_step,
        )
        return np.asarray(times), np.asarray(fields), np.asarray(residuals)
