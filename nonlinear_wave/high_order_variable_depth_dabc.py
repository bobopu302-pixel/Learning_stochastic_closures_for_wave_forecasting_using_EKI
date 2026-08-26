"""High-order variable-depth vKdV solvers with a matched linear DABC.

Origin: 3. KDV_nonlinear_case/high_order_variable_depth_dabc.py
Changes vs origin: comments/docstrings only (this provenance header added).

The shoreward coordinate is ``y = L - x``: ``y=0`` is the offshore inflow
and ``y=L`` is the nearshore outflow.  In dimensionless surface variables,

    u_T + p u_y + 0.5 p_y u + delta u_yyy = 0,
    p = sqrt(d),  delta = (mu/6) d**(5/2),  d = h/h_ref.

The advanced Green-normalised state is ``v=d**(1/4) u``.  The isolated
spatial operator is therefore

    L_v = S[-0.5(P D1 + D1 P) - B D3]R,

where ``R=diag(d**(-1/4))``, ``S=R**(-1)``, ``P=diag(sqrt(d))`` and
``B=diag(delta)``.  ``D1`` is the centred sixth-order first derivative and
``D3`` is the centred fourth-order third derivative from Experiment 12.

Three surface traces at the offshore edge close the seven-point stencil.  At
the nearshore edge, three shifts of the constant-shelf C6/C4--CN discrete
artificial boundary condition (DABC) are imposed.  Its convolutions are
evaluated in the surface variable ``u``; the constraint rows include ``R``
when applied to the normalised unknown ``v``.

This module is deliberately separate from the retained production solver.
The linear class is the Experiment-14 candidate.  Its Experiment-15 subclass
adds the retained second-order directional-upwind nonlinearity through CNAB2;
the exterior DABC itself remains linear and must therefore be validated by
finite-versus-extended-domain sensitivity tests before production use.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.sparse import csc_matrix, diags, eye
from scipy.sparse.linalg import splu
from scipy.special import betainc

from high_order_matched_dabc import (
    FourthOrderCNDiscreteKernels,
    fourth_order_derivative_matrices,
)
from transparent_boundary_vkdv import march_convolution_system


def transition_shape(values: np.ndarray | float, kind: str = "beta_C7") -> np.ndarray:
    """Return a clipped 0--1 smoothstep with the requested endpoint regularity."""

    r = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    if kind == "cubic_C1":
        return 3.0 * r**2 - 2.0 * r**3
    if kind == "septic_C3":
        return 35.0 * r**4 - 84.0 * r**5 + 70.0 * r**6 - 20.0 * r**7
    if kind == "beta_C7":
        # I_r(8,8) is a degree-15 generalized smoothstep.  Its first seven
        # derivatives vanish at both endpoints, so joining it to constant
        # shelves yields a globally C7 bathymetry.
        return np.asarray(betainc(8.0, 8.0, r), dtype=float)
    raise ValueError("kind must be cubic_C1, septic_C3 or beta_C7")


def coastal_depth_ratio_y(
    y: np.ndarray | float,
    *,
    length_ref_m: float,
    depth_ref_m: float = 15.0,
    offshore_depth_m: float = 15.0,
    nearshore_depth_m: float = 5.0,
    transition_start_m: float = 1000.0,
    transition_end_m: float = 3000.0,
    kind: str = "beta_C7",
) -> np.ndarray:
    """Return ``d=h/h_ref`` in the shoreward coordinate ``y``.

    The default profile is 15 m on 0--1 km, smoothly descends on 1--3 km,
    and is 5 m from 3 km onwards.  Extending the grid therefore extends only
    the constant nearshore shelf, which is required by the matched DABC.
    """

    if length_ref_m <= 0.0 or depth_ref_m <= 0.0:
        raise ValueError("reference length and depth must be positive")
    if offshore_depth_m <= 0.0 or nearshore_depth_m <= 0.0:
        raise ValueError("physical depths must be positive")
    if transition_end_m <= transition_start_m:
        raise ValueError("transition_end_m must exceed transition_start_m")
    y_m = np.asarray(y, dtype=float) * float(length_ref_m)
    r = (y_m - transition_start_m) / (transition_end_m - transition_start_m)
    shape = transition_shape(r, kind)
    depth_m = offshore_depth_m + (nearshore_depth_m - offshore_depth_m) * shape
    return np.asarray(depth_m / depth_ref_m, dtype=float)


def assemble_normalized_linear_operator(
    depth_ratio: np.ndarray,
    mu: float,
    d1: csc_matrix,
    d3: csc_matrix,
) -> tuple[csc_matrix, dict[str, np.ndarray]]:
    """Assemble ``S[-(PD1+D1P)/2-BD3]R`` and return its coefficient fields."""

    depth = np.asarray(depth_ratio, dtype=float)
    if depth.ndim != 1 or np.any(~np.isfinite(depth)) or np.any(depth <= 0.0):
        raise ValueError("depth_ratio must be a finite positive vector")
    if d1.shape != (depth.size, depth.size) or d3.shape != d1.shape:
        raise ValueError("derivative matrices must match depth_ratio")
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError("mu must be finite and positive")

    p = np.sqrt(depth)
    surface_to_green = depth**0.25
    green_to_surface = depth**(-0.25)
    delta = (float(mu) / 6.0) * depth**2.5
    pmat = diags(p, format="csc")
    rmat = diags(green_to_surface, format="csc")
    smat = diags(surface_to_green, format="csc")
    bmat = diags(delta, format="csc")
    surface_operator = -(0.5 * (pmat @ d1 + d1 @ pmat) + bmat @ d3)
    normalized_operator = (smat @ surface_operator @ rmat).tocsc()
    return normalized_operator, {
        "p": p,
        "delta": delta,
        "surface_to_green": surface_to_green,
        "green_to_surface": green_to_surface,
    }


class CoastalHighOrderLinearCNDABCSolver:
    """C6-D1/C4-D3--CN solver for the Green-normalised linear vKdV.

    Parameters are dimensionless.  Both ends must lie on constant-depth
    shelves: the left shelf supports the three modal inflow traces and the
    right shelf is the constant-coefficient exterior assumed by the DABC.
    """

    def __init__(
        self,
        y: np.ndarray,
        depth_ratio: np.ndarray,
        mu: float,
        dt: float,
        n_steps: int,
        *,
        d1_order: int = 6,
        kernel_transform_size: int | None = None,
        shelf_points: int = 8,
        shelf_tolerance: float = 1.0e-13,
    ) -> None:
        self.y = np.asarray(y, dtype=float)
        self.depth_ratio = np.asarray(depth_ratio, dtype=float)
        if self.y.ndim != 1 or self.y.size < 16:
            raise ValueError("y must be a one-dimensional grid with at least 16 points")
        if self.depth_ratio.shape != self.y.shape:
            raise ValueError("depth_ratio must match y")
        if np.any(~np.isfinite(self.depth_ratio)) or np.any(self.depth_ratio <= 0.0):
            raise ValueError("depth_ratio must be finite and positive")
        self.n = self.y.size
        self.dy = float(self.y[1] - self.y[0])
        if self.dy <= 0.0 or not np.allclose(np.diff(self.y), self.dy):
            raise ValueError("the high-order DABC requires a uniform increasing grid")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if int(n_steps) < 1:
            raise ValueError("n_steps must be positive")
        if d1_order != 6:
            raise ValueError("Experiment 14 requires the C6 first derivative")
        if shelf_points < 8 or 2 * shelf_points > self.n:
            raise ValueError("shelf_points must be at least 8 and fit at both ends")
        self.mu = float(mu)
        self.dt = float(dt)
        self.n_steps = int(n_steps)
        self.d1_order = int(d1_order)
        self.shelf_points = int(shelf_points)

        if not np.allclose(
            self.depth_ratio[: self.shelf_points],
            self.depth_ratio[0],
            rtol=0.0,
            atol=shelf_tolerance,
        ):
            raise ValueError("the three-trace inflow stencil must be on a constant shelf")
        if not np.allclose(
            self.depth_ratio[-self.shelf_points :],
            self.depth_ratio[-1],
            rtol=0.0,
            atol=shelf_tolerance,
        ):
            raise ValueError("the DABC outflow stencil must be on a constant shelf")

        self.d1, self.d3 = fourth_order_derivative_matrices(
            self.n, self.dy, d1_order=self.d1_order
        )
        self.linear, coefficients = assemble_normalized_linear_operator(
            self.depth_ratio, self.mu, self.d1, self.d3
        )
        self.root_depth = coefficients["p"]
        self.delta = coefficients["delta"]
        self.surface_to_green = coefficients["surface_to_green"]
        self.green_to_surface = coefficients["green_to_surface"]

        self.outflow_advection = float(self.root_depth[-1])
        self.outflow_dispersion = float(self.delta[-1])
        self.kernels = FourthOrderCNDiscreteKernels.build(
            self.outflow_advection,
            self.outflow_dispersion,
            self.dy,
            self.dt,
            self.n_steps,
            transform_size=kernel_transform_size,
            d1_order=self.d1_order,
        )

        identity = eye(self.n, format="csc")
        self.left_matrix = (identity - 0.5 * self.dt * self.linear).tocsc()
        self.right_matrix = (identity + 0.5 * self.dt * self.linear).tocsc()

        # The annihilator is written in surface u.  Since the linear system
        # advances v and u=R v, every spatial coefficient carries R at its
        # own node.  This matters whenever a constraint is moved off a
        # strictly constant shelf and makes the state conversion unambiguous.
        self.constraints: list[np.ndarray] = []
        for shift in range(3):
            anchor = self.n - 1 - shift
            row = np.zeros(self.n)
            row[anchor] = self.green_to_surface[anchor]
            row[anchor - 1] = (
                -self.kernels.root_sum[0] * self.green_to_surface[anchor - 1]
            )
            row[anchor - 2] = (
                self.kernels.root_pair_sum[0]
                * self.green_to_surface[anchor - 2]
            )
            row[anchor - 3] = (
                -self.kernels.root_product[0]
                * self.green_to_surface[anchor - 3]
            )
            self.constraints.append(row)

        bordered = self.left_matrix.tolil()
        for row in range(3):
            bordered[row, :] = 0.0
            bordered[row, row] = 1.0
        for shift, constraint in enumerate(self.constraints):
            bordered[self.n - 1 - shift, :] = constraint
        self.lu = splu(bordered.tocsc())

    def to_normalized(self, surface: np.ndarray) -> np.ndarray:
        """Map surface ``u`` to Green-normalised ``v`` along the last axis."""

        values = np.asarray(surface, dtype=float)
        if values.shape[-1] != self.n:
            raise ValueError("the last surface axis must match y")
        return values * self.surface_to_green

    def to_surface(self, normalized: np.ndarray) -> np.ndarray:
        """Map Green-normalised ``v`` to surface ``u`` along the last axis."""

        values = np.asarray(normalized, dtype=float)
        if values.shape[-1] != self.n:
            raise ValueError("the last normalized axis must match y")
        return values * self.green_to_surface

    def run(
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
        initial_outflow_relative_tolerance: float = 1.0e-10,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """March and return times, surface fields, normalised fields and residuals."""

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
        if initial_outflow_relative_tolerance <= 0.0:
            raise ValueError("initial_outflow_relative_tolerance must be positive")

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
                "the homogeneous DABC requires zero/compatible exterior initial "
                "data; the rightmost six surface values are not negligible "
                f"(relative tail {tail_ratio:.3e} > "
                f"{initial_outflow_relative_tolerance:.3e})"
            )
        initial_trace_values = tuple(float(trace(0.0)) for trace in boundary_traces)
        for row, trace_value in enumerate(initial_trace_values):
            surface_initial[row] = trace_value
        current = self.to_normalized(surface_initial)

        times = [0.0]
        surface_outputs = [surface_initial.copy()]
        normalized_outputs = [current.copy()]
        initial_left_residuals = tuple(
            float(surface_initial[row] - trace_value)
            for row, trace_value in enumerate(initial_trace_values)
        )
        # A homogeneous exterior correction makes these zero for the intended
        # zero-tail initial data.  Reporting their actual values avoids hiding
        # a small but permitted non-zero tail behind hard-coded diagnostics.
        initial_right_residuals = tuple(
            float(constraint @ current) for constraint in self.constraints
        )
        residuals = [initial_left_residuals + initial_right_residuals]
        holder = [current]

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
            time_value = step * self.dt
            trace_values = tuple(float(trace(time_value)) for trace in boundary_traces)
            for row, trace_value in enumerate(trace_values):
                rhs[row] = self.surface_to_green[row] * trace_value

            boundary_rhs = np.empty(3)
            for shift in range(3):
                h1, h2, h3 = histories[3 * shift : 3 * shift + 3]
                boundary_rhs[shift] = h1 - h2 + h3
                rhs[self.n - 1 - shift] = boundary_rhs[shift]

            new_state = self.lu.solve(rhs)
            if not np.all(np.isfinite(new_state)):
                raise FloatingPointError(
                    f"non-finite high-order variable-depth state at step {step}"
                )
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
                residuals.append(
                    left_residuals + right_residuals
                )
            # Convolution source traces are surface u[J-1],...,u[J-5].
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


class CoastalHighOrderCNAB2DABCSolver(CoastalHighOrderLinearCNDABCSolver):
    """Weakly nonlinear C6/C4--CNAB2 extension with the matched linear DABC.

    The implicit linear part and all six boundary rows are exactly those of
    :class:`CoastalHighOrderLinearCNDABCSolver`.  The retained production
    nonlinearity is evaluated explicitly in the surface variable,

        N_u(u) = -gamma*u*u_y,  gamma=(3*epsilon/2)*d**(-1/2),

    using the existing second-order directional upwind derivative selected by
    the local nonlinear characteristic speed ``gamma*u``.  Thus the full
    nonlinear scheme remains globally second order even though its linear
    dispersive operator is fourth order in space.  AB2 is used after an
    IMEX-Euler/CN start step.

    The outflow recurrence remains the linear constant-shelf DABC.  It is not
    claimed to be an exact nonlinear transparent boundary; finite-versus-
    extended-domain tests must quantify that approximation before production
    use.
    """

    def __init__(
        self,
        y: np.ndarray,
        depth_ratio: np.ndarray,
        epsilon: float,
        mu: float,
        dt: float,
        n_steps: int,
        *,
        d1_order: int = 6,
        kernel_transform_size: int | None = None,
        shelf_points: int = 8,
        shelf_tolerance: float = 1.0e-13,
    ) -> None:
        if not np.isfinite(epsilon) or epsilon < 0.0:
            raise ValueError("epsilon must be finite and non-negative")
        super().__init__(
            y,
            depth_ratio,
            mu,
            dt,
            n_steps,
            d1_order=d1_order,
            kernel_transform_size=kernel_transform_size,
            shelf_points=shelf_points,
            shelf_tolerance=shelf_tolerance,
        )
        self.epsilon = float(epsilon)
        self.gamma = 1.5 * self.epsilon * self.depth_ratio ** (-0.5)

    def nonlinear(self, normalized: np.ndarray) -> np.ndarray:
        """Return the retained second-order directional-upwind nonlinear drift."""

        values = np.asarray(normalized, dtype=float)
        if values.shape != self.y.shape:
            raise ValueError("nonlinear expects one normalized field matching y")
        surface = self.to_surface(values)
        backward = np.empty_like(surface)
        forward = np.empty_like(surface)
        backward[0] = (surface[1] - surface[0]) / self.dy
        backward[1] = backward[0]
        backward[2:] = (
            3.0 * surface[2:]
            - 4.0 * surface[1:-1]
            + surface[:-2]
        ) / (2.0 * self.dy)
        forward[-1] = (surface[-1] - surface[-2]) / self.dy
        forward[-2] = forward[-1]
        forward[:-2] = (
            -3.0 * surface[:-2]
            + 4.0 * surface[1:-1]
            - surface[2:]
        ) / (2.0 * self.dy)
        characteristic_speed = self.gamma * surface
        derivative = np.where(characteristic_speed >= 0.0, backward, forward)
        surface_drift = -characteristic_speed * derivative
        return self.surface_to_green * surface_drift

    def run(
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
        initial_outflow_relative_tolerance: float = 1.0e-10,
        step_diagnostic: Callable[
            [int, np.ndarray, np.ndarray, np.ndarray], None
        ]
        | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """March the nonlinear equation and return the same arrays as ``run`` above."""

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
        if initial_outflow_relative_tolerance <= 0.0:
            raise ValueError("initial_outflow_relative_tolerance must be positive")

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
                "the homogeneous linear DABC requires zero/compatible exterior "
                "initial data; the rightmost six surface values are not negligible "
                f"(relative tail {tail_ratio:.3e} > "
                f"{initial_outflow_relative_tolerance:.3e})"
            )

        initial_trace_values = tuple(float(trace(0.0)) for trace in boundary_traces)
        for row, trace_value in enumerate(initial_trace_values):
            surface_initial[row] = trace_value
        current = self.to_normalized(surface_initial)
        nonlinear_current = self.nonlinear(current)
        holder = [current, nonlinear_current, nonlinear_current]

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
            previous, nonlinearity, older_nonlinearity = holder
            explicit = (
                nonlinearity
                if step == 1
                else 1.5 * nonlinearity - 0.5 * older_nonlinearity
            )
            rhs = (
                np.asarray(self.right_matrix @ previous).ravel()
                + self.dt * explicit
            )
            time_value = step * self.dt
            trace_values = tuple(float(trace(time_value)) for trace in boundary_traces)
            for row, trace_value in enumerate(trace_values):
                rhs[row] = self.surface_to_green[row] * trace_value

            boundary_rhs = np.empty(3)
            for shift in range(3):
                h1, h2, h3 = histories[3 * shift : 3 * shift + 3]
                boundary_rhs[shift] = h1 - h2 + h3
                rhs[self.n - 1 - shift] = boundary_rhs[shift]

            new_state = self.lu.solve(rhs)
            if not np.all(np.isfinite(new_state)):
                previous_amplitude = float(
                    np.max(np.abs(self.to_surface(previous)))
                )
                raise FloatingPointError(
                    f"non-finite high-order CNAB2 state at step {step}; "
                    f"previous max|u|={previous_amplitude:.6g}"
                )
            if step_diagnostic is not None:
                step_diagnostic(step, previous, new_state, explicit)
            new_nonlinearity = self.nonlinear(new_state)
            holder[0] = new_state
            holder[2] = nonlinearity
            holder[1] = new_nonlinearity
            new_surface = self.to_surface(new_state)

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
