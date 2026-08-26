"""Entropy-C6 nonlinear implicit-midpoint candidate for vKdV screening.

Origin: 3. KDV_nonlinear_case/high_order_implicit_midpoint_candidate.py
Changes vs origin: comments/docstrings only (this provenance header added).

The candidate retains the frozen C6-D1/C4-D3 Crank--Nicolson linear system,
three incident traces and three linear DABC rows.  It replaces explicit AB
nonlinear history by the fully centred step

    (I-dt*L/2) v^(n+1) = (I+dt*L/2) v^n
        + dt*N((v^n+v^(n+1))/2),

where ``N`` is the isolated entropy/split C6 drift.  The nonlinear equation
is solved by fixed-point iteration while reusing the same frozen CN LU at
every iteration.  This module is a screening candidate, not production code.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from high_order_nonlinear_candidates import CoastalHighOrderSplitCNAB2DABCSolver
from transparent_boundary_vkdv import march_convolution_system


class CoastalHighOrderImplicitMidpointDABCSolver(
    CoastalHighOrderSplitCNAB2DABCSolver
):
    """Entropy-C6 drift with a fixed-point implicit midpoint step."""

    candidate_name = "C6_entropy_split_implicit_midpoint"

    def __init__(
        self,
        *args: object,
        fixed_point_relative_tolerance: float = 1.0e-10,
        fixed_point_absolute_tolerance: float = 1.0e-12,
        equation_relative_tolerance: float = 1.0e-11,
        equation_absolute_tolerance: float = 1.0e-12,
        fixed_point_maximum_iterations: int = 12,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        if fixed_point_relative_tolerance <= 0.0:
            raise ValueError("fixed_point_relative_tolerance must be positive")
        if fixed_point_absolute_tolerance <= 0.0:
            raise ValueError("fixed_point_absolute_tolerance must be positive")
        if equation_relative_tolerance <= 0.0:
            raise ValueError("equation_relative_tolerance must be positive")
        if equation_absolute_tolerance <= 0.0:
            raise ValueError("equation_absolute_tolerance must be positive")
        if int(fixed_point_maximum_iterations) < 1:
            raise ValueError("fixed_point_maximum_iterations must be positive")
        self.fixed_point_relative_tolerance = float(
            fixed_point_relative_tolerance
        )
        self.fixed_point_absolute_tolerance = float(
            fixed_point_absolute_tolerance
        )
        self.equation_relative_tolerance = float(
            equation_relative_tolerance
        )
        self.equation_absolute_tolerance = float(
            equation_absolute_tolerance
        )
        self.fixed_point_maximum_iterations = int(
            fixed_point_maximum_iterations
        )
        self.fixed_point_iteration_counts: list[int] = []
        self.maximum_fixed_point_update = 0.0
        self.maximum_fixed_point_scaled_update = 0.0
        self.maximum_equation_residual = 0.0
        self.maximum_scaled_equation_residual = 0.0

    def fixed_point_summary(self) -> dict[str, object]:
        """Return convergence diagnostics from the most recent run."""

        counts = np.asarray(self.fixed_point_iteration_counts, dtype=int)
        return {
            "step_count": int(counts.size),
            "maximum_iterations": int(np.max(counts)) if counts.size else 0,
            "mean_iterations": float(np.mean(counts)) if counts.size else 0.0,
            "p99_iterations": (
                float(np.percentile(counts, 99.0)) if counts.size else 0.0
            ),
            "maximum_fixed_point_update": self.maximum_fixed_point_update,
            "maximum_fixed_point_scaled_update": (
                self.maximum_fixed_point_scaled_update
            ),
            "maximum_equation_residual": self.maximum_equation_residual,
            "maximum_scaled_equation_residual": (
                self.maximum_scaled_equation_residual
            ),
            "relative_tolerance": self.fixed_point_relative_tolerance,
            "absolute_tolerance": self.fixed_point_absolute_tolerance,
            "equation_relative_tolerance": self.equation_relative_tolerance,
            "equation_absolute_tolerance": self.equation_absolute_tolerance,
            "maximum_allowed_iterations": self.fixed_point_maximum_iterations,
            "history_frozen_during_iterations": True,
        }

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
        """March using the nonlinear implicit-midpoint equation."""

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
            time_value = step * self.dt
            trace_values = tuple(float(trace(time_value)) for trace in boundary_traces)
            boundary_rhs = np.empty(3)
            for shift in range(3):
                h1, h2, h3 = histories[3 * shift : 3 * shift + 3]
                boundary_rhs[shift] = h1 - h2 + h3

            def solve_with_nonlinearity(nonlinearity: np.ndarray) -> np.ndarray:
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
                """Return absolute/scaled residual of all bordered rows."""

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

            # Explicit midpoint predictor.  When epsilon=0 this is already
            # the exact frozen linear CN step and is accepted without an
            # unnecessary repeated solve, preserving bitwise degeneration.
            nonlinear_previous = self.nonlinear(previous)
            iterate = solve_with_nonlinearity(nonlinear_previous)
            if not np.all(np.isfinite(iterate)):
                raise FloatingPointError(
                    f"non-finite implicit-midpoint predictor at step {step}"
                )
            if self.epsilon == 0.0:
                iteration_count = 1
                final_nonlinearity = nonlinear_previous
                final_update = 0.0
                final_scaled_update = 0.0
                final_equation_residual, final_scaled_equation_residual = (
                    true_bordered_equation_residual(
                        iterate, final_nonlinearity
                    )
                )
                if (
                    final_equation_residual
                    > self.equation_absolute_tolerance
                    or final_scaled_equation_residual
                    > self.equation_relative_tolerance
                ):
                    raise FloatingPointError(
                        "linear-degeneration bordered residual failed at "
                        f"step {step}: absolute={final_equation_residual:.3e}, "
                        f"scaled={final_scaled_equation_residual:.3e}"
                    )
            else:
                converged = False
                final_update = float("inf")
                final_scaled_update = float("inf")
                final_equation_residual = float("inf")
                final_scaled_equation_residual = float("inf")
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
                            "non-finite implicit-midpoint iterate at "
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
                        "implicit-midpoint fixed point did not converge at "
                        f"step {step} in {self.fixed_point_maximum_iterations} "
                        f"iterations; update={final_update:.3e}, "
                        f"scaled update={final_scaled_update:.3e}, "
                        f"equation residual={final_equation_residual:.3e}, "
                        "scaled equation residual="
                        f"{final_scaled_equation_residual:.3e}, "
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
            if step_diagnostic is not None:
                step_diagnostic(step, previous, new_state, final_nonlinearity)
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
