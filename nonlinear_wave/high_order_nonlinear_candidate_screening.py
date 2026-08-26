"""Experiment 16: non-production screening of nonlinear vKdV candidates.
The frozen Experiment-14/15 files are read-only inputs.  This script screens
the isolated C6 entropy/split nonlinear drift implemented in
``high_order_nonlinear_candidates.py`` while retaining the Experiment-15
linear operator, CNAB2 recurrence, three incident traces and three DABC rows.

Every result produced here is a *screening* result.  Passing a gate does not
authorise replacement of the retained solver.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import scipy

from pde_core import CoastalParameters
from high_order_matched_dabc import fourth_order_derivative_matrices
from high_order_nonlinear_candidates import (
    CoastalHighOrderSplitCNAB2DABCSolver,
    split_entropy_nonlinear_drift,
)
from high_order_variable_depth_dabc import (
    CoastalHighOrderLinearCNDABCSolver,
    coastal_depth_ratio_y,
)
from high_order_variable_depth_dabc_study import (
    StudyConfig as LinearStudyConfig,
    _aligned_steps,
    _grid_for_length,
    _relative_l2,
    _smooth_trace_triplet,
    _tma_inputs,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    PROJECT_DIR
    / "results"
    / "transparent_boundary"
    / "nonlinear_candidate_screening_exp16"
)
RAW_DIR_NAME = "raw_data"


@dataclass(frozen=True)
class ScreeningConfig:
    """Controls for the isolated Experiment-16 screening run."""

    final_time_s: float = 900.0
    analysis_start_s: float = 600.0
    physical_length_m: float = 4000.0
    computational_length_m: float = 6000.0
    medium_n_physical: int = 3073
    medium_dt: float = 0.002
    output_stride: int = 70
    random_seed: int = 20260718

    soliton_length: float = 25.0
    soliton_initial_centre: float = 5.0
    soliton_amplitude: float = 1.0
    soliton_final_time: float = 8.0


def _json_default(value: object) -> object:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"cannot serialise {type(value)!r}")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_hashes(directory: Path, *, exclude: set[str] | None = None) -> dict[str, str]:
    omitted = set() if exclude is None else set(exclude)
    if not directory.exists():
        return {}
    return {
        str(path.relative_to(directory)): _sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name not in omitted
    }


def _observed_order(coarse: float, fine: float) -> float | None:
    if coarse <= 0.0 or fine <= 0.0:
        return None
    return float(np.log(coarse / fine) / np.log(2.0))


def _high_wavenumber_fraction(field: np.ndarray) -> np.ndarray:
    values = np.asarray(field, dtype=float)
    window = np.hanning(values.shape[1])
    tapered = (values - np.mean(values, axis=1, keepdims=True)) * window[None, :]
    power = np.abs(np.fft.rfft(tapered, axis=1)) ** 2
    start = int(np.floor(0.8 * power.shape[1]))
    return np.sum(power[:, start:], axis=1) / np.maximum(
        np.sum(power, axis=1), np.finfo(float).tiny
    )


class ScreeningStepAudit:
    """Track recurrence, CFL and weak-nonlinearity diagnostics."""

    def __init__(
        self,
        solver: CoastalHighOrderSplitCNAB2DABCSolver,
        parameters: CoastalParameters,
    ) -> None:
        self.solver = solver
        self.parameters = parameters
        self.maximum_nonlinear_cfl = 0.0
        self.maximum_interior_recurrence_residual = 0.0
        self.maximum_abs_eta_over_local_depth = 0.0
        self.maximum_nonlinear_to_linear_characteristic_ratio = 0.0
        self.maximum_outflow_nonlinearity_ratio = 0.0
        self.step_count = 0

    def _include_surface(self, surface: np.ndarray) -> None:
        speed = np.abs(self.solver.gamma * surface)
        self.maximum_nonlinear_cfl = max(
            self.maximum_nonlinear_cfl,
            float(self.solver.dt / self.solver.dy * np.max(speed)),
        )
        dimensional_ratio = (
            self.parameters.a_ref_m
            * np.abs(surface)
            / (self.parameters.h_ref_m * self.solver.depth_ratio)
        )
        self.maximum_abs_eta_over_local_depth = max(
            self.maximum_abs_eta_over_local_depth,
            float(np.max(dimensional_ratio)),
        )
        characteristic_ratio = speed / np.maximum(
            self.solver.root_depth, np.finfo(float).tiny
        )
        self.maximum_nonlinear_to_linear_characteristic_ratio = max(
            self.maximum_nonlinear_to_linear_characteristic_ratio,
            float(np.max(characteristic_ratio)),
        )
        self.maximum_outflow_nonlinearity_ratio = max(
            self.maximum_outflow_nonlinearity_ratio,
            float(np.max(characteristic_ratio[-8:])),
        )

    def include_initial(self, normalized: np.ndarray) -> None:
        self._include_surface(self.solver.to_surface(normalized))

    def __call__(
        self,
        step: int,
        previous: np.ndarray,
        new: np.ndarray,
        explicit: np.ndarray,
    ) -> None:
        del step
        midpoint = 0.5 * (previous + new)
        recurrence = (
            new
            - previous
            - self.solver.dt * np.asarray(self.solver.linear @ midpoint).ravel()
            - self.solver.dt * explicit
        )
        self.maximum_interior_recurrence_residual = max(
            self.maximum_interior_recurrence_residual,
            float(np.max(np.abs(recurrence[3:-3]))),
        )
        self._include_surface(self.solver.to_surface(new))
        self.step_count += 1

    def summary(self) -> dict[str, object]:
        return {
            "step_count": self.step_count,
            "maximum_nonlinear_CFL": self.maximum_nonlinear_cfl,
            "maximum_interior_recurrence_residual": (
                self.maximum_interior_recurrence_residual
            ),
            "maximum_abs_eta_over_local_depth": (
                self.maximum_abs_eta_over_local_depth
            ),
            "maximum_nonlinear_to_linear_characteristic_ratio": (
                self.maximum_nonlinear_to_linear_characteristic_ratio
            ),
            "maximum_outflow_nonlinearity_ratio": (
                self.maximum_outflow_nonlinearity_ratio
            ),
        }


def run_manufactured_nonlinear_operator(
    output: Path, parameters: CoastalParameters
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Verify the interior C6 split drift against an analytic smooth field."""

    rows: list[dict[str, object]] = []
    arrays: dict[str, np.ndarray] = {}
    for n in (65, 129, 257, 513):
        y = np.linspace(0.0, 2.0 * np.pi, n)
        dy = float(y[1] - y[0])
        depth = 1.0 + 0.15 * np.sin(y) + 0.05 * np.cos(3.0 * y)
        surface = (
            0.35
            + 0.22 * np.sin(2.0 * y)
            + 0.11 * np.cos(5.0 * y)
            - 0.06 * np.sin(7.0 * y)
        )
        derivative = (
            0.44 * np.cos(2.0 * y)
            - 0.55 * np.sin(5.0 * y)
            - 0.42 * np.cos(7.0 * y)
        )
        gamma = 1.5 * parameters.epsilon * depth ** (-0.5)
        scale = depth**0.25
        d1, _ = fourth_order_derivative_matrices(n, dy, d1_order=6)
        numerical = split_entropy_nonlinear_drift(
            surface, gamma, scale, d1
        )
        exact = scale * (-gamma * surface * derivative)
        interior = slice(3, -3)
        error = numerical[interior] - exact[interior]
        relative_l2 = float(
            np.linalg.norm(error)
            / max(np.linalg.norm(exact[interior]), np.finfo(float).tiny)
        )
        relative_linf = float(
            np.max(np.abs(error))
            / max(np.max(np.abs(exact[interior])), np.finfo(float).tiny)
        )
        rows.append(
            {
                "n": n,
                "dy": dy,
                "relative_L2": relative_l2,
                "relative_Linf": relative_linf,
                "maximum_boundary_row_absolute_drift": float(
                    max(
                        np.max(np.abs(numerical[:3])),
                        np.max(np.abs(numerical[-3:])),
                    )
                ),
            }
        )
        arrays[f"n{n}_y"] = y
        arrays[f"n{n}_numerical"] = numerical
        arrays[f"n{n}_exact"] = exact
    for index in range(1, len(rows)):
        rows[index]["observed_L2_order"] = _observed_order(
            float(rows[index - 1]["relative_L2"]),
            float(rows[index]["relative_L2"]),
        )
        rows[index]["observed_Linf_order"] = _observed_order(
            float(rows[index - 1]["relative_Linf"]),
            float(rows[index]["relative_Linf"]),
        )
    summary = {
        "continuous_target": (
            "S[-gamma/3*(u*u_y+(u^2)_y)] = S[-gamma*u*u_y]"
        ),
        "finest_relative_L2": rows[-1]["relative_L2"],
        "finest_relative_Linf": rows[-1]["relative_Linf"],
        "finest_observed_L2_order": rows[-1]["observed_L2_order"],
        "finest_observed_Linf_order": rows[-1]["observed_Linf_order"],
        "maximum_boundary_row_absolute_drift": max(
            float(row["maximum_boundary_row_absolute_drift"]) for row in rows
        ),
        "rows": rows,
    }
    raw = output / RAW_DIR_NAME
    _write_csv(raw / "manufactured_nonlinear_operator.csv", rows)
    np.savez_compressed(raw / "manufactured_nonlinear_operator.npz", **arrays)
    return summary, arrays


def run_epsilon_zero(
    output: Path,
    parameters: CoastalParameters,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Require bitwise degeneration to the frozen linear solver."""

    y, dy, _ = _grid_for_length(4000.0, parameters, short_n=257)
    depth = coastal_depth_ratio_y(
        y, length_ref_m=parameters.lambda_ref_m, kind="beta_C7"
    )
    dt = 0.004
    n_steps = 256
    traces, _ = _smooth_trace_triplet(
        1.0 / 15.0,
        0.2,
        1.0,
        parameters.mu / 6.0,
        dy,
        dt,
        parameters.time_ref_s,
        ramp_s=20.0,
    )
    linear = CoastalHighOrderLinearCNDABCSolver(
        y, depth, parameters.mu, dt, n_steps
    )
    candidate = CoastalHighOrderSplitCNAB2DABCSolver(
        y, depth, 0.0, parameters.mu, dt, n_steps
    )
    initial = np.zeros_like(y)
    linear_result = linear.run(initial, 16, traces)
    candidate_result = candidate.run(initial, 16, traces)
    names = ("times", "surface", "normalized", "residuals")
    summary: dict[str, object] = {}
    arrays: dict[str, np.ndarray] = {"y": y, "depth_ratio": depth}
    for name, linear_values, candidate_values in zip(
        names, linear_result, candidate_result
    ):
        summary[f"{name}_bitwise_equal"] = bool(
            np.array_equal(linear_values, candidate_values)
        )
        summary[f"{name}_maximum_absolute_difference"] = float(
            np.max(np.abs(linear_values - candidate_values))
        )
        arrays[f"linear_{name}"] = linear_values
        arrays[f"candidate_{name}"] = candidate_values
    raw = output / RAW_DIR_NAME
    _write_csv(raw / "epsilon_zero.csv", [summary])
    np.savez_compressed(raw / "epsilon_zero.npz", **arrays)
    return summary, arrays


def _sech_squared(values: np.ndarray | float) -> np.ndarray:
    argument = np.asarray(values, dtype=float)
    return 1.0 / np.cosh(argument) ** 2


def _parabolic_peak_coordinate(y: np.ndarray, values: np.ndarray) -> float:
    index = int(np.argmax(values))
    if index == 0 or index == values.size - 1:
        return float(y[index])
    left, centre, right = values[index - 1 : index + 2]
    denominator = left - 2.0 * centre + right
    if abs(denominator) <= np.finfo(float).tiny:
        return float(y[index])
    offset = 0.5 * (left - right) / denominator
    return float(y[index] + offset * (y[index + 1] - y[index]))


def run_soliton_screening(
    output: Path,
    parameters: CoastalParameters,
    config: ScreeningConfig,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Converge the candidate to the exact constant-depth KdV soliton."""

    amplitude = config.soliton_amplitude
    gamma = 1.5 * parameters.epsilon
    delta = parameters.mu / 6.0
    kappa = float(np.sqrt(gamma * amplitude / (12.0 * delta)))
    speed = float(1.0 + gamma * amplitude / 3.0)

    def exact(coordinates: np.ndarray | float, time_value: float) -> np.ndarray:
        phase = (
            np.asarray(coordinates, dtype=float)
            - config.soliton_initial_centre
            - speed * time_value
        )
        return amplitude * _sech_squared(kappa * phase)

    rows: list[dict[str, object]] = []
    arrays: dict[str, np.ndarray] = {}
    for label, n, dt in (
        ("coarse", 513, 0.010),
        ("medium", 1025, 0.005),
        ("fine", 2049, 0.0025),
    ):
        y = np.linspace(0.0, config.soliton_length, n)
        n_steps = int(round(config.soliton_final_time / dt))
        solver = CoastalHighOrderSplitCNAB2DABCSolver(
            y,
            np.ones_like(y),
            parameters.epsilon,
            parameters.mu,
            dt,
            n_steps,
        )
        traces = tuple(
            (
                lambda coordinate: (
                    lambda time_value: float(exact(coordinate, time_value))
                )
            )(float(y[index]))
            for index in range(3)
        )
        audit = ScreeningStepAudit(solver, parameters)
        initial = exact(y, 0.0)
        audit.include_initial(solver.to_normalized(initial))
        started = time.perf_counter()
        times, numerical, normalized, residuals = solver.run(
            initial,
            n_steps,
            traces,
            step_diagnostic=audit,
        )
        runtime = time.perf_counter() - started
        del normalized
        final_exact = exact(y, config.soliton_final_time)
        error = numerical[-1] - final_exact
        expected_peak = config.soliton_initial_centre + speed * config.soliton_final_time
        numerical_peak = _parabolic_peak_coordinate(y, numerical[-1])
        row = {
            "level": label,
            "n": n,
            "dy": float(y[1] - y[0]),
            "dt": dt,
            "relative_L2": float(
                np.sqrt(
                    np.trapezoid(error * error, y)
                    / np.trapezoid(final_exact * final_exact, y)
                )
            ),
            "relative_Linf": float(np.max(np.abs(error)) / amplitude),
            "peak_position_error_dimensionless": abs(
                numerical_peak - expected_peak
            ),
            "peak_amplitude_relative_error": abs(
                float(np.max(numerical[-1])) - amplitude
            )
            / amplitude,
            "maximum_six_residual": float(np.max(np.abs(residuals))),
            "runtime_s": runtime,
            **audit.summary(),
        }
        rows.append(row)
        arrays[f"{label}_y"] = y
        arrays[f"{label}_numerical_final"] = numerical[-1]
        arrays[f"{label}_exact_final"] = final_exact
        arrays[f"{label}_error_final"] = error
        arrays[f"{label}_times"] = times
    for index in range(1, len(rows)):
        rows[index]["observed_L2_order"] = _observed_order(
            float(rows[index - 1]["relative_L2"]),
            float(rows[index]["relative_L2"]),
        )
        rows[index]["observed_Linf_order"] = _observed_order(
            float(rows[index - 1]["relative_Linf"]),
            float(rows[index]["relative_Linf"]),
        )
    summary = {
        "exact_solution": (
            "u=A sech^2(kappa(y-y0-C T)); "
            "kappa=sqrt(gamma A/(12 delta)); C=1+gamma A/3"
        ),
        "finest_relative_L2": rows[-1]["relative_L2"],
        "finest_relative_Linf": rows[-1]["relative_Linf"],
        "finest_observed_L2_order": rows[-1]["observed_L2_order"],
        "finest_observed_Linf_order": rows[-1]["observed_Linf_order"],
        "finest_peak_position_error_dimensionless": rows[-1][
            "peak_position_error_dimensionless"
        ],
        "maximum_six_residual": max(
            float(row["maximum_six_residual"]) for row in rows
        ),
        "maximum_nonlinear_CFL": max(
            float(row["maximum_nonlinear_CFL"]) for row in rows
        ),
        "maximum_interior_recurrence_residual": max(
            float(row["maximum_interior_recurrence_residual"]) for row in rows
        ),
        "rows": rows,
        "reference_scope": "analytic continuous constant-depth KdV soliton",
    }
    raw = output / RAW_DIR_NAME
    _write_csv(raw / "soliton_screening.csv", rows)
    np.savez_compressed(raw / "soliton_screening.npz", **arrays)
    return summary, arrays


def _run_coastal_candidate(
    parameters: CoastalParameters,
    config: ScreeningConfig,
    *,
    n_physical: int,
    dt: float,
    n_steps: int,
    output_stride: int,
    traces: tuple[Callable[[float], float], ...],
) -> dict[str, object]:
    y, dy, n = _grid_for_length(
        config.computational_length_m,
        parameters,
        short_n=n_physical,
    )
    depth = coastal_depth_ratio_y(
        y, length_ref_m=parameters.lambda_ref_m, kind="beta_C7"
    )
    solver = CoastalHighOrderSplitCNAB2DABCSolver(
        y,
        depth,
        parameters.epsilon,
        parameters.mu,
        dt,
        n_steps,
    )
    audit = ScreeningStepAudit(solver, parameters)
    initial = np.zeros_like(y)
    audit.include_initial(solver.to_normalized(initial))
    started = time.perf_counter()
    times, surface, normalized, residuals = solver.run(
        initial,
        output_stride,
        traces,
        step_diagnostic=audit,
    )
    runtime = time.perf_counter() - started
    if not np.all(np.isfinite(surface)) or not np.all(np.isfinite(normalized)):
        raise FloatingPointError("candidate returned non-finite coastal fields")
    del normalized
    return {
        "y": y,
        "dy": dy,
        "n": n,
        "depth": depth,
        "times": times,
        "surface": surface,
        "residuals": residuals,
        "runtime_s": runtime,
        "audit": audit.summary(),
    }


def _pair_screening_metrics(
    candidate: np.ndarray,
    reference: np.ndarray,
    times_s: np.ndarray,
    analysis: np.ndarray,
    parameters: CoastalParameters,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    difference = np.asarray(candidate) - np.asarray(reference)
    reference_scale = max(
        float(np.sqrt(np.mean(np.asarray(reference)[analysis] ** 2))),
        np.finfo(float).tiny,
    )
    rms_history = np.sqrt(np.mean(difference * difference, axis=1))
    hs_candidate = 4.0 * np.std(np.asarray(candidate)[analysis], axis=0) * parameters.a_ref_m
    hs_reference = 4.0 * np.std(np.asarray(reference)[analysis], axis=0) * parameters.a_ref_m
    spatial_rms_m = np.sqrt(np.mean(difference[analysis] ** 2, axis=0)) * parameters.a_ref_m

    def first_threshold(threshold: float) -> float | None:
        indices = np.flatnonzero(rms_history / reference_scale > threshold)
        return None if indices.size == 0 else float(times_s[indices[0]])

    metrics = {
        "field_spacetime_relative_L2": _relative_l2(
            np.asarray(candidate)[analysis], np.asarray(reference)[analysis]
        ),
        "Hs_profile_relative_L2": _relative_l2(hs_candidate, hs_reference),
        "maximum_Hs_absolute_difference_m": float(
            np.max(np.abs(hs_candidate - hs_reference))
        ),
        "maximum_normalized_RMS_difference": float(
            np.max(rms_history[analysis]) / reference_scale
        ),
        "first_time_normalized_RMS_exceeds_1_percent_s": first_threshold(0.01),
        "first_time_normalized_RMS_exceeds_5_percent_s": first_threshold(0.05),
    }
    arrays = {
        "normalized_RMS_history": rms_history / reference_scale,
        "Hs_candidate_m": hs_candidate,
        "Hs_reference_m": hs_reference,
        "spatial_RMS_difference_m": spatial_rms_m,
    }
    return metrics, arrays


def run_tma_four_corner_screening(
    output: Path,
    parameters: CoastalParameters,
    config: ScreeningConfig,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Run the 900 s TMA medium/fine four-corner screening matrix."""

    medium_steps = _aligned_steps(
        config.final_time_s, parameters, config.medium_dt
    )
    specifications = {
        "medium": {
            "n_physical": config.medium_n_physical,
            "dt": config.medium_dt,
            "n_steps": medium_steps,
            "stride": config.output_stride,
        },
        "time_fine": {
            "n_physical": config.medium_n_physical,
            "dt": 0.5 * config.medium_dt,
            "n_steps": 2 * medium_steps,
            "stride": 2 * config.output_stride,
        },
        "space_fine": {
            "n_physical": 2 * (config.medium_n_physical - 1) + 1,
            "dt": config.medium_dt,
            "n_steps": medium_steps,
            "stride": config.output_stride,
        },
        "coupled_fine": {
            "n_physical": 2 * (config.medium_n_physical - 1) + 1,
            "dt": 0.5 * config.medium_dt,
            "n_steps": 2 * medium_steps,
            "stride": 2 * config.output_stride,
        },
    }
    linear_config = LinearStudyConfig(random_seed=config.random_seed)
    runs: dict[str, dict[str, object]] = {}
    failures: dict[str, dict[str, object]] = {}
    lifting: dict[str, object] = {}
    for name, specification in specifications.items():
        _, dy, _ = _grid_for_length(
            config.physical_length_m,
            parameters,
            short_n=int(specification["n_physical"]),
        )
        sea_state, _, traces, lifting_metadata = _tma_inputs(
            parameters,
            linear_config,
            dy,
            float(specification["dt"]),
        )
        lifting[name] = lifting_metadata
        print(
            "Experiment 16 screening TMA: "
            f"{name}, N4={specification['n_physical']}, "
            f"dt={specification['dt']}",
            flush=True,
        )
        started = time.perf_counter()
        try:
            runs[name] = _run_coastal_candidate(
                parameters,
                config,
                n_physical=int(specification["n_physical"]),
                dt=float(specification["dt"]),
                n_steps=int(specification["n_steps"]),
                output_stride=int(specification["stride"]),
                traces=traces,
            )
        except Exception as error:  # preserve candidate failure as evidence
            failures[name] = {
                "exception_type": type(error).__name__,
                "message": str(error),
                "elapsed_s": time.perf_counter() - started,
            }

    summary: dict[str, object] = {
        "status": "completed" if not failures else "candidate_failure",
        "failures": failures,
        "specifications": specifications,
        "TMA_sea_state": sea_state.metadata(),
        "lifting": lifting,
        "actual_final_time_s": (
            medium_steps * config.medium_dt * parameters.time_ref_s
        ),
        "analysis_start_s": config.analysis_start_s,
        "computational_domain_m": [0.0, config.computational_length_m],
        "reported_physical_domain_m": [0.0, config.physical_length_m],
    }
    arrays: dict[str, np.ndarray] = {}
    if failures:
        _write_json(output / RAW_DIR_NAME / "tma_candidate_failures.json", failures)
        return summary, arrays

    base_times = np.asarray(runs["medium"]["times"])
    for name, run in runs.items():
        if not np.allclose(
            np.asarray(run["times"]), base_times, rtol=0.0, atol=2.0e-14
        ):
            raise RuntimeError(f"unaligned TMA output times for {name}")
    times_s = base_times * parameters.time_ref_s
    analysis = times_s >= config.analysis_start_s
    n_medium = config.medium_n_physical
    n_fine = 2 * (n_medium - 1) + 1
    medium = np.asarray(runs["medium"]["surface"])[:, :n_medium]
    time_fine = np.asarray(runs["time_fine"]["surface"])[:, :n_medium]
    space_fine_full = np.asarray(runs["space_fine"]["surface"])[:, :n_fine]
    coupled_fine_full = np.asarray(runs["coupled_fine"]["surface"])[:, :n_fine]
    space_fine = space_fine_full[:, ::2]
    coupled_fine = coupled_fine_full[:, ::2]

    pair_fields = {
        "coupled_medium_vs_fine": (medium, coupled_fine),
        "time_refinement_at_medium_space": (medium, time_fine),
        "space_refinement_at_medium_time": (medium, space_fine),
        "space_refinement_at_fine_time": (time_fine, coupled_fine),
    }
    comparisons: dict[str, dict[str, object]] = {}
    comparison_arrays: dict[str, dict[str, np.ndarray]] = {}
    for name, (candidate, reference) in pair_fields.items():
        comparisons[name], comparison_arrays[name] = _pair_screening_metrics(
            candidate,
            reference,
            times_s,
            analysis,
            parameters,
        )
    time_fine_space_metrics, time_fine_space_arrays = _pair_screening_metrics(
        space_fine_full,
        coupled_fine_full,
        times_s,
        analysis,
        parameters,
    )
    comparisons["time_refinement_at_fine_space"] = time_fine_space_metrics
    comparison_arrays["time_refinement_at_fine_space"] = time_fine_space_arrays

    interaction = coupled_fine - time_fine - space_fine + medium
    interaction_relative = float(
        np.linalg.norm(interaction[analysis])
        / max(np.linalg.norm(coupled_fine[analysis]), np.finfo(float).tiny)
    )
    transition_index = int(
        round(3000.0 / (config.physical_length_m / (n_medium - 1)))
    ) + 1
    coupled_metrics = comparisons["coupled_medium_vs_fine"]
    coupled_metrics["offshore_0_3km_field_relative_L2"] = _relative_l2(
        medium[analysis, :transition_index],
        coupled_fine[analysis, :transition_index],
    )
    coupled_metrics["nearshore_3_4km_field_relative_L2"] = _relative_l2(
        medium[analysis, transition_index - 1 :],
        coupled_fine[analysis, transition_index - 1 :],
    )
    coupled_metrics["four_corner_interaction_relative_L2"] = interaction_relative

    medium_high_k = _high_wavenumber_fraction(medium[analysis])
    coupled_high_k = _high_wavenumber_fraction(coupled_fine[analysis])
    gauge_rows: list[dict[str, object]] = []
    spacing_m = config.physical_length_m / (n_medium - 1)
    for location_m in (0.0, 500.0, 1500.0, 2500.0, 3500.0, 4000.0):
        index = int(round(location_m / spacing_m))
        gauge_rows.append(
            {
                "location_m": location_m,
                "medium_vs_coupled_relative_L2": _relative_l2(
                    medium[analysis, index], coupled_fine[analysis, index]
                ),
                "time_only_relative_L2": _relative_l2(
                    medium[analysis, index], time_fine[analysis, index]
                ),
                "space_only_relative_L2": _relative_l2(
                    medium[analysis, index], space_fine[analysis, index]
                ),
            }
        )

    maximum_residual = max(
        float(np.max(np.abs(run["residuals"]))) for run in runs.values()
    )
    maximum_cfl = max(
        float(run["audit"]["maximum_nonlinear_CFL"]) for run in runs.values()
    )
    maximum_recurrence = max(
        float(run["audit"]["maximum_interior_recurrence_residual"])
        for run in runs.values()
    )
    summary.update(
        {
            "comparisons": comparisons,
            "gauge_comparisons": gauge_rows,
            "maximum_windowed_highest_20pct_wavenumber_fraction_medium": float(
                np.max(medium_high_k)
            ),
            "maximum_windowed_highest_20pct_wavenumber_fraction_coupled_fine": float(
                np.max(coupled_high_k)
            ),
            "maximum_six_residual": maximum_residual,
            "maximum_nonlinear_CFL": maximum_cfl,
            "maximum_interior_recurrence_residual": maximum_recurrence,
            "run_audits": {name: run["audit"] for name, run in runs.items()},
            "run_runtimes_s": {
                name: float(run["runtime_s"]) for name, run in runs.items()
            },
            "reference_scope": (
                "four-corner self-refinement; coupled fine is not exact truth; "
                "each corner uses its discretisation-consistent modal lifting"
            ),
        }
    )
    y_medium_m = np.asarray(runs["medium"]["y"])[:n_medium] * parameters.lambda_ref_m
    arrays.update(
        {
            "times_s": times_s,
            "analysis_mask": analysis,
            "y_medium_m": y_medium_m,
            "depth_medium_m": np.asarray(runs["medium"]["depth"])[:n_medium]
            * parameters.h_ref_m,
            "eta_medium_m": medium * parameters.a_ref_m,
            "eta_time_fine_m": time_fine * parameters.a_ref_m,
            "eta_space_fine_on_medium_m": space_fine * parameters.a_ref_m,
            "eta_coupled_fine_on_medium_m": coupled_fine * parameters.a_ref_m,
            "eta_four_corner_interaction_m": interaction * parameters.a_ref_m,
            "medium_high_k_fraction": medium_high_k,
            "coupled_fine_high_k_fraction": coupled_high_k,
        }
    )
    for name, values in comparison_arrays.items():
        for key, array in values.items():
            arrays[f"{name}_{key}"] = array
    raw = output / RAW_DIR_NAME
    np.savez_compressed(raw / "tma_four_corner_screening.npz", **arrays)
    _write_csv(raw / "tma_gauge_screening.csv", gauge_rows)
    _write_csv(
        raw / "tma_four_corner_metrics.csv",
        [
            {"comparison": name, **metrics}
            for name, metrics in comparisons.items()
        ],
    )
    return summary, arrays


def _load_frozen_u2_baseline(exp15_directory: Path) -> dict[str, object]:
    metrics_path = exp15_directory / "metrics.json"
    if not metrics_path.exists():
        return {"status": "missing", "path": str(metrics_path)}
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    values = metrics["TMA_medium_fine"]
    return {
        "status": "loaded_read_only",
        "metrics_path": str(metrics_path),
        "metrics_sha256": _sha256(metrics_path),
        "field_spacetime_relative_L2": values[
            "medium_fine_spacetime_relative_L2"
        ],
        "Hs_profile_relative_L2": values[
            "medium_fine_Hs_profile_relative_L2"
        ],
        "offshore_0_3km_field_relative_L2": values[
            "offshore_0_3km_spacetime_relative_L2"
        ],
        "nearshore_3_4km_field_relative_L2": values[
            "nearshore_3_4km_spacetime_relative_L2"
        ],
    }


def _report_text(metrics: dict[str, object]) -> str:
    manufactured = metrics["manufactured_operator"]
    epsilon_zero = metrics["epsilon_zero"]
    soliton = metrics["soliton"]
    tma = metrics["TMA_four_corner"]
    baseline = metrics["frozen_U2_baseline"]
    gates = metrics["screening_gates"]
    lines = [
        "# Experiment 16: C6 split nonlinear candidate screening (not production)",
        "",
        "## Positioning",
        "",
        "This experiment only screens candidate schemes; it does not replace the",
        "frozen Exp14/Exp15 solvers. The candidate is",
        "`Nv=S[-gamma/3*(u D1u + D1(u^2))]`, active on interior rows only; the",
        "three inflow traces and the three linear DABC rows still own their",
        "equations. The time integrator remains the same CNAB2.",
        "",
        "## Manufactured operator and linear degeneration",
        "",
        f"- Finest manufactured-operator relative L2: `{manufactured['finest_relative_L2']:.3e}`.",
        f"- Finest observed L2 order: `{manufactured['finest_observed_L2_order']:.3f}`.",
        f"- Maximum nonlinear drift on the six boundary rows: `{manufactured['maximum_boundary_row_absolute_drift']:.3e}`.",
        f"- epsilon=0 surface bitwise equal to linear: `{epsilon_zero['surface_bitwise_equal']}`.",
        "",
        "## Constant-depth analytic soliton",
        "",
        f"- Finest relative L2/Linf: `{soliton['finest_relative_L2']:.3e}` / `{soliton['finest_relative_Linf']:.3e}`.",
        f"- Observed L2/Linf orders: `{soliton['finest_observed_L2_order']:.3f}` / `{soliton['finest_observed_Linf_order']:.3f}`.",
        "",
        "## 900 s TMA four-corner screening",
        "",
    ]
    if tma["status"] != "completed":
        lines.extend(
            [
                "The candidate failed in one or more TMA cases; the failure is "
                "kept as-is and no spurious convergence conclusion is generated.",
                "",
                "```json",
                json.dumps(tma["failures"], indent=2, ensure_ascii=False),
                "```",
                "",
            ]
        )
    else:
        coupled = tma["comparisons"]["coupled_medium_vs_fine"]
        time_only = tma["comparisons"]["time_refinement_at_medium_space"]
        space_only = tma["comparisons"]["space_refinement_at_medium_time"]
        lines.extend(
            [
                f"- Coupled medium/fine field: `{coupled['field_spacetime_relative_L2']:.3e}`.",
                f"- Coupled medium/fine Hs: `{coupled['Hs_profile_relative_L2']:.3e}`.",
                f"- Time-only (medium space) field: `{time_only['field_spacetime_relative_L2']:.3e}`.",
                f"- Space-only (medium time) field: `{space_only['field_spacetime_relative_L2']:.3e}`.",
                f"- 0--3 / 3--4 km field: `{coupled['offshore_0_3km_field_relative_L2']:.3e}` / `{coupled['nearshore_3_4km_field_relative_L2']:.3e}`.",
                f"- Frozen U2 field/Hs: `{baseline.get('field_spacetime_relative_L2', float('nan')):.3e}` / `{baseline.get('Hs_profile_relative_L2', float('nan')):.3e}`.",
                "",
                "The four-corner differences locate temporal and spatial",
                "sensitivity; they are not a three-level Richardson order, and each",
                "corner uses the linear modal lifting consistent with its own",
                "dx/dt, so the effect of the inflow discretisation on the whole",
                "algorithm is included.",
                "",
            ]
        )
    lines.extend(["## Screening gates", ""])
    for name, value in gates.items():
        state = "NOT RUN" if value is None else ("PASS" if value else "FAIL")
        lines.append(f"- {state}: `{name}`")
    lines.extend(
        [
            "",
            "## Strict limitations",
            "",
            "- The coupled fine grid remains a same-equation self-convergence "
            "reference, not exact truth.",
            "- The linear DABC is not an exact nonlinear transparent boundary; "
            "this screening does not replace an independent pulse-exit "
            "reflection test.",
            "- The split scheme has no upwind dissipation or shock capturing; if "
            "a sea state develops overly steep gradients, stability must be "
            "reviewed separately.",
            "- The conclusions cover only the C7 bathymetry, the 0.03--0.105 Hz "
            "TMA band, Tp=15 s, 900 s and the current amplitude.",
            "- The U5 candidate was not run in this round; its performance "
            "cannot be inferred from these results.",
            "- Every gate here is a screening gate, not a production approval.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    raw = output / RAW_DIR_NAME
    raw.mkdir(parents=True, exist_ok=True)

    config = ScreeningConfig()
    parameters = CoastalParameters()
    exp14_directory = (
        PROJECT_DIR
        / "results"
        / "transparent_boundary"
        / "high_order_variable_depth_dabc"
    )
    exp15_directory = (
        PROJECT_DIR
        / "results"
        / "transparent_boundary"
        / "high_order_nonlinear_vkdv"
    )
    frozen_sources = (
        PROJECT_DIR / "transparent_boundary_vkdv.py",
        PROJECT_DIR / "pde_core.py",
        PROJECT_DIR / "high_order_matched_dabc.py",
        PROJECT_DIR / "high_order_incident_lifting.py",
        PROJECT_DIR / "high_order_variable_depth_dabc.py",
        PROJECT_DIR / "high_order_variable_depth_dabc_study.py",
        PROJECT_DIR / "high_order_nonlinear_vkdv_study.py",
    )
    all_sources = (
        PROJECT_DIR / "high_order_nonlinear_candidates.py",
        Path(__file__).resolve(),
        *frozen_sources,
    )
    source_hashes_before = {str(path): _sha256(path) for path in all_sources}
    exp14_hashes_before = _tree_hashes(exp14_directory)
    exp15_hashes_before = _tree_hashes(exp15_directory)
    started = time.perf_counter()

    manufactured, manufactured_arrays = run_manufactured_nonlinear_operator(
        output, parameters
    )
    epsilon_zero, epsilon_arrays = run_epsilon_zero(output, parameters)
    soliton, soliton_arrays = run_soliton_screening(output, parameters, config)
    frozen_baseline = _load_frozen_u2_baseline(exp15_directory)
    tma, tma_arrays = run_tma_four_corner_screening(output, parameters, config)

    source_hashes_after = {str(path): _sha256(path) for path in all_sources}
    exp14_hashes_after = _tree_hashes(exp14_directory)
    exp15_hashes_after = _tree_hashes(exp15_directory)
    frozen_unchanged = bool(
        source_hashes_before == source_hashes_after
        and exp14_hashes_before == exp14_hashes_after
        and exp15_hashes_before == exp15_hashes_after
    )

    tma_completed = tma["status"] == "completed"
    coupled = (
        tma.get("comparisons", {}).get("coupled_medium_vs_fine", {})
        if tma_completed
        else {}
    )
    gates: dict[str, bool | None] = {
        "manufactured_C6_L2_order_above_5p5": bool(
            float(manufactured["finest_observed_L2_order"]) > 5.5
        ),
        "manufactured_finest_relative_L2_below_2e-8": bool(
            float(manufactured["finest_relative_L2"]) < 2.0e-8
        ),
        "nonlinear_boundary_rows_exactly_zero": bool(
            float(manufactured["maximum_boundary_row_absolute_drift"]) == 0.0
        ),
        "epsilon_zero_surface_bitwise_linear": bool(
            epsilon_zero["surface_bitwise_equal"]
        ),
        "epsilon_zero_normalized_bitwise_linear": bool(
            epsilon_zero["normalized_bitwise_equal"]
        ),
        "soliton_finest_relative_L2_below_1e-5": bool(
            float(soliton["finest_relative_L2"]) < 1.0e-5
        ),
        "soliton_L2_order_between_1p8_and_2p3": bool(
            1.8 < float(soliton["finest_observed_L2_order"]) < 2.3
        ),
        "soliton_six_residual_below_1e-10": bool(
            float(soliton["maximum_six_residual"]) < 1.0e-10
        ),
        "TMA_all_four_corners_completed_and_finite": tma_completed,
        "TMA_coupled_field_below_1_percent": (
            bool(float(coupled["field_spacetime_relative_L2"]) < 0.01)
            if tma_completed
            else False
        ),
        "TMA_coupled_Hs_below_0p5_percent": (
            bool(float(coupled["Hs_profile_relative_L2"]) < 0.005)
            if tma_completed
            else False
        ),
        "TMA_six_residual_below_1e-10": (
            bool(float(tma["maximum_six_residual"]) < 1.0e-10)
            if tma_completed
            else False
        ),
        "TMA_interior_recurrence_residual_below_1e-10": (
            bool(float(tma["maximum_interior_recurrence_residual"]) < 1.0e-10)
            if tma_completed
            else False
        ),
        "TMA_nonlinear_CFL_below_0p2": (
            bool(float(tma["maximum_nonlinear_CFL"]) < 0.2)
            if tma_completed
            else False
        ),
        "frozen_Exp14_Exp15_sources_and_outputs_unchanged": frozen_unchanged,
        "isolated_nonlinear_pulse_exit_reflection_gate": None,
        "U5_candidate_gate": None,
    }
    all_executed = bool(all(value for value in gates.values() if value is not None))
    metrics: dict[str, object] = {
        "experiment": "Experiment 16 nonlinear candidate screening",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "screening_only": True,
        "candidate": {
            "name": CoastalHighOrderSplitCNAB2DABCSolver.candidate_name,
            "formula": "Nv=S[-gamma/3*(u D1u + D1(u^2))]",
            "interior_rows": "3:-3",
            "time_integrator": "same CNAB2 as frozen Experiment 15",
            "boundary_rows": "same 3 prescribed traces and 3 linear DABC rows",
        },
        "parameters": asdict(parameters),
        "config": asdict(config),
        "manufactured_operator": manufactured,
        "epsilon_zero": epsilon_zero,
        "soliton": soliton,
        "TMA_four_corner": tma,
        "frozen_U2_baseline": frozen_baseline,
        "screening_gates": gates,
        "all_executed_screening_gates_passed": all_executed,
        "runtime_s": time.perf_counter() - started,
        "provenance": {
            "source_hashes_before": source_hashes_before,
            "source_hashes_after": source_hashes_after,
            "Exp14_hashes_before": exp14_hashes_before,
            "Exp14_hashes_after": exp14_hashes_after,
            "Exp15_hashes_before": exp15_hashes_before,
            "Exp15_hashes_after": exp15_hashes_after,
            "frozen_inputs_unchanged": frozen_unchanged,
        },
    }
    _write_json(output / "metrics.json", metrics)
    _write_csv(
        raw / "screening_gates.csv",
        [
            {
                "gate": name,
                "status": "NOT_RUN" if value is None else ("PASS" if value else "FAIL"),
            }
            for name, value in gates.items()
        ],
    )
    report = output / "experiment_16_candidate_screening_report.md"
    report.write_text(_report_text(metrics), encoding="utf-8")

    artifacts = _tree_hashes(output, exclude={"manifest.json"})
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": Path(__file__).name,
        "screening_only": True,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "source_hashes": source_hashes_after,
        "frozen_Exp14_hashes": exp14_hashes_after,
        "frozen_Exp15_hashes": exp15_hashes_after,
        "Experiment16_artifact_hashes": artifacts,
        "manifest_self_hash": "excluded by construction",
        "outputs": {
            "metrics": str(output / "metrics.json"),
            "report": str(report),
            "raw_data": str(raw),
        },
    }
    _write_json(output / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "output": str(output),
                "all_executed_screening_gates_passed": all_executed,
                "runtime_s": metrics["runtime_s"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
