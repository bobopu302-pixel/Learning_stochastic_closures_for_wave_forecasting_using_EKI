"""Experiment 14: validate the isolated high-order linear variable-depth solver.

Origin: 3. KDV_nonlinear_case/high_order_variable_depth_dabc_study.py
Changes vs origin (numerics untouched):
* matplotlib imports, the colour constants and the three figure
  functions (_plot_operator_summary, _plot_extended_summary,
  _plot_refinement) deleted, together with their call sites and the
  'figures'/matplotlib entries in metrics.json and manifest.json
  (release ships no plotting); all data saving is preserved;
* the Chinese Markdown report emitted by _write_report translated to
  English (same numbers and structure).

The default study is deliberately staged.  It first checks the variable-depth
operator against an analytic manufactured reference and verifies exact
constant-depth degeneration.  Only then does it run the 4 km candidate against
an 8 km same-scheme extension for a single carrier and the deterministic TMA
record.  The extension appends a constant 5 m shelf.  It is a domain-extension
invariance check, not an independent PDE truth.  A deliberately 1% perturbed
DABC kernel is included as a positive control for the comparison metric.

Optional ``--full-refinement`` runs coupled grid/time and fixed-grid time
refinement.  It is kept out of the default gate because it approximately
doubles the PDE work.

This release generates no figures: all evidence lands in ``metrics.json``,
the raw ``.npz``/``.csv`` files and the Markdown report, from which any chart
can be re-plotted.

No production, Experiment-12 or Experiment-13 source is modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import scipy
from scipy.sparse.linalg import splu

from pde_core import CoastalParameters
from high_order_incident_lifting import (
    ModalThreeTraceLifting,
    PeriodicBoundarySpectrum,
    build_modal_three_trace_lifting,
    discrete_long_branch_phase_increments,
    half_cosine_ramp,
)
from high_order_matched_dabc import (
    FourthOrderCNDABCSolver,
    fourth_order_cn_group_velocity,
    fourth_order_derivative_matrices,
)
from high_order_variable_depth_dabc import (
    CoastalHighOrderLinearCNDABCSolver,
    assemble_normalized_linear_operator,
    coastal_depth_ratio_y,
)
from sea_state_boundary import SeaStateParameters, TMASeaState


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    PROJECT_DIR
    / "results"
    / "transparent_boundary"
    / "high_order_variable_depth_dabc"
)
RAW_DIR_NAME = "raw_data"


@dataclass(frozen=True)
class StudyConfig:
    """Experiment-14 physical and numerical controls."""

    final_time_s: float = 900.0
    analysis_start_s: float = 600.0
    single_frequency_hz: float = 0.085
    short_length_m: float = 4000.0
    extended_length_m: float = 8000.0
    medium_n_short: int = 3073
    medium_dt: float = 0.002
    output_stride: int = 70
    constant_identity_length_m: float = 1000.0
    constant_identity_n: int = 257
    constant_identity_dt: float = 0.004
    constant_identity_duration_s: float = 120.0
    refinement_final_dimensionless: float = 56.0
    refinement_analysis_start_s: float = 500.0
    refinement_frequency_hz: float = 1.0 / 15.0
    random_seed: int = 20260718


def _json_default(value: object) -> object:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialise {type(value)!r}")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
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


def _relative_l2(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    denominator = max(float(np.linalg.norm(b.ravel())), np.finfo(float).tiny)
    return float(np.linalg.norm((a - b).ravel()) / denominator)


def _sparse_relative_frobenius(first: object, second: object) -> float:
    difference = first - second
    numerator = float(np.sqrt(np.sum(np.abs(difference.data) ** 2)))
    denominator = max(
        float(np.sqrt(np.sum(np.abs(second.data) ** 2))), np.finfo(float).tiny
    )
    return numerator / denominator


def _observed_order(coarse: float, fine: float) -> float:
    if coarse <= 0.0 or fine <= 0.0:
        return float("nan")
    return float(np.log(coarse / fine) / np.log(2.0))


def _maximum_kernel_relative_difference(
    first: Iterable[np.ndarray], second: Iterable[np.ndarray]
) -> float:
    result = 0.0
    for a, b in zip(first, second, strict=True):
        scale = max(float(np.linalg.norm(np.asarray(b))), np.finfo(float).tiny)
        result = max(result, float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / scale))
    return result


def _grid_for_length(
    length_m: float, p: CoastalParameters, *, short_n: int
) -> tuple[np.ndarray, float, int]:
    short_intervals = int(short_n) - 1
    dy = (4000.0 / p.lambda_ref_m) / short_intervals
    intervals = int(round((length_m / p.lambda_ref_m) / dy))
    if not np.isclose(intervals * dy, length_m / p.lambda_ref_m, atol=1.0e-13):
        raise ValueError("requested domain is not nested at the selected spacing")
    return np.arange(intervals + 1, dtype=float) * dy, dy, intervals + 1


def _aligned_steps(seconds: float, p: CoastalParameters, dt: float) -> int:
    return int(round(seconds / p.time_ref_s / dt))


def _smooth_trace_triplet(
    frequency_hz: float,
    amplitude: float,
    advection: float,
    dispersion: float,
    dy: float,
    dt: float,
    time_ref_s: float,
    ramp_s: float = 20.0,
) -> tuple[tuple[Callable[[float], float], ...], dict[str, float]]:
    omega = np.array((2.0 * np.pi * frequency_hz * time_ref_s,))
    theta, metadata = discrete_long_branch_phase_increments(
        omega, advection, dispersion, dy, dt, d1_order=6
    )
    group_velocity = float(
        fourth_order_cn_group_velocity(
            theta[0] / dy,
            advection,
            dispersion,
            dy,
            dt,
            d1_order=6,
        )
    )
    ramp_duration = ramp_s / time_ref_s
    group_delay = dy / group_velocity

    def make(offset: int) -> Callable[[float], float]:
        def trace(time_value: float) -> float:
            delayed = time_value - offset * group_delay
            return float(
                amplitude
                * half_cosine_ramp(delayed, ramp_duration)
                * np.cos(omega[0] * time_value - offset * theta[0] + 0.37)
            )

        return trace

    return (
        (make(0), make(1), make(2)),
        {
            "omega_dimensionless": float(omega[0]),
            "theta_rad": float(theta[0]),
            "group_velocity_dimensionless": group_velocity,
            "ramp_duration_dimensionless": ramp_duration,
            **{str(key): float(value) for key, value in metadata.items()},
        },
    )


def analytic_manufactured_operator(
    output: Path, p: CoastalParameters
) -> tuple[list[dict[str, object]], dict[str, float], dict[str, np.ndarray]]:
    """Compare the coded variable operator with a closed-form continuum field."""

    rows: list[dict[str, object]] = []
    arrays: dict[str, np.ndarray] = {}
    length = 2.0 * np.pi
    previous_l2: float | None = None
    previous_linf: float | None = None
    for n in (129, 257, 513, 1025):
        y = np.linspace(0.0, length, n)
        dy = float(y[1] - y[0])
        kappa = 2.0 * np.pi / length
        s = kappa * y
        depth = 0.70 + 0.12 * np.cos(s) + 0.06 * np.sin(2.0 * s)
        surface = (
            0.30 * np.sin(2.0 * s)
            + 0.15 * np.cos(5.0 * s)
            + 0.05 * np.sin(7.0 * s)
        )
        normalized = depth**0.25 * surface
        depth_y = kappa * (-0.12 * np.sin(s) + 0.12 * np.cos(2.0 * s))
        root_depth = np.sqrt(depth)
        root_depth_y = 0.5 * depth_y / root_depth
        surface_y = kappa * (
            0.60 * np.cos(2.0 * s)
            - 0.75 * np.sin(5.0 * s)
            + 0.35 * np.cos(7.0 * s)
        )
        surface_yyy = (
            -0.30 * (2.0 * kappa) ** 3 * np.cos(2.0 * s)
            + 0.15 * (5.0 * kappa) ** 3 * np.sin(5.0 * s)
            - 0.05 * (7.0 * kappa) ** 3 * np.cos(7.0 * s)
        )
        delta = (p.mu / 6.0) * depth**2.5
        exact = depth**0.25 * (
            -root_depth * surface_y
            - 0.5 * root_depth_y * surface
            - delta * surface_yyy
        )
        d1, d3 = fourth_order_derivative_matrices(n, dy, d1_order=6)
        operator, _ = assemble_normalized_linear_operator(depth, p.mu, d1, d3)
        numerical = np.asarray(operator @ normalized).ravel()
        mask = np.zeros(n, dtype=bool)
        mask[3:-3] = True
        error = numerical[mask] - exact[mask]
        relative_l2 = float(np.linalg.norm(error) / np.linalg.norm(exact[mask]))
        relative_linf = float(
            np.max(np.abs(error)) / np.max(np.abs(exact[mask]))
        )
        row = {
            "n": n,
            "dy": dy,
            "relative_L2_error": relative_l2,
            "relative_Linf_error": relative_linf,
            "observed_L2_order": (
                None
                if previous_l2 is None
                else _observed_order(previous_l2, relative_l2)
            ),
            "observed_Linf_order": (
                None
                if previous_linf is None
                else _observed_order(previous_linf, relative_linf)
            ),
            "included_node_count": int(np.count_nonzero(mask)),
        }
        rows.append(row)
        previous_l2, previous_linf = relative_l2, relative_linf
        if n == 1025:
            arrays = {
                "y": y,
                "depth_ratio": depth,
                "normalized_field": normalized,
                "exact_operator": exact,
                "numerical_operator": numerical,
                "interior_mask": mask,
            }
    _write_csv(output / RAW_DIR_NAME / "manufactured_operator_convergence.csv", rows)
    np.savez_compressed(
        output / RAW_DIR_NAME / "manufactured_operator_finest.npz", **arrays
    )
    summary = {
        "finest_relative_L2_error": float(rows[-1]["relative_L2_error"]),
        "finest_relative_Linf_error": float(rows[-1]["relative_Linf_error"]),
        "finest_L2_order": float(rows[-1]["observed_L2_order"]),
        "finest_Linf_order": float(rows[-1]["observed_Linf_order"]),
    }
    return rows, summary, arrays


def bathymetry_regularity_audit(
    output: Path, p: CoastalParameters
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, np.ndarray]]:
    """Nested-grid self-convergence including both shelf--slope junctions."""

    rows: list[dict[str, object]] = []
    profile_arrays: dict[str, np.ndarray] = {}
    for kind in ("cubic_C1", "septic_C3", "beta_C7"):
        outputs: list[tuple[np.ndarray, np.ndarray]] = []
        for n in (193, 385, 769, 1537):
            y = np.linspace(0.0, p.domain_m / p.lambda_ref_m, n)
            dy = float(y[1] - y[0])
            depth = coastal_depth_ratio_y(
                y, length_ref_m=p.lambda_ref_m, kind=kind
            )
            normalized = (
                np.sin(2.0 * np.pi * y / y[-1])
                + 0.2 * np.cos(3.0 * np.pi * y / y[-1])
                + 0.1 * np.sin(5.0 * np.pi * y / y[-1])
            )
            d1, d3 = fourth_order_derivative_matrices(n, dy, d1_order=6)
            operator, _ = assemble_normalized_linear_operator(depth, p.mu, d1, d3)
            outputs.append((y, np.asarray(operator @ normalized).ravel()))
            if n == 1537:
                profile_arrays[f"y_{kind}"] = y
                profile_arrays[f"depth_{kind}"] = depth

        previous_rms: float | None = None
        previous_linf: float | None = None
        for level in range(len(outputs) - 1):
            y, coarse = outputs[level]
            fine = outputs[level + 1][1][::2]
            mask = np.zeros(y.size, dtype=bool)
            mask[3:-3] = True
            difference = coarse[mask] - fine[mask]
            rms = float(np.sqrt(np.mean(difference**2)))
            linf = float(np.max(np.abs(difference)))
            rows.append(
                {
                    "bathymetry": kind,
                    "coarse_n": int(y.size),
                    "fine_n": int(outputs[level + 1][0].size),
                    "coarse_dy": float(y[1] - y[0]),
                    "RMS_self_difference": rms,
                    "Linf_self_difference": linf,
                    "RMS_observed_order": (
                        None
                        if previous_rms is None
                        else _observed_order(previous_rms, rms)
                    ),
                    "Linf_observed_order": (
                        None
                        if previous_linf is None
                        else _observed_order(previous_linf, linf)
                    ),
                    "junctions_included": True,
                }
            )
            previous_rms, previous_linf = rms, linf

    _write_csv(output / RAW_DIR_NAME / "bathymetry_regularity.csv", rows)
    np.savez_compressed(
        output / RAW_DIR_NAME / "bathymetry_profiles.npz", **profile_arrays
    )
    by_kind = {
        kind: [row for row in rows if row["bathymetry"] == kind]
        for kind in ("cubic_C1", "septic_C3", "beta_C7")
    }
    c1_finest = float(by_kind["cubic_C1"][-1]["RMS_self_difference"])
    c7_finest = float(by_kind["beta_C7"][-1]["RMS_self_difference"])
    # Physical-shape diagnostics are deliberately not acceptance gates.  A
    # higher-order compact transition changes the modelled bathymetry as well
    # as its differentiability, so its slope and shallow-water travel time
    # must be disclosed rather than treated as a purely numerical change.
    dense_y = np.linspace(0.0, p.domain_m / p.lambda_ref_m, 200001)
    dense_y_m = dense_y * p.lambda_ref_m
    physical_shape: dict[str, dict[str, float]] = {}
    for kind in ("cubic_C1", "septic_C3", "beta_C7"):
        dense_depth_m = p.h_ref_m * coastal_depth_ratio_y(
            dense_y, length_ref_m=p.lambda_ref_m, kind=kind
        )
        depth_slope = np.gradient(dense_depth_m, dense_y_m, edge_order=2)
        travel_time_s = float(
            np.trapezoid(1.0 / np.sqrt(p.gravity * dense_depth_m), dense_y_m)
        )
        physical_shape[kind] = {
            "maximum_absolute_bed_slope": float(np.max(np.abs(depth_slope))),
            "maximum_lambda_ref_abs_hx_over_h": float(
                np.max(p.lambda_ref_m * np.abs(depth_slope) / dense_depth_m)
            ),
            "shallow_water_travel_time_s": travel_time_s,
        }
    cubic_travel = physical_shape["cubic_C1"]["shallow_water_travel_time_s"]
    for values in physical_shape.values():
        values["travel_time_relative_to_cubic"] = (
            values["shallow_water_travel_time_s"] / cubic_travel - 1.0
        )

    summary: dict[str, object] = {
        "profiles": {
            kind: {
                "finest_RMS_self_difference": float(values[-1]["RMS_self_difference"]),
                "finest_Linf_self_difference": float(values[-1]["Linf_self_difference"]),
                "finest_RMS_order": float(values[-1]["RMS_observed_order"]),
                "finest_Linf_order": float(values[-1]["Linf_observed_order"]),
            }
            for kind, values in by_kind.items()
        },
        "C1_to_C7_finest_RMS_ratio": c1_finest / max(c7_finest, np.finfo(float).tiny),
        "physical_shape_diagnostics": physical_shape,
    }
    return rows, summary, profile_arrays


def constant_depth_identity(
    output: Path, p: CoastalParameters, config: StudyConfig
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, np.ndarray]]:
    """Verify surface-level identity with the Experiment-12 constant solver."""

    rows: list[dict[str, object]] = []
    arrays: dict[str, np.ndarray] = {}
    y = np.linspace(
        0.0,
        config.constant_identity_length_m / p.lambda_ref_m,
        config.constant_identity_n,
    )
    dy = float(y[1] - y[0])
    n_steps = _aligned_steps(config.constant_identity_duration_s, p, config.constant_identity_dt)
    for depth_m in (15.0, 5.0):
        depth = depth_m / p.h_ref_m
        advection = float(np.sqrt(depth))
        dispersion = float((p.mu / 6.0) * depth**2.5)
        traces, trace_metadata = _smooth_trace_triplet(
            config.single_frequency_hz,
            0.1,
            advection,
            dispersion,
            dy,
            config.constant_identity_dt,
            p.time_ref_s,
        )
        variable = CoastalHighOrderLinearCNDABCSolver(
            y,
            np.full_like(y, depth),
            p.mu,
            config.constant_identity_dt,
            n_steps,
        )
        reference = FourthOrderCNDABCSolver(
            y,
            advection,
            dispersion,
            config.constant_identity_dt,
            n_steps,
            d1_order=6,
        )
        initial = np.zeros_like(y)
        start = time.perf_counter()
        times, surface, normalized, residuals = variable.run(initial, 1, traces)
        reference_times, reference_surface, reference_residuals = reference.run(
            initial, 1, traces
        )
        runtime_s = time.perf_counter() - start
        if not np.array_equal(times, reference_times):
            raise RuntimeError("constant-depth identity produced different time grids")
        kernel_difference = _maximum_kernel_relative_difference(
            (
                variable.kernels.root_sum,
                variable.kernels.root_pair_sum,
                variable.kernels.root_product,
            ),
            (
                reference.kernels.root_sum,
                reference.kernels.root_pair_sum,
                reference.kernels.root_product,
            ),
        )
        field_difference = _relative_l2(surface, reference_surface)
        matrix_difference = _sparse_relative_frobenius(
            variable.linear, reference.linear
        )

        # Exact algebraic CN work ledger.  The correction term contains only
        # rows replaced by the three incident and three DABC constraints.
        delta_energy: list[float] = []
        linear_work: list[float] = []
        constraint_work: list[float] = []
        closure: list[float] = []
        for index in range(normalized.shape[0] - 1):
            previous = normalized[index]
            current = normalized[index + 1]
            midpoint = 0.5 * (previous + current)
            increment = current - previous
            linear_increment = config.constant_identity_dt * np.asarray(
                variable.linear @ midpoint
            ).ravel()
            correction = increment - linear_increment
            de = 0.5 * dy * (
                float(np.dot(current, current)) - float(np.dot(previous, previous))
            )
            lw = dy * float(np.dot(midpoint, linear_increment))
            cw = dy * float(np.dot(midpoint, correction))
            delta_energy.append(de)
            linear_work.append(lw)
            constraint_work.append(cw)
            closure.append(de - lw - cw)
        energy_scale = max(
            float(np.max(0.5 * dy * np.sum(normalized**2, axis=1))),
            np.finfo(float).tiny,
        )
        maximum_closure = float(np.max(np.abs(closure)) / energy_scale)
        maximum_residual = float(np.max(np.abs(residuals)))
        maximum_reference_residual = float(np.max(np.abs(reference_residuals)))
        label = f"h{depth_m:g}m"
        arrays[f"{label}_times"] = times
        arrays[f"{label}_surface"] = surface
        arrays[f"{label}_reference_surface"] = reference_surface
        arrays[f"{label}_six_residuals"] = residuals
        arrays[f"{label}_reference_residuals"] = reference_residuals
        arrays[f"{label}_energy_delta"] = np.asarray(delta_energy)
        arrays[f"{label}_linear_work"] = np.asarray(linear_work)
        arrays[f"{label}_constraint_work"] = np.asarray(constraint_work)
        arrays[f"{label}_energy_closure"] = np.asarray(closure)
        rows.append(
            {
                "depth_m": depth_m,
                "n": y.size,
                "dy": dy,
                "dt": config.constant_identity_dt,
                "n_steps": n_steps,
                "matrix_relative_Frobenius_difference": matrix_difference,
                "kernel_maximum_relative_difference": kernel_difference,
                "surface_spacetime_relative_L2": field_difference,
                "maximum_six_column_residual": maximum_residual,
                "maximum_reference_residual": maximum_reference_residual,
                "maximum_normalized_CN_work_closure": maximum_closure,
                "maximum_absolute_surface": float(np.max(np.abs(surface))),
                "all_fields_finite": bool(np.all(np.isfinite(surface))),
                "runtime_s": runtime_s,
                "theta_rad": trace_metadata["theta_rad"],
            }
        )
    _write_csv(output / RAW_DIR_NAME / "constant_depth_identity.csv", rows)
    np.savez_compressed(
        output / RAW_DIR_NAME / "constant_depth_identity.npz", y=y, **arrays
    )
    summary: dict[str, object] = {
        "maximum_matrix_relative_Frobenius_difference": max(
            float(row["matrix_relative_Frobenius_difference"]) for row in rows
        ),
        "maximum_kernel_relative_difference": max(
            float(row["kernel_maximum_relative_difference"]) for row in rows
        ),
        "maximum_surface_spacetime_relative_L2": max(
            float(row["surface_spacetime_relative_L2"]) for row in rows
        ),
        "maximum_six_column_residual": max(
            float(row["maximum_six_column_residual"]) for row in rows
        ),
        "maximum_CN_work_closure": max(
            float(row["maximum_normalized_CN_work_closure"]) for row in rows
        ),
        "all_fields_finite": all(bool(row["all_fields_finite"]) for row in rows),
    }
    return rows, summary, arrays


def _single_frequency_inputs(
    p: CoastalParameters,
    frequency_hz: float,
    dy: float,
    dt: float,
) -> tuple[ModalThreeTraceLifting, tuple[Callable[[float], float], ...], dict[str, object]]:
    omega = np.array((2.0 * np.pi * frequency_hz * p.time_ref_s,))
    spectrum = PeriodicBoundarySpectrum(
        angular_frequency=omega,
        complex_amplitude=np.array((0.1 * np.exp(0.37j),)),
        period=1.0 / frequency_hz / p.time_ref_s,
        sample_dt=1.0 / frequency_hz / p.time_ref_s / 256.0,
        removed_mean=0.0,
        reconstruction_relative_l2=0.0,
        retained_variance_fraction=1.0,
    )
    lifting, metadata = build_modal_three_trace_lifting(
        spectrum,
        "discrete_c6c4",
        1.0,
        p.mu / 6.0,
        dy,
        dt,
        p.boundary_ramp_s / p.time_ref_s,
        d1_order=6,
    )
    return lifting, lifting.callables(), metadata


def _tma_inputs(
    p: CoastalParameters,
    config: StudyConfig,
    dy: float,
    dt: float,
) -> tuple[
    TMASeaState,
    ModalThreeTraceLifting,
    tuple[Callable[[float], float], ...],
    dict[str, object],
]:
    parameters = SeaStateParameters(
        significant_wave_height_m=0.3,
        peak_period_s=15.0,
        peak_enhancement_gamma=3.3,
        water_depth_m=15.0,
        frequency_min_hz=0.03,
        high_frequency_taper_start_hz=0.085,
        frequency_max_hz=0.105,
        synthesis_period_s=1800.0,
        ramp_duration_s=p.boundary_ramp_s,
        random_seed=config.random_seed,
    )
    sea_state = TMASeaState(parameters)
    spectrum = PeriodicBoundarySpectrum(
        angular_frequency=2.0
        * np.pi
        * sea_state.frequencies_hz
        * p.time_ref_s,
        complex_amplitude=(
            sea_state.amplitudes_m
            / p.a_ref_m
            * np.exp(1.0j * sea_state.phases_rad)
        ),
        period=parameters.synthesis_period_s / p.time_ref_s,
        sample_dt=parameters.synthesis_period_s / p.time_ref_s / 32768.0,
        removed_mean=0.0,
        reconstruction_relative_l2=0.0,
        retained_variance_fraction=1.0,
    )
    lifting, metadata = build_modal_three_trace_lifting(
        spectrum,
        "discrete_c6c4",
        1.0,
        p.mu / 6.0,
        dy,
        dt,
        p.boundary_ramp_s / p.time_ref_s,
        d1_order=6,
    )

    def physical_g0(time_value: float) -> float:
        return float(sea_state.truth_m(time_value * p.time_ref_s) / p.a_ref_m)

    return sea_state, lifting, lifting.callables(physical_g0), metadata


def _complex_carrier_coefficient(
    times_s: np.ndarray, values: np.ndarray, frequency_hz: float
) -> complex:
    phase = 2.0 * np.pi * frequency_hz * np.asarray(times_s)
    design = np.column_stack((np.cos(phase), np.sin(phase), np.ones_like(phase)))
    coefficients, *_ = np.linalg.lstsq(design, np.asarray(values), rcond=None)
    return complex(coefficients[0], -coefficients[1])


def _run_solver(
    p: CoastalParameters,
    length_m: float,
    n_short: int,
    dt: float,
    n_steps: int,
    output_stride: int,
    traces: tuple[Callable[[float], float], ...],
    *,
    bathymetry_kind: str = "beta_C7",
    root_sum_multiplier: float = 1.0,
) -> dict[str, object]:
    y, dy, n = _grid_for_length(length_m, p, short_n=n_short)
    depth = coastal_depth_ratio_y(
        y, length_ref_m=p.lambda_ref_m, kind=bathymetry_kind
    )
    solver = CoastalHighOrderLinearCNDABCSolver(
        y, depth, p.mu, dt, n_steps
    )
    if root_sum_multiplier <= 0.0:
        raise ValueError("root_sum_multiplier must be positive")
    if root_sum_multiplier != 1.0:
        # Positive control: perturb the complete e1 convolution kernel and
        # rebuild its zero-lag rows consistently.  Algebraic boundary
        # residuals therefore remain small; the extended-domain field metric,
        # not the row residual, must detect the deliberately wrong kernel.
        solver.kernels = replace(
            solver.kernels,
            root_sum=root_sum_multiplier * solver.kernels.root_sum,
        )
        constraints: list[np.ndarray] = []
        for shift in range(3):
            anchor = solver.n - 1 - shift
            row = np.zeros(solver.n)
            row[anchor] = solver.green_to_surface[anchor]
            row[anchor - 1] = (
                -solver.kernels.root_sum[0]
                * solver.green_to_surface[anchor - 1]
            )
            row[anchor - 2] = (
                solver.kernels.root_pair_sum[0]
                * solver.green_to_surface[anchor - 2]
            )
            row[anchor - 3] = (
                -solver.kernels.root_product[0]
                * solver.green_to_surface[anchor - 3]
            )
            constraints.append(row)
        solver.constraints = constraints
        bordered = solver.left_matrix.tolil()
        for row_index in range(3):
            bordered[row_index, :] = 0.0
            bordered[row_index, row_index] = 1.0
        for shift, constraint in enumerate(solver.constraints):
            bordered[solver.n - 1 - shift, :] = constraint
        solver.lu = splu(bordered.tocsc())
    start = time.perf_counter()
    times, surface, normalized, residuals = solver.run(
        np.zeros_like(y), output_stride, traces
    )
    runtime_s = time.perf_counter() - start
    return {
        "y": y,
        "dy": dy,
        "n": n,
        "depth": depth,
        "times": times,
        "surface": surface,
        "normalized": normalized,
        "residuals": residuals,
        "runtime_s": runtime_s,
        "solver": solver,
    }


def _extended_pair_metrics(
    p: CoastalParameters,
    config: StudyConfig,
    short: dict[str, object],
    extended: dict[str, object],
    frequency_hz: float | None,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    short_times = np.asarray(short["times"])
    long_times = np.asarray(extended["times"])
    if not np.array_equal(short_times, long_times):
        raise RuntimeError("short and extended domains have different saved times")
    short_surface = np.asarray(short["surface"])
    extended_common = np.asarray(extended["surface"])[:, : short_surface.shape[1]]
    difference = short_surface - extended_common
    times_s = short_times * p.time_ref_s
    analysis = times_s >= config.analysis_start_s
    if np.count_nonzero(analysis) < 8:
        raise RuntimeError("extended-domain analysis window is too short")
    spacetime = _relative_l2(short_surface[analysis], extended_common[analysis])
    reference_scale = max(
        float(np.sqrt(np.mean(extended_common[analysis] ** 2))),
        np.finfo(float).tiny,
    )
    time_error = np.sqrt(np.mean(difference**2, axis=1)) / reference_scale
    initial_or_forced_scale = max(
        float(np.max(np.sqrt(np.mean(extended_common**2, axis=1)))),
        np.finfo(float).tiny,
    )
    squared_ratio = np.mean(difference**2, axis=1) / initial_or_forced_scale**2
    result: dict[str, object] = {
        "analysis_window_s": [
            float(times_s[analysis][0]),
            float(times_s[analysis][-1]),
        ],
        "common_domain_spacetime_relative_L2": spacetime,
        "maximum_normalized_RMS_difference": float(np.max(time_error[analysis])),
        "maximum_squared_RMS_ratio": float(np.max(squared_ratio[analysis])),
        "maximum_short_residual": float(np.max(np.abs(short["residuals"]))),
        "maximum_extended_residual": float(np.max(np.abs(extended["residuals"]))),
        "short_runtime_s": float(short["runtime_s"]),
        "extended_runtime_s": float(extended["runtime_s"]),
        "all_fields_finite": bool(
            np.all(np.isfinite(short_surface))
            and np.all(np.isfinite(np.asarray(extended["surface"])))
        ),
    }
    if frequency_hz is not None:
        gauge_indices = [
            int(round(position / p.lambda_ref_m / float(short["dy"])))
            for position in (500.0, 1500.0, 2500.0, 3500.0)
        ]
        amplitude_errors: list[float] = []
        phase_errors: list[float] = []
        for index in gauge_indices:
            c_short = _complex_carrier_coefficient(
                times_s[analysis], short_surface[analysis, index], frequency_hz
            )
            c_long = _complex_carrier_coefficient(
                times_s[analysis], extended_common[analysis, index], frequency_hz
            )
            amplitude_errors.append(
                abs(abs(c_short) - abs(c_long))
                / max(abs(c_long), np.finfo(float).tiny)
            )
            phase_errors.append(abs(float(np.angle(c_short / c_long))))
        result.update(
            {
                "gauge_positions_m": [500.0, 1500.0, 2500.0, 3500.0],
                "maximum_gauge_amplitude_relative_error": float(max(amplitude_errors)),
                "maximum_gauge_phase_error_rad": float(max(phase_errors)),
            }
        )
    else:
        hs_short = 4.0 * np.std(short_surface[analysis], axis=0) * p.a_ref_m
        hs_long = 4.0 * np.std(extended_common[analysis], axis=0) * p.a_ref_m
        result.update(
            {
                "Hs_profile_relative_L2": _relative_l2(hs_short, hs_long),
                "maximum_Hs_absolute_difference_m": float(
                    np.max(np.abs(hs_short - hs_long))
                ),
            }
        )
    arrays = {
        "times_s": times_s,
        "y_short_m": np.asarray(short["y"]) * p.lambda_ref_m,
        "y_extended_m": np.asarray(extended["y"]) * p.lambda_ref_m,
        "depth_short_m": np.asarray(short["depth"]) * p.h_ref_m,
        "depth_extended_m": np.asarray(extended["depth"]) * p.h_ref_m,
        "eta_short_m": short_surface * p.a_ref_m,
        "eta_extended_m": np.asarray(extended["surface"]) * p.a_ref_m,
        "six_residuals_short": np.asarray(short["residuals"]),
        "six_residuals_extended": np.asarray(extended["residuals"]),
        "common_difference_m": difference * p.a_ref_m,
        "normalized_RMS_difference": time_error,
        "squared_RMS_ratio": squared_ratio,
        "analysis_mask": analysis,
    }
    if frequency_hz is None:
        arrays["Hs_short_m"] = 4.0 * np.std(short_surface[analysis], axis=0) * p.a_ref_m
        arrays["Hs_extended_common_m"] = (
            4.0 * np.std(extended_common[analysis], axis=0) * p.a_ref_m
        )
    return result, arrays


def run_extended_domain_cases(
    output: Path, p: CoastalParameters, config: StudyConfig
) -> tuple[dict[str, object], dict[str, dict[str, np.ndarray]]]:
    y_short, dy, _ = _grid_for_length(
        config.short_length_m, p, short_n=config.medium_n_short
    )
    n_steps = _aligned_steps(config.final_time_s, p, config.medium_dt)
    actual_final_s = n_steps * config.medium_dt * p.time_ref_s
    if actual_final_s < config.analysis_start_s + 100.0:
        raise ValueError("final time must leave at least 100 s for analysis")

    _, single_traces, single_lifting_meta = _single_frequency_inputs(
        p, config.single_frequency_hz, dy, config.medium_dt
    )
    single_short = _run_solver(
        p,
        config.short_length_m,
        config.medium_n_short,
        config.medium_dt,
        n_steps,
        config.output_stride,
        single_traces,
    )
    single_extended = _run_solver(
        p,
        config.extended_length_m,
        config.medium_n_short,
        config.medium_dt,
        n_steps,
        config.output_stride,
        single_traces,
    )
    single_metrics, single_arrays = _extended_pair_metrics(
        p, config, single_short, single_extended, config.single_frequency_hz
    )
    np.savez_compressed(
        output / RAW_DIR_NAME / "single_frequency_4km_8km.npz", **single_arrays
    )
    del single_short, single_extended

    sea_state, _, tma_traces, tma_lifting_meta = _tma_inputs(
        p, config, dy, config.medium_dt
    )
    tma_short = _run_solver(
        p,
        config.short_length_m,
        config.medium_n_short,
        config.medium_dt,
        n_steps,
        config.output_stride,
        tma_traces,
    )
    tma_extended = _run_solver(
        p,
        config.extended_length_m,
        config.medium_n_short,
        config.medium_dt,
        n_steps,
        config.output_stride,
        tma_traces,
    )
    tma_metrics, tma_arrays = _extended_pair_metrics(
        p, config, tma_short, tma_extended, None
    )
    perturbed_short = _run_solver(
        p,
        config.short_length_m,
        config.medium_n_short,
        config.medium_dt,
        n_steps,
        config.output_stride,
        tma_traces,
        root_sum_multiplier=1.01,
    )
    positive_metrics, positive_arrays = _extended_pair_metrics(
        p, config, perturbed_short, tma_extended, None
    )
    np.savez_compressed(
        output / RAW_DIR_NAME / "tma_4km_8km.npz", **tma_arrays
    )
    np.savez_compressed(
        output / RAW_DIR_NAME / "tma_perturbed_kernel_positive_control.npz",
        **positive_arrays,
    )
    del tma_short, tma_extended, perturbed_short

    summary: dict[str, object] = {
        "n_steps": n_steps,
        "actual_final_time_s": actual_final_s,
        "medium_short_n": int(y_short.size),
        "medium_extended_n": int(
            _grid_for_length(
                config.extended_length_m, p, short_n=config.medium_n_short
            )[2]
        ),
        "dy_dimensionless": dy,
        "single_frequency": single_metrics,
        "TMA": tma_metrics,
        "perturbed_kernel_positive_control": {
            **positive_metrics,
            "root_sum_multiplier": 1.01,
            "positive_to_matched_field_error_ratio": float(
                positive_metrics["common_domain_spacetime_relative_L2"]
                / max(
                    float(tma_metrics["common_domain_spacetime_relative_L2"]),
                    np.finfo(float).tiny,
                )
            ),
            "interpretation": (
                "A matched algebraic residual can remain tiny for a wrong "
                "kernel; the extended-domain field metric detects the error."
            ),
        },
        "single_lifting": single_lifting_meta,
        "TMA_lifting": tma_lifting_meta,
        "TMA_sea_state": sea_state.metadata(),
        "reference_scope": (
            "The 8 km calculation is a same-scheme extension-consistency "
            "reference, not an independent PDE truth.  Resolved-band travel "
            "times separate the energetic return from the main window, but "
            "dispersive/implicit precursors preclude a strict finite-domain "
            "causality claim."
        ),
    }
    return summary, {
        "single": single_arrays,
        "TMA": tma_arrays,
        "positive_control": positive_arrays,
    }


def run_full_refinement(
    output: Path, p: CoastalParameters, config: StudyConfig
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Optional low-cost single-carrier coupled and time-only convergence."""

    target_medium_steps = int(round(config.refinement_final_dimensionless / config.medium_dt))
    if target_medium_steps % 2:
        target_medium_steps += 1
    levels = (
        ("coarse", 1537, 0.004, target_medium_steps // 2, 35),
        ("medium", 3073, 0.002, target_medium_steps, 70),
        ("fine", 6145, 0.001, 2 * target_medium_steps, 140),
    )
    runs: dict[str, dict[str, object]] = {}
    arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for label, n, dt, n_steps, stride in levels:
        _, dy, _ = _grid_for_length(4000.0, p, short_n=n)
        _, traces, _ = _single_frequency_inputs(
            p, config.refinement_frequency_hz, dy, dt
        )
        run = _run_solver(p, 4000.0, n, dt, n_steps, stride, traces)
        runs[label] = run
        arrays[f"{label}_times_s"] = np.asarray(run["times"]) * p.time_ref_s
        arrays[f"{label}_y_m"] = np.asarray(run["y"]) * p.lambda_ref_m
        arrays[f"{label}_eta_m"] = np.asarray(run["surface"]) * p.a_ref_m
        rows.append(
            {
                "family": "coupled",
                "level": label,
                "n": n,
                "dt": dt,
                "n_steps": n_steps,
                "runtime_s": float(run["runtime_s"]),
                "maximum_residual": float(np.max(np.abs(run["residuals"]))),
            }
        )
    if not (
        np.array_equal(runs["coarse"]["times"], runs["medium"]["times"])
        and np.array_equal(runs["medium"]["times"], runs["fine"]["times"])
    ):
        raise RuntimeError("refinement output times are not exactly aligned")
    times_s = np.asarray(runs["medium"]["times"]) * p.time_ref_s
    analysis = times_s >= config.refinement_analysis_start_s
    coarse = np.asarray(runs["coarse"]["surface"])
    medium = np.asarray(runs["medium"]["surface"])
    fine = np.asarray(runs["fine"]["surface"])
    coarse_medium = float(
        np.sqrt(np.mean((coarse[analysis] - medium[analysis, ::2]) ** 2))
    )
    medium_fine = float(
        np.sqrt(np.mean((medium[analysis] - fine[analysis, ::2]) ** 2))
    )
    coupled_order = _observed_order(coarse_medium, medium_fine)
    coupled_relative = _relative_l2(medium[analysis], fine[analysis, ::2])
    hs_medium = 4.0 * np.std(medium[analysis], axis=0)
    hs_fine = 4.0 * np.std(fine[analysis], axis=0)[::2]
    hs_relative = _relative_l2(hs_medium, hs_fine)

    # Time-only: medium grid at dt=.004 and .001; reuse dt=.002 above.
    time_runs: dict[str, dict[str, object]] = {"dt_0p002": runs["medium"]}
    for label, dt, steps, stride in (
        ("dt_0p004", 0.004, target_medium_steps // 2, 35),
        ("dt_0p001", 0.001, 2 * target_medium_steps, 140),
    ):
        _, dy, _ = _grid_for_length(4000.0, p, short_n=3073)
        _, traces, _ = _single_frequency_inputs(
            p, config.refinement_frequency_hz, dy, dt
        )
        run = _run_solver(p, 4000.0, 3073, dt, steps, stride, traces)
        time_runs[label] = run
        arrays[f"{label}_times_s"] = np.asarray(run["times"]) * p.time_ref_s
        arrays[f"{label}_eta_m"] = np.asarray(run["surface"]) * p.a_ref_m
        rows.append(
            {
                "family": "time_only",
                "level": label,
                "n": 3073,
                "dt": dt,
                "n_steps": steps,
                "runtime_s": float(run["runtime_s"]),
                "maximum_residual": float(np.max(np.abs(run["residuals"]))),
            }
        )
    time_coarse = np.asarray(time_runs["dt_0p004"]["surface"])
    time_medium = np.asarray(time_runs["dt_0p002"]["surface"])
    time_fine = np.asarray(time_runs["dt_0p001"]["surface"])
    time_cm = float(np.sqrt(np.mean((time_coarse[analysis] - time_medium[analysis]) ** 2)))
    time_mf = float(np.sqrt(np.mean((time_medium[analysis] - time_fine[analysis]) ** 2)))
    time_order = _observed_order(time_cm, time_mf)
    time_relative = _relative_l2(time_medium[analysis], time_fine[analysis])

    # Same dt=.001, different spatial grids: medium/time-refined versus fine.
    # This isolates the remaining grid/lifting contribution from the CN time
    # contribution measured immediately above.
    spatial_medium = time_fine
    spatial_fine = fine[:, ::2]
    spatial_mf = float(
        np.sqrt(np.mean((spatial_medium[analysis] - spatial_fine[analysis]) ** 2))
    )
    spatial_relative = _relative_l2(
        spatial_medium[analysis], spatial_fine[analysis]
    )
    spatial_hs_medium = 4.0 * np.std(spatial_medium[analysis], axis=0)
    spatial_hs_fine = 4.0 * np.std(spatial_fine[analysis], axis=0)
    spatial_hs_relative = _relative_l2(spatial_hs_medium, spatial_hs_fine)
    gauge_positions_m = (500.0, 1500.0, 2500.0, 3500.0)
    spatial_gauge_amplitude_errors: list[float] = []
    spatial_gauge_phase_errors: list[float] = []
    spatial_gauge_time_shifts_s: list[float] = []
    for position_m in gauge_positions_m:
        index = int(round(position_m / (p.domain_m / (3073 - 1))))
        medium_coefficient = _complex_carrier_coefficient(
            times_s[analysis],
            spatial_medium[analysis, index],
            config.refinement_frequency_hz,
        )
        fine_coefficient = _complex_carrier_coefficient(
            times_s[analysis],
            spatial_fine[analysis, index],
            config.refinement_frequency_hz,
        )
        phase_error = abs(float(np.angle(medium_coefficient / fine_coefficient)))
        spatial_gauge_amplitude_errors.append(
            abs(abs(medium_coefficient) - abs(fine_coefficient))
            / max(abs(fine_coefficient), np.finfo(float).tiny)
        )
        spatial_gauge_phase_errors.append(phase_error)
        spatial_gauge_time_shifts_s.append(
            phase_error / (2.0 * np.pi * config.refinement_frequency_hz)
        )
    _write_csv(output / RAW_DIR_NAME / "full_refinement_runs.csv", rows)
    np.savez_compressed(output / RAW_DIR_NAME / "full_refinement.npz", **arrays)
    summary: dict[str, object] = {
        "analysis_window_s": [float(times_s[analysis][0]), float(times_s[analysis][-1])],
        "coupled_coarse_medium_RMS_difference": coarse_medium,
        "coupled_medium_fine_RMS_difference": medium_fine,
        "coupled_observed_order": coupled_order,
        "coupled_medium_fine_spacetime_relative_L2": coupled_relative,
        "coupled_medium_fine_Hs_profile_relative_L2": hs_relative,
        "time_only_coarse_medium_RMS_difference": time_cm,
        "time_only_medium_fine_RMS_difference": time_mf,
        "time_only_observed_order": time_order,
        "time_only_medium_fine_spacetime_relative_L2": time_relative,
        "spatial_only_medium_fine_RMS_difference": spatial_mf,
        "spatial_only_medium_fine_spacetime_relative_L2": spatial_relative,
        "spatial_only_medium_fine_Hs_profile_relative_L2": spatial_hs_relative,
        "spatial_to_time_only_RMS_difference_ratio": spatial_mf
        / max(time_mf, np.finfo(float).tiny),
        "spatial_gauge_positions_m": list(gauge_positions_m),
        "maximum_spatial_gauge_amplitude_relative_error": float(
            max(spatial_gauge_amplitude_errors)
        ),
        "maximum_spatial_gauge_phase_error_rad": float(
            max(spatial_gauge_phase_errors)
        ),
        "maximum_spatial_gauge_time_shift_s": float(
            max(spatial_gauge_time_shifts_s)
        ),
        "maximum_residual": max(float(row["maximum_residual"]) for row in rows),
        "frequency_hz": config.refinement_frequency_hz,
        "final_time_dimensionless": config.refinement_final_dimensionless,
        "reference_scope": "self-convergence only; no exact truth is implied",
    }
    return summary, arrays


def _build_gates(
    manufactured: dict[str, float],
    regularity: dict[str, object],
    identity: dict[str, object],
    extended: dict[str, object] | None,
    refinement: dict[str, object] | None,
) -> dict[str, bool | None]:
    c1 = regularity["profiles"]["cubic_C1"]
    c7 = regularity["profiles"]["beta_C7"]
    gates: dict[str, bool | None] = {
        "analytic_manufactured_L2_order_above_3p7": manufactured["finest_L2_order"] >= 3.7,
        "analytic_manufactured_Linf_order_above_3p5": manufactured["finest_Linf_order"] >= 3.5,
        "analytic_manufactured_finest_relative_L2_below_1e-7": manufactured["finest_relative_L2_error"] < 1.0e-7,
        "C7_junction_RMS_order_above_3p4": float(c7["finest_RMS_order"]) >= 3.4,
        "C7_junction_Linf_order_above_3p2": float(c7["finest_Linf_order"]) >= 3.2,
        "C7_finest_RMS_improves_C1_by_1e4": float(regularity["C1_to_C7_finest_RMS_ratio"]) > 1.0e4,
        "C1_positive_control_does_not_false_converge": float(c1["finest_RMS_order"]) < 0.0,
        "constant_depth_matrix_identity_below_1e-13": float(identity["maximum_matrix_relative_Frobenius_difference"]) < 1.0e-13,
        "constant_depth_kernel_identity_below_1e-12": float(identity["maximum_kernel_relative_difference"]) < 1.0e-12,
        "constant_depth_field_identity_below_1e-11": float(identity["maximum_surface_spacetime_relative_L2"]) < 1.0e-11,
        "constant_depth_six_residuals_below_1e-10": float(identity["maximum_six_column_residual"]) < 1.0e-10,
        "constant_depth_CN_algebraic_ledger_roundoff_below_1e-12": float(identity["maximum_CN_work_closure"]) < 1.0e-12,
        "constant_depth_fields_are_finite": bool(identity["all_fields_finite"]),
    }
    if extended is None:
        gates.update(
            {
                "single_4km_8km_field_difference_below_1e-6": None,
                "single_4km_8km_gauge_amplitude_below_1e-5": None,
                "single_4km_8km_gauge_phase_below_1e-5_rad": None,
                "TMA_4km_8km_field_difference_below_1e-6": None,
                "perturbed_kernel_positive_control_above_1e-3": None,
                "positive_control_to_matched_error_ratio_above_1e6": None,
                "all_extended_fields_finite": None,
                "all_extended_boundary_residuals_below_1e-10": None,
            }
        )
    else:
        single = extended["single_frequency"]
        tma = extended["TMA"]
        positive = extended["perturbed_kernel_positive_control"]
        gates.update(
            {
                "single_4km_8km_field_difference_below_1e-6": float(single["common_domain_spacetime_relative_L2"]) < 1.0e-6,
                "single_4km_8km_gauge_amplitude_below_1e-5": float(single["maximum_gauge_amplitude_relative_error"]) < 1.0e-5,
                "single_4km_8km_gauge_phase_below_1e-5_rad": float(single["maximum_gauge_phase_error_rad"]) < 1.0e-5,
                "TMA_4km_8km_field_difference_below_1e-6": float(tma["common_domain_spacetime_relative_L2"]) < 1.0e-6,
                "perturbed_kernel_positive_control_above_1e-3": float(
                    positive["common_domain_spacetime_relative_L2"]
                ) > 1.0e-3,
                "positive_control_to_matched_error_ratio_above_1e6": float(
                    positive["positive_to_matched_field_error_ratio"]
                ) > 1.0e6,
                "all_extended_fields_finite": bool(single["all_fields_finite"] and tma["all_fields_finite"]),
                "all_extended_boundary_residuals_below_1e-10": max(
                    float(single["maximum_short_residual"]),
                    float(single["maximum_extended_residual"]),
                    float(tma["maximum_short_residual"]),
                    float(tma["maximum_extended_residual"]),
                ) < 1.0e-10,
            }
        )
    if refinement is None:
        gates.update(
            {
                "coupled_refinement_order_between_1p7_and_2p3": None,
                "coupled_medium_fine_field_difference_below_1_percent": None,
                "coupled_medium_fine_Hs_difference_below_0p5_percent": None,
                "time_only_order_between_1p8_and_2p2": None,
                "spatial_only_max_gauge_phase_below_0p02_rad": None,
                "spatial_only_max_gauge_time_shift_below_0p02s": None,
            }
        )
    else:
        gates.update(
            {
                "coupled_refinement_order_between_1p7_and_2p3": 1.7 <= float(refinement["coupled_observed_order"]) <= 2.3,
                "coupled_medium_fine_field_difference_below_1_percent": float(refinement["coupled_medium_fine_spacetime_relative_L2"]) < 0.01,
                "coupled_medium_fine_Hs_difference_below_0p5_percent": float(refinement["coupled_medium_fine_Hs_profile_relative_L2"]) < 0.005,
                "time_only_order_between_1p8_and_2p2": 1.8 <= float(refinement["time_only_observed_order"]) <= 2.2,
                "spatial_only_max_gauge_phase_below_0p02_rad": float(
                    refinement["maximum_spatial_gauge_phase_error_rad"]
                ) < 0.02,
                "spatial_only_max_gauge_time_shift_below_0p02s": float(
                    refinement["maximum_spatial_gauge_time_shift_s"]
                ) < 0.02,
            }
        )
    return gates


def _write_report(output: Path, metrics: dict[str, object]) -> Path:
    m = metrics["manufactured_operator"]
    r = metrics["bathymetry_regularity"]
    i = metrics["constant_depth_identity"]
    e = metrics.get("extended_domain")
    refinement = metrics.get("full_refinement")
    gates = metrics["acceptance_gates"]
    lines = [
        "# Experiment 14: high-order linear variable-depth CN with matched DABC",
        "",
        "## Scope of the conclusions",
        "",
        "This experiment validates the isolated C6-D1/C4-D3 variable-depth linear",
        "solver. It contains no U2 nonlinearity and does not modify the production",
        "solver. The analytic manufactured solution is a non-circular reference for",
        "the interior variable-coefficient operator; the 8 km solution is a",
        "same-scheme extended-domain consistency reference, not an independent PDE",
        "truth; the 1% kernel perturbation serves as a positive control for the",
        "error metric.",
        "The solver coordinate is the shoreward-increasing y=(L-x)/lambda_ref with",
        "waves travelling along +y; propagation_sign=-1 in the parameter table",
        "belongs only to the legacy x-coordinate implementation.",
        "",
        "## Analytic manufactured operator",
        "",
        f"- Finest-grid relative L2: `{float(m['finest_relative_L2_error']):.3e}`; observed order: `{float(m['finest_L2_order']):.3f}`.",
        f"- Finest-grid relative Linf: `{float(m['finest_relative_Linf_error']):.3e}`; observed order: `{float(m['finest_Linf_order']):.3f}`.",
        "",
        "## Bathymetry regularity",
        "",
        f"- C1 finest RMS order: `{float(r['profiles']['cubic_C1']['finest_RMS_order']):.3f}`.",
        f"- C3 finest RMS order: `{float(r['profiles']['septic_C3']['finest_RMS_order']):.3f}`.",
        f"- C7 finest RMS/Linf orders: `{float(r['profiles']['beta_C7']['finest_RMS_order']):.3f}` / `{float(r['profiles']['beta_C7']['finest_Linf_order']):.3f}`.",
        f"- C1/C7 finest RMS self-difference ratio: `{float(r['C1_to_C7_finest_RMS_ratio']):.3e}`.",
        f"- C7 maximum bed slope: `{float(r['physical_shape_diagnostics']['beta_C7']['maximum_absolute_bed_slope']):.5f}`; "
        f"maximum `lambda_ref |h_x|/h`: `{float(r['physical_shape_diagnostics']['beta_C7']['maximum_lambda_ref_abs_hx_over_h']):.3f}`.",
        f"- C7 shallow-water phase-speed travel-time change relative to C1: `{100.0 * float(r['physical_shape_diagnostics']['beta_C7']['travel_time_relative_to_cubic']):.3f}%`.",
        "",
        "C1 and C3 are positive controls that are expected to fail; only C7 is",
        "compatible with the junction regularity required by the seven-point",
        "high-order stencil. This does not mean the C7 and C1 physical slope shapes",
        "are fully equivalent; slope/WKB sensitivity should still be reported.",
        "",
        "## Constant-depth degeneration",
        "",
        f"- Maximum matrix difference: `{float(i['maximum_matrix_relative_Frobenius_difference']):.3e}`.",
        f"- Maximum kernel difference: `{float(i['maximum_kernel_relative_difference']):.3e}`.",
        f"- Maximum space-time field difference: `{float(i['maximum_surface_spacetime_relative_L2']):.3e}`.",
        f"- Maximum six-column boundary residual: `{float(i['maximum_six_column_residual']):.3e}`.",
        f"- CN algebraic work-ledger round-off check: `{float(i['maximum_CN_work_closure']):.3e}` (an identity by construction; not evidence of stability or physical energy).",
        "",
    ]
    if e is not None:
        s = e["single_frequency"]
        t = e["TMA"]
        lines.extend(
            [
                "## 4 km / 8 km same-scheme extended domain",
                "",
                f"- Single-frequency common 0--4 km space-time relative L2: `{float(s['common_domain_spacetime_relative_L2']):.3e}`.",
                f"- Single-frequency maximum gauge amplitude/phase difference: `{float(s['maximum_gauge_amplitude_relative_error']):.3e}` / `{float(s['maximum_gauge_phase_error_rad']):.3e} rad`.",
                f"- TMA common 0--4 km space-time relative L2: `{float(t['common_domain_spacetime_relative_L2']):.3e}`.",
                f"- TMA Hs profile relative L2: `{float(t['Hs_profile_relative_L2']):.3e}`.",
                f"- 1% `e1` kernel-perturbation positive-control space-time relative L2: `{float(e['perturbed_kernel_positive_control']['common_domain_spacetime_relative_L2']):.3e}`; "
                f"amplification over the matched DABC: `{float(e['perturbed_kernel_positive_control']['positive_to_matched_field_error_ratio']):.3e}` times.",
                "- The six-column constraint residual only checks the boundary algebraic equations; the positive control can also keep a small residual, so it must not be read as a reflection coefficient.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## 4 km / 8 km same-scheme extended domain",
                "",
                "Not run in this invocation (`--static-only`).",
                "",
            ]
        )
    if refinement is not None:
        lines.extend(
            [
                "## Optional full refinement",
                "",
                f"- Coupled order: `{float(refinement['coupled_observed_order']):.3f}`.",
                f"- Time-only order: `{float(refinement['time_only_observed_order']):.3f}`.",
                f"- Medium/fine field relative L2: `{float(refinement['coupled_medium_fine_spacetime_relative_L2']):.3e}`.",
                f"- Space-only medium/fine relative L2 at the same dt: `{float(refinement['spatial_only_medium_fine_spacetime_relative_L2']):.3e}`.",
                f"- Space-only / time-only RMS ratio: `{float(refinement['spatial_to_time_only_RMS_difference_ratio']):.3e}`.",
                f"- Maximum space-only gauge phase error / equivalent time shift: `{float(refinement['maximum_spatial_gauge_phase_error_rad']):.3e} rad` / `{float(refinement['maximum_spatial_gauge_time_shift_s']):.3e} s`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Predefined acceptance gates",
            "",
            *[
                f"- {'PASS' if value else ('NOT RUN' if value is None else 'FAIL')}: `{name}`"
                for name, value in gates.items()
            ],
            "",
            "## What can and cannot be claimed",
            "",
            "- Can be claimed: consistency of the analytic smooth",
            "  variable-coefficient operator, high-order self-convergence under the",
            "  C7 junction, implementation identity with the Experiment 12 solver",
            "  at constant depth, and the DABC truncation error within the tested",
            "  window.",
            "- Cannot be claimed: high-order convergence of the full nonlinear",
            "  vKdV, the 8 km solution as an independent PDE truth, a strictly",
            "  causal lifting for arbitrary real-time boundaries, or continuous",
            "  physical energy conservation.",
            "- A variable-depth open domain does not require a constant quadratic",
            "  invariant; the work ledger here is a solver-consistent algebraic",
            "  account, not a continuous physical energy-flux proof.",
            "",
        ]
    )
    path = output / "experiment_14_high_order_variable_depth_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_study(
    output: Path,
    *,
    static_only: bool = False,
    full_refinement: bool = False,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    (output / RAW_DIR_NAME).mkdir(parents=True, exist_ok=True)
    # Remove only files produced by earlier Experiment 14 layouts.  In
    # particular, do not allow the retired 4/6-km datasets to be silently
    # carried into a new manifest after the reference was changed to 8 km.
    retired_outputs = (
        output / RAW_DIR_NAME / "single_frequency_4km_6km.npz",
        output / RAW_DIR_NAME / "tma_4km_6km.npz",
    )
    for retired in retired_outputs:
        retired.unlink(missing_ok=True)
    p = CoastalParameters()
    config = StudyConfig()
    protected_sources = (
        PROJECT_DIR / "transparent_boundary_vkdv.py",
        PROJECT_DIR / "high_order_matched_dabc.py",
        PROJECT_DIR / "high_order_incident_lifting.py",
        PROJECT_DIR / "high_order_variable_depth_dabc.py",
        PROJECT_DIR / "pde_core.py",
        PROJECT_DIR / "sea_state_boundary.py",
    )
    hashes_before = {str(path): _sha256(path) for path in protected_sources}
    started = time.perf_counter()

    manufactured_rows, manufactured_summary, _ = analytic_manufactured_operator(output, p)
    regularity_rows, regularity_summary, regularity_arrays = bathymetry_regularity_audit(output, p)
    identity_rows, identity_summary, _ = constant_depth_identity(output, p, config)

    extended_summary: dict[str, object] | None = None
    extended_arrays: dict[str, dict[str, np.ndarray]] | None = None
    if not static_only:
        extended_summary, extended_arrays = run_extended_domain_cases(output, p, config)

    refinement_summary: dict[str, object] | None = None
    if full_refinement:
        if static_only:
            raise ValueError("--full-refinement cannot be combined with --static-only")
        refinement_summary, _ = run_full_refinement(output, p, config)

    gates = _build_gates(
        manufactured_summary,
        regularity_summary,
        identity_summary,
        extended_summary,
        refinement_summary,
    )
    hashes_after = {str(path): _sha256(path) for path in protected_sources}
    production_unchanged = hashes_before == hashes_after
    gates["protected_production_and_Exp12_13_sources_unchanged"] = production_unchanged
    executed_gates = [value for value in gates.values() if value is not None]
    metrics: dict[str, object] = {
        "experiment": "14_high_order_linear_variable_depth_CN_DABC",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analytical_question": (
            "Does the isolated C6-D1/C4-D3--CN variable-depth solver retain "
            "analytic operator consistency, exact constant-depth degeneration "
            "and same-scheme extension invariance of its matched outflow on "
            "the tested C7 bathymetry?"
        ),
        "scope": {
            "linear_only": True,
            "bathymetry_candidate": "beta_C7",
            "physical_boundary_count": 1,
            "numerical_incident_trace_count": 3,
            "solver_coordinate": "y=(L-x)/lambda_ref, increasing shoreward",
            "solver_propagation_direction": "+y",
            "legacy_parameter_note": (
                "CoastalParameters.propagation_sign=-1 applies to the old "
                "offshore-increasing x coordinate, not to this y solver"
            ),
            "production_solver_modified": False,
            "static_only": static_only,
            "full_refinement": full_refinement,
        },
        "reference_parameters": asdict(p),
        "study_config": asdict(config),
        "manufactured_operator": manufactured_summary,
        "bathymetry_regularity": regularity_summary,
        "constant_depth_identity": identity_summary,
        "extended_domain": extended_summary,
        "full_refinement": refinement_summary,
        "reference_hierarchy": {
            "non_circular_operator_reference": "closed-form analytic manufactured operator",
            "non_circular_constant_depth_reference": "Exp13 analytic fully-discrete modal field, linked by the constant-depth identity gate",
            "boundary_extension_consistency_reference": "8 km same-scheme constant-shelf extension; not an independent PDE truth",
            "boundary_metric_positive_control": "4 km DABC with the complete e1 kernel multiplied by 1.01",
            "not_truth_references": ["grid self-convergence", "WKB", "same-scheme solver identity"],
        },
        "acceptance_gates": gates,
        "all_executed_gates_passed": bool(all(executed_gates)),
        "runtime_s": time.perf_counter() - started,
        "provenance": {
            "protected_hashes_before": hashes_before,
            "protected_hashes_after": hashes_after,
            "protected_sources_unchanged": production_unchanged,
        },
    }
    _write_json(output / "metrics.json", metrics)
    _write_csv(
        output / RAW_DIR_NAME / "acceptance_gates.csv",
        [
            {"gate": name, "status": "NOT_RUN" if value is None else ("PASS" if value else "FAIL")}
            for name, value in gates.items()
        ],
    )
    report = _write_report(output, metrics)
    upstream_results = (
        PROJECT_DIR
        / "results"
        / "transparent_boundary"
        / "high_order_incident_lifting"
        / "metrics.json",
        PROJECT_DIR
        / "results"
        / "transparent_boundary"
        / "high_order_incident_lifting"
        / "manifest.json",
    )
    artifact_hashes = [
        {
            "relative_path": str(path.relative_to(output)).replace("\\", "/"),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "source_hashes": {
            str(path): _sha256(path)
            for path in (
                Path(__file__).resolve(),
                PROJECT_DIR / "high_order_variable_depth_dabc.py",
                PROJECT_DIR / "high_order_matched_dabc.py",
                PROJECT_DIR / "high_order_incident_lifting.py",
                PROJECT_DIR / "pde_core.py",
                PROJECT_DIR / "sea_state_boundary.py",
                PROJECT_DIR / "transparent_boundary_vkdv.py",
            )
        },
        "upstream_result_hashes": {
            str(path): _sha256(path) for path in upstream_results if path.exists()
        },
        "artifact_hashes": artifact_hashes,
        "outputs": {
            "metrics": str(output / "metrics.json"),
            "report": str(report),
            "raw_data": str(output / RAW_DIR_NAME),
        },
    }
    _write_json(output / "manifest.json", manifest)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="run manufactured, regularity and constant-depth gates only",
    )
    parser.add_argument(
        "--full-refinement",
        action="store_true",
        help="also run coupled coarse/medium/fine and fixed-grid time refinement",
    )
    args = parser.parse_args()
    metrics = run_study(
        args.output.resolve(),
        static_only=args.static_only,
        full_refinement=args.full_refinement,
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "all_executed_gates_passed": metrics["all_executed_gates_passed"],
        "runtime_s": metrics["runtime_s"],
    }, indent=2))


if __name__ == "__main__":
    main()
