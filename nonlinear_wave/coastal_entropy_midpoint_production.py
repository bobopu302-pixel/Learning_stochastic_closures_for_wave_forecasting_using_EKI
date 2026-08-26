"""Final physical-time coastal vKdV production driver.

Origin: 3. KDV_nonlinear_case/coastal_entropy_midpoint_production.py
Changes vs origin (numerics untouched; this remains the reference PDE
production driver):
* matplotlib imports, colour constants, DOMAIN_FOOTER and the five
  figure functions (_plot_bathymetry_and_boundary, _plot_xt,
  _plot_colour_3d, _plot_waterfall, _plot_diagnostics) deleted along
  with their call sites (release ships no plotting); the replot npz,
  every CSV, metrics.json, manifest.json, the Markdown report and the
  post-run audit are all still written;
* _exact_modal_spectrum deleted (its only consumer was a deleted
  figure); the vacuous 'plots_exclude_4_to_10km_guard' gate dropped;
* note: the frozen_Exp14_to_Exp17 tree gates audit archived result
  trees that are NOT shipped in this release, so they report
  file_count=0 / False here; that is expected and does not affect the
  numerical gates.

The driver *directly* reuses the validated
``CoastalHighOrderImplicitMidpointDABCSolver``.  It does not alter or write to
Experiments 14--17.  The reported physical interval is x in [0, 4] km, with

    x = 4 km - y,

so the TMA truth input is imposed at x=4 km (y=0).  The numerical grid extends
to y=10 km solely to delay the artificial outflow boundary.  Hence y=4--10 km
is a numerical guard and is never plotted as a physical coastal prediction.
The point x=0 is a 5 m-depth reporting limit followed by the same 5 m shelf;
it is not a shoreline and this model makes no run-up claim.

Default production command (intentionally expensive)::

    python coastal_entropy_midpoint_production.py

Fast end-to-end workflow check::

    python coastal_entropy_midpoint_production.py --smoke
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import scipy

from pde_core import CoastalParameters
from high_order_implicit_midpoint_candidate import (
    CoastalHighOrderImplicitMidpointDABCSolver,
)
from high_order_nonlinear_candidate_screening import ScreeningStepAudit
from high_order_variable_depth_dabc import coastal_depth_ratio_y
from high_order_variable_depth_dabc_study import (
    StudyConfig as LinearStudyConfig,
    _aligned_steps,
    _grid_for_length,
    _tma_inputs,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_BASE = PROJECT_DIR / "results" / "physical_time_final"
RAW_DIRECTORY_NAME = "raw_data"


@dataclass(frozen=True)
class ProductionConfig:
    """Numerical and reporting controls for one production bundle."""

    requested_duration_s: float = 1800.0
    physical_length_m: float = 4000.0
    computational_length_m: float = 10000.0
    medium_n4: int = 3073
    fine_n4: int = 6145
    medium_dt: float = 0.002
    fine_dt: float = 0.001
    medium_output_stride: int = 70
    fine_output_stride: int = 140
    analysis_start_s: float = 600.0
    random_seed: int = 20260718
    gauge_x_m: tuple[float, ...] = (
        0.0,
        500.0,
        1000.0,
        2000.0,
        3000.0,
        4000.0,
    )


@dataclass(frozen=True)
class ResolutionSpec:
    name: str
    n4: int
    dt: float
    n_steps: int
    output_stride: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: object) -> object:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    raise TypeError(f"cannot serialise {type(value)!r}")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _tree_hashes(directory: Path, *, exclude: set[str] | None = None) -> dict[str, str]:
    omitted = set() if exclude is None else set(exclude)
    if not directory.exists():
        return {}
    return {
        str(path.relative_to(directory)): _sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name not in omitted
    }


def _tree_state(directory: Path) -> dict[str, object]:
    """Return a compact, deterministic digest of a complete result tree."""

    hashes = _tree_hashes(directory)
    digest = hashlib.sha256()
    for relative, value in sorted(hashes.items()):
        digest.update(relative.replace("\\", "/").encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    total_bytes = sum((directory / relative).stat().st_size for relative in hashes)
    return {
        "path": str(directory.resolve()),
        "file_count": len(hashes),
        "total_bytes": total_bytes,
        "tree_sha256": digest.hexdigest(),
    }


def _nonfinite_json_paths(value: object, location: str = "$") -> list[str]:
    issues: list[str] = []
    if isinstance(value, float) and not np.isfinite(value):
        issues.append(location)
    elif isinstance(value, dict):
        for key, child in value.items():
            issues.extend(_nonfinite_json_paths(child, f"{location}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            issues.extend(_nonfinite_json_paths(child, f"{location}[{index}]"))
    return issues


def _postrun_bundle_audit(output: Path) -> tuple[dict[str, object], Path, Path]:
    """Audit the completed manifest without mutating any hashed artifact.

    The two audit files are intentionally detached from ``artifact_hashes``:
    they contain the final manifest hash, so including them in that same
    manifest would create a circular checksum dependency.
    """

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_mismatches = []
    for name, expected in manifest["source_hashes"].items():
        path = Path(name)
        actual = _sha256(path) if path.exists() else "MISSING"
        if actual != expected:
            source_mismatches.append(
                {"path": name, "expected": expected, "actual": actual}
            )
    artifact_mismatches = []
    for name, expected in manifest["artifact_hashes"].items():
        path = output / name
        actual = _sha256(path) if path.exists() else "MISSING"
        if actual != expected:
            artifact_mismatches.append(
                {"path": name, "expected": expected, "actual": actual}
            )
    json_issues = []
    for path in sorted(output.rglob("*.json")):
        if path.name == "postrun_audit.json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            nonfinite = _nonfinite_json_paths(payload)
            if nonfinite:
                json_issues.append(
                    {"path": str(path.relative_to(output)), "nonfinite": nonfinite}
                )
        except Exception as error:
            json_issues.append(
                {
                    "path": str(path.relative_to(output)),
                    "parse_error": f"{type(error).__name__}: {error}",
                }
            )
    csv_issues = []
    for path in sorted(output.rglob("*.csv")):
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row_index, row in enumerate(csv.reader(handle), start=1):
                for column_index, cell in enumerate(row, start=1):
                    stripped = cell.strip()
                    if not stripped:
                        continue
                    try:
                        numeric = float(stripped)
                    except ValueError:
                        continue
                    if not np.isfinite(numeric):
                        csv_issues.append(
                            {
                                "path": str(path.relative_to(output)),
                                "row": row_index,
                                "column": column_index,
                                "value": stripped,
                            }
                        )
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "detached_from_manifest_reason": (
            "This audit records the final manifest hash and is excluded from "
            "that manifest to avoid a circular checksum dependency."
        ),
        "source_hash_mismatches": source_mismatches,
        "artifact_hash_mismatches": artifact_mismatches,
        "json_parse_or_nonfinite_issues": json_issues,
        "csv_nonfinite_issues": csv_issues,
        "frozen_outputs_unchanged": metrics["provenance"][
            "frozen_outputs_unchanged"
        ],
        "frozen_trees_present_and_nonempty": metrics["provenance"][
            "frozen_trees_present_and_nonempty"
        ],
        "all_executed_gates_passed": metrics["all_executed_gates_passed"],
    }
    audit["integrity_audit_passed"] = bool(
        not source_mismatches
        and not artifact_mismatches
        and not json_issues
        and not csv_issues
        and audit["frozen_outputs_unchanged"]
        and audit["frozen_trees_present_and_nonempty"]
    )
    audit["numerical_gates_passed"] = bool(
        audit["all_executed_gates_passed"]
    )
    audit["bundle_accepted"] = bool(
        audit["integrity_audit_passed"]
        and audit["numerical_gates_passed"]
    )
    json_path = output / "postrun_audit.json"
    _write_json(json_path, audit)
    markdown_path = output / "postrun_audit.md"
    markdown_path.write_text(
        "\n".join(
            [
                "# Detached post-run integrity audit",
                "",
                f"- Integrity audit: `{'PASS' if audit['integrity_audit_passed'] else 'FAIL'}`.",
                f"- Executed numerical gates: `{'PASS' if audit['numerical_gates_passed'] else 'FAIL'}`.",
                f"- Bundle accepted: `{'YES' if audit['bundle_accepted'] else 'NO'}`.",
                f"- Manifest SHA256: `{audit['manifest_sha256']}`.",
                f"- Source-hash mismatches: `{len(source_mismatches)}`.",
                f"- Artifact-hash mismatches: `{len(artifact_mismatches)}`.",
                f"- JSON parse/non-finite issues: `{len(json_issues)}`.",
                f"- CSV non-finite issues: `{len(csv_issues)}`.",
                f"- Frozen outputs unchanged: `{audit['frozen_outputs_unchanged']}`.",
                "- Frozen Exp14--17 trees present and non-empty: "
                f"`{audit['frozen_trees_present_and_nonempty']}`.",
                "",
                "This file and `postrun_audit.json` are detached from the manifest because",
                "they record the final manifest hash; hashing them into that same manifest",
                "would create a circular checksum dependency.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return audit, json_path, markdown_path


def _relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    numerator = float(np.linalg.norm(np.asarray(candidate) - np.asarray(reference)))
    denominator = max(float(np.linalg.norm(reference)), np.finfo(float).tiny)
    return numerator / denominator


def _analysis_mask(times_s: np.ndarray, requested_start_s: float) -> tuple[np.ndarray, float]:
    mask = np.asarray(times_s) >= requested_start_s
    if np.count_nonzero(mask) < 16:
        start_index = max(0, times_s.size // 2)
        mask = np.arange(times_s.size) >= start_index
    return mask, float(times_s[np.flatnonzero(mask)[0]])


def _one_sided_psd(
    times_s: np.ndarray,
    values: np.ndarray,
    analysis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Hann-periodogram on a uniform interpolation of saved output times."""

    selected_time = np.asarray(times_s)[analysis]
    selected = np.asarray(values)[analysis]
    if selected_time.size < 8:
        raise ValueError("at least eight analysis samples are required for a PSD")
    sample_dt = float(np.median(np.diff(selected_time)))
    uniform_time = np.arange(
        selected_time[0], selected_time[-1] + 0.25 * sample_dt, sample_dt
    )
    if selected.ndim == 1:
        uniform = np.interp(uniform_time, selected_time, selected)[None, :]
    elif selected.ndim == 2:
        uniform = np.vstack(
            [
                np.interp(uniform_time, selected_time, selected[:, index])
                for index in range(selected.shape[1])
            ]
        )
    else:
        raise ValueError("PSD input must be one- or two-dimensional")
    uniform -= np.mean(uniform, axis=1, keepdims=True)
    window = np.hanning(uniform.shape[1])
    sampling_frequency = 1.0 / sample_dt
    coefficients = np.fft.rfft(uniform * window[None, :], axis=1)
    psd = np.abs(coefficients) ** 2 / (
        sampling_frequency * float(np.sum(window * window))
    )
    if uniform.shape[1] % 2 == 0:
        psd[:, 1:-1] *= 2.0
    else:
        psd[:, 1:] *= 2.0
    frequencies = np.fft.rfftfreq(uniform.shape[1], d=sample_dt)
    return frequencies, psd[0] if selected.ndim == 1 else psd, sample_dt


def _maximum_high_wavenumber_fraction(field: np.ndarray) -> float:
    values = np.asarray(field, dtype=float)
    if values.shape[0] < 1 or values.shape[1] < 8:
        return 0.0
    window = np.hanning(values.shape[1])
    tapered = (values - np.mean(values, axis=1, keepdims=True)) * window[None, :]
    power = np.abs(np.fft.rfft(tapered, axis=1)) ** 2
    start = int(np.floor(0.8 * power.shape[1]))
    fractions = np.sum(power[:, start:], axis=1) / np.maximum(
        np.sum(power, axis=1), np.finfo(float).tiny
    )
    return float(np.max(fractions))


def _prepare_output(path: Path, overwrite: bool) -> None:
    resolved = path.resolve()
    if resolved.exists() and any(resolved.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"output is non-empty: {resolved}; pass --overwrite deliberately"
            )
        expected_parent = DEFAULT_OUTPUT_BASE.resolve()
        if resolved.parent != expected_parent:
            raise RuntimeError(
                "--overwrite is restricted to a direct child of the default "
                f"output base {expected_parent}"
            )
        shutil.rmtree(resolved)
    (resolved / RAW_DIRECTORY_NAME).mkdir(parents=True, exist_ok=True)


def _resolution_specs(
    config: ProductionConfig,
    selection: str,
) -> list[ResolutionSpec]:
    medium_steps = _aligned_steps(
        config.requested_duration_s,
        CoastalParameters(),
        config.medium_dt,
    )
    # Fine has exactly twice the medium step count.  Together with twice the
    # output stride this guarantees identical saved physical times.
    medium = ResolutionSpec(
        "medium",
        config.medium_n4,
        config.medium_dt,
        medium_steps,
        config.medium_output_stride,
    )
    fine = ResolutionSpec(
        "fine",
        config.fine_n4,
        config.fine_dt,
        2 * medium_steps,
        config.fine_output_stride,
    )
    if selection == "medium":
        return [medium]
    if selection == "fine":
        return [fine]
    return [medium, fine]


def _run_resolution(
    spec: ResolutionSpec,
    config: ProductionConfig,
    parameters: CoastalParameters,
) -> tuple[dict[str, object], object]:
    """Run one grid and retain only the reported 0--4 km physical field."""

    y, dy, n_computational = _grid_for_length(
        config.computational_length_m,
        parameters,
        short_n=spec.n4,
    )
    depth_ratio = coastal_depth_ratio_y(
        y,
        length_ref_m=parameters.lambda_ref_m,
        depth_ref_m=parameters.h_ref_m,
        offshore_depth_m=parameters.h_offshore_m,
        nearshore_depth_m=parameters.nearshore_depth_m,
        transition_start_m=parameters.transition_start_m,
        transition_end_m=parameters.transition_end_m,
        kind="beta_C7",
    )
    physical_points = spec.n4
    y_physical_m = y[:physical_points] * parameters.lambda_ref_m
    x_physical_m = config.physical_length_m - y_physical_m
    depth_physical_m = parameters.h_ref_m * depth_ratio[:physical_points]
    linear_config = LinearStudyConfig(random_seed=config.random_seed)
    sea_state, _, traces, lifting_metadata = _tma_inputs(
        parameters,
        linear_config,
        dy,
        spec.dt,
    )
    solver = CoastalHighOrderImplicitMidpointDABCSolver(
        y,
        depth_ratio,
        parameters.epsilon,
        parameters.mu,
        spec.dt,
        spec.n_steps,
    )
    initial = np.zeros_like(y)
    audit = ScreeningStepAudit(solver, parameters)
    audit.include_initial(solver.to_normalized(initial))
    started = time.perf_counter()
    times, surface, normalized, residuals = solver.run(
        initial,
        spec.output_stride,
        traces,
        step_diagnostic=audit,
    )
    runtime_s = time.perf_counter() - started
    times_s = np.asarray(times) * parameters.time_ref_s
    eta_physical_m = (
        np.asarray(surface[:, :physical_points], dtype=float)
        * parameters.a_ref_m
    )
    if not np.all(np.isfinite(eta_physical_m)):
        raise FloatingPointError(f"{spec.name} field contains non-finite values")
    trace_values_m = np.vstack(
        [
            parameters.a_ref_m
            * np.asarray([trace(value) for value in times], dtype=float)
            for trace in traces
        ]
    )
    boundary_truth_m = np.asarray(sea_state.truth_m(times_s), dtype=float)
    trace0_truth_error_m = trace_values_m[0] - boundary_truth_m
    boundary_row_error_m = eta_physical_m[:, 0] - boundary_truth_m
    gauge_indices = np.asarray(
        [
            int(round((config.physical_length_m - value) / (dy * parameters.lambda_ref_m)))
            for value in config.gauge_x_m
        ],
        dtype=int,
    )
    if np.any(gauge_indices < 0) or np.any(gauge_indices >= physical_points):
        raise RuntimeError("a gauge lies outside the reported physical grid")
    gauge_eta_m = eta_physical_m[:, gauge_indices]
    analysis, actual_analysis_start_s = _analysis_mask(
        times_s, config.analysis_start_s
    )
    hs_m = 4.0 * np.std(eta_physical_m[analysis], axis=0, ddof=0)
    spectral_frequency_hz, gauge_spectrum_m2_hz, spectrum_dt_s = _one_sided_psd(
        times_s, gauge_eta_m, analysis
    )
    boundary_frequency_hz, boundary_spectrum_m2_hz, boundary_spectrum_dt_s = (
        _one_sided_psd(times_s, boundary_truth_m, analysis)
    )
    if not np.array_equal(spectral_frequency_hz, boundary_frequency_hz):
        raise RuntimeError("gauge and boundary spectral grids differ")
    fixed_point_counts = np.asarray(
        solver.fixed_point_iteration_counts, dtype=np.int16
    )
    summary = {
        "name": spec.name,
        "n4_physical": spec.n4,
        "n10_computational": n_computational,
        "dimensionless_dt": spec.dt,
        "physical_dt_s": spec.dt * parameters.time_ref_s,
        "n_steps": spec.n_steps,
        "output_stride": spec.output_stride,
        "saved_output_count": int(times_s.size),
        "actual_final_time_s": float(times_s[-1]),
        "physical_grid_spacing_m": dy * parameters.lambda_ref_m,
        "runtime_s": runtime_s,
        "all_fields_finite": bool(
            np.all(np.isfinite(surface)) and np.all(np.isfinite(normalized))
        ),
        "maximum_abs_eta_m": float(np.max(np.abs(eta_physical_m))),
        "maximum_abs_input_boundary_error_m": float(
            np.max(np.abs(boundary_row_error_m))
        ),
        "maximum_abs_incident_trace0_minus_TMA_truth_m": float(
            np.max(np.abs(trace0_truth_error_m))
        ),
        "maximum_six_boundary_residual": float(np.max(np.abs(residuals))),
        "maximum_highest_20pct_wavenumber_fraction": (
            _maximum_high_wavenumber_fraction(eta_physical_m[analysis])
        ),
        "analysis_start_requested_s": config.analysis_start_s,
        "analysis_start_actual_s": actual_analysis_start_s,
        "Hs_definition": "4 times the temporal standard deviation after analysis start",
        "spectrum_saved_output_dt_s": spectrum_dt_s,
        "boundary_spectrum_saved_output_dt_s": boundary_spectrum_dt_s,
        "fixed_point": solver.fixed_point_summary(),
        "step_audit": audit.summary(),
        "TMA_sea_state": sea_state.metadata(),
        "incident_lifting": lifting_metadata,
        "incident_trace_contract": (
            "trace 0 is the common physical TMA truth; traces 1 and 2 are "
            "resolution-specific discrete C6/C4 matched lifting traces"
        ),
    }
    arrays = {
        "y_computational_m": y * parameters.lambda_ref_m,
        "depth_computational_m": depth_ratio * parameters.h_ref_m,
        "x_physical_m": x_physical_m,
        "z_bottom_physical_m": -depth_physical_m,
        "times_s": times_s,
        "eta_physical_m": eta_physical_m,
        "boundary_truth_m": boundary_truth_m,
        "incident_traces_m": trace_values_m,
        "boundary_row_error_m": boundary_row_error_m,
        "incident_trace0_minus_truth_m": trace0_truth_error_m,
        "gauge_indices": gauge_indices,
        "gauge_eta_m": gauge_eta_m,
        "Hs_m": hs_m,
        "spectrum_frequency_hz": spectral_frequency_hz,
        "boundary_spectrum_m2_hz": boundary_spectrum_m2_hz,
        "gauge_spectrum_m2_hz": gauge_spectrum_m2_hz,
        "six_boundary_residuals": np.asarray(residuals),
        "fixed_point_iterations": fixed_point_counts,
        "analysis_mask": analysis,
    }
    del surface, normalized, solver
    return {"summary": summary, "arrays": arrays}, sea_state


def _compare_grids(
    medium: dict[str, object],
    fine: dict[str, object],
    config: ProductionConfig,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    ma = medium["arrays"]
    fa = fine["arrays"]
    if not np.allclose(ma["times_s"], fa["times_s"], rtol=0.0, atol=2.0e-11):
        raise RuntimeError("medium and fine saved times are not aligned")
    if not np.allclose(
        ma["x_physical_m"], fa["x_physical_m"][::2], rtol=0.0, atol=2.0e-10
    ):
        raise RuntimeError("fine physical grid is not nested on the medium grid")
    eta_medium = ma["eta_physical_m"]
    eta_fine_on_medium = fa["eta_physical_m"][:, ::2]
    difference = eta_medium - eta_fine_on_medium
    analysis = ma["analysis_mask"]
    hs_medium = ma["Hs_m"]
    hs_fine = fa["Hs_m"][::2]
    hs_difference = hs_medium - hs_fine
    spatial_rms_m = np.sqrt(np.mean(difference[analysis] ** 2, axis=0))
    rms_history_m = np.sqrt(np.mean(difference**2, axis=1))
    fine_rms_history_m = np.sqrt(np.mean(eta_fine_on_medium**2, axis=1))
    normalized_rms_history = rms_history_m / np.maximum(
        fine_rms_history_m,
        max(float(np.sqrt(np.mean(eta_fine_on_medium[analysis] ** 2))), 1.0e-14),
    )
    gauge_difference = ma["gauge_eta_m"] - fa["gauge_eta_m"]
    gauge_rows: list[dict[str, object]] = []
    for index, gauge_x in enumerate(config.gauge_x_m):
        candidate = ma["gauge_eta_m"][analysis, index]
        reference = fa["gauge_eta_m"][analysis, index]
        candidate_rms = float(np.sqrt(np.mean(candidate * candidate)))
        reference_rms = float(np.sqrt(np.mean(reference * reference)))
        signal_present = max(candidate_rms, reference_rms) >= 1.0e-8
        if signal_present and np.std(candidate) > 0.0 and np.std(reference) > 0.0:
            correlation = float(np.corrcoef(candidate, reference)[0, 1])
        else:
            correlation = None
        gauge_rows.append(
            {
                "x_m": gauge_x,
                "signal_status": "arrived" if signal_present else "not_arrived",
                "medium_fine_RMSE_m": float(
                    np.sqrt(np.mean((candidate - reference) ** 2))
                ),
                "medium_fine_relative_L2": (
                    _relative_l2(candidate, reference) if signal_present else None
                ),
                "medium_fine_correlation": correlation,
                "Hs_medium_m": float(4.0 * np.std(candidate, ddof=0)),
                "Hs_fine_m": float(4.0 * np.std(reference, ddof=0)),
            }
        )
    summary = {
        "saved_times_aligned": True,
        "nested_physical_grids": True,
        "field_spacetime_relative_L2": _relative_l2(
            eta_medium[analysis], eta_fine_on_medium[analysis]
        ),
        "field_spacetime_RMSE_m": float(
            np.sqrt(np.mean(difference[analysis] ** 2))
        ),
        "maximum_absolute_field_difference_m": float(
            np.max(np.abs(difference[analysis]))
        ),
        "Hs_profile_relative_L2": _relative_l2(hs_medium, hs_fine),
        "maximum_absolute_Hs_difference_m": float(np.max(np.abs(hs_difference))),
        "maximum_normalized_RMS_history": float(
            np.max(normalized_rms_history[analysis])
        ),
        "gauge_metrics": gauge_rows,
        "reference_scope": (
            "same-model fine-grid self-convergence reference; not exact truth"
        ),
    }
    arrays = {
        "eta_fine_on_medium_m": eta_fine_on_medium,
        "eta_medium_minus_fine_m": difference,
        "Hs_fine_on_medium_m": hs_fine,
        "Hs_medium_minus_fine_m": hs_difference,
        "spatial_RMS_difference_m": spatial_rms_m,
        "RMS_difference_history_m": rms_history_m,
        "normalized_RMS_difference_history": normalized_rms_history,
        "gauge_medium_minus_fine_m": gauge_difference,
    }
    return summary, arrays


def _save_replot_npz(
    path: Path,
    runs: dict[str, dict[str, object]],
    comparison_arrays: dict[str, np.ndarray] | None,
    sea_state: object,
    config: ProductionConfig,
) -> None:
    preferred_name = "fine" if "fine" in runs else "medium"
    preferred = runs[preferred_name]["arrays"]
    payload: dict[str, np.ndarray] = {
        "times_s": preferred["times_s"],
        "gauge_x_m": np.asarray(config.gauge_x_m),
        "boundary_input_truth_m": preferred["boundary_truth_m"],
        "target_tma_frequency_hz": sea_state.frequencies_hz,
        "target_tma_spectrum_m2_hz": sea_state.spectrum_m2_hz,
        "empirical_spectrum_frequency_hz": preferred["spectrum_frequency_hz"],
        "boundary_empirical_spectrum_m2_hz": preferred["boundary_spectrum_m2_hz"],
        "reported_physical_x_limits_m": np.asarray((0.0, 4000.0)),
        "computational_y_limits_m": np.asarray((0.0, 10000.0)),
        "guard_y_limits_m": np.asarray((4000.0, 10000.0)),
    }
    for name, run in runs.items():
        arrays = run["arrays"]
        payload.update(
            {
                f"x_physical_m_{name}": arrays["x_physical_m"],
                f"x_physical_ascending_m_{name}": arrays["x_physical_m"][::-1],
                f"z_bottom_physical_m_{name}": arrays["z_bottom_physical_m"],
                f"eta_{name}_m": arrays["eta_physical_m"],
                f"boundary_truth_{name}_m": arrays["boundary_truth_m"],
                f"incident_traces_{name}_m": arrays["incident_traces_m"],
                f"gauge_eta_{name}_m": arrays["gauge_eta_m"],
                f"Hs_{name}_m": arrays["Hs_m"],
                f"gauge_spectrum_{name}_m2_hz": arrays["gauge_spectrum_m2_hz"],
                f"six_boundary_residuals_{name}": arrays["six_boundary_residuals"],
                f"fixed_point_iterations_{name}": arrays["fixed_point_iterations"],
                f"analysis_mask_{name}": arrays["analysis_mask"],
                f"y_computational_m_{name}": arrays["y_computational_m"],
                f"depth_computational_m_{name}": arrays["depth_computational_m"],
                f"eta_column_order_note_{name}": np.asarray(
                    "eta columns match x_physical_m (descending 4000 to 0 m); "
                    "reverse columns to use x_physical_ascending_m"
                ),
            }
        )
    if comparison_arrays is not None:
        payload.update(comparison_arrays)
    np.savez_compressed(path, **payload)


def _build_gates(
    runs: dict[str, dict[str, object]],
    comparison: dict[str, object] | None,
    config: ProductionConfig,
    smoke: bool,
    frozen_unchanged: bool,
    frozen_present_and_nonempty: bool,
    sources_unchanged: bool,
) -> dict[str, bool | None]:
    summaries = [run["summary"] for run in runs.values()]
    actual_end = min(float(summary["actual_final_time_s"]) for summary in summaries)
    gates: dict[str, bool | None] = {
        "requested_duration_reached": actual_end + 0.05 >= config.requested_duration_s,
        "all_fields_finite": all(summary["all_fields_finite"] for summary in summaries),
        "medium_fine_saved_times_aligned": (
            None if comparison is None else bool(comparison["saved_times_aligned"])
        ),
        "input_boundary_error_below_1e-10_m": max(
            summary["maximum_abs_input_boundary_error_m"] for summary in summaries
        )
        < 1.0e-10,
        "incident_trace0_matches_common_TMA_truth_below_1e-12_m": max(
            summary["maximum_abs_incident_trace0_minus_TMA_truth_m"]
            for summary in summaries
        )
        < 1.0e-12,
        "six_boundary_residual_below_1e-10": max(
            summary["maximum_six_boundary_residual"] for summary in summaries
        )
        < 1.0e-10,
        "true_bordered_equation_scaled_residual_below_1e-11": max(
            summary["fixed_point"]["maximum_scaled_equation_residual"]
            for summary in summaries
        )
        < 1.0e-11,
        "interior_recurrence_residual_below_1e-10": max(
            summary["step_audit"]["maximum_interior_recurrence_residual"]
            for summary in summaries
        )
        < 1.0e-10,
        "fixed_point_maximum_iterations_below_8": max(
            summary["fixed_point"]["maximum_iterations"] for summary in summaries
        )
        <= 8,
        "nonlinear_CFL_below_0p2": max(
            summary["step_audit"]["maximum_nonlinear_CFL"]
            for summary in summaries
        )
        < 0.2,
        # Before the wave has crossed much of the ROI, a spatial FFT mostly
        # measures the sharp wet/no-arrival footprint and is not an instability
        # diagnostic.  Therefore the production threshold is intentionally not
        # assessed by the 30 s smoke run.
        "highest_20pct_wavenumber_fraction_below_1e-4": (
            None
            if smoke
            else max(
                summary["maximum_highest_20pct_wavenumber_fraction"]
                for summary in summaries
            )
            < 1.0e-4
        ),
        "weak_nonlinearity_max_abs_eta_over_h_below_0p1": max(
            summary["step_audit"]["maximum_abs_eta_over_local_depth"]
            for summary in summaries
        )
        < 0.1,
        "frozen_Exp14_to_Exp17_outputs_unchanged": frozen_unchanged,
        "frozen_Exp14_to_Exp17_trees_present_and_nonempty": (
            frozen_present_and_nonempty
        ),
        "imported_solver_sources_unchanged_during_run": sources_unchanged,
        "production_medium_fine_field_relative_L2_below_1_percent": (
            None
            if smoke or comparison is None
            else comparison["field_spacetime_relative_L2"] < 0.01
        ),
        "production_medium_fine_Hs_relative_L2_below_0p5_percent": (
            None
            if smoke or comparison is None
            else comparison["Hs_profile_relative_L2"] < 0.005
        ),
    }
    return gates


def _report_text(metrics: dict[str, object], command: str) -> str:
    lines = [
        "# Final physical-time coastal vKdV production bundle",
        "",
        "## Model and domain interpretation",
        "",
        "The numerical solution directly uses `CoastalHighOrderImplicitMidpointDABCSolver` with",
        "the entropy-split C6 nonlinear drift and fully implicit midpoint update. TMA truth is",
        "imposed at `x=4 km` (`y=0`). The reported physical coordinate is `x=4 km-y`.",
        "Only `x in [0,4] km` is shown. The extension `y in [4,10] km` is a numerical guard",
        "that delays the artificial outflow boundary and is not a physical forecast.",
        "At `x=0`, `h=5 m` and the shelf continues; this is not a shoreline or run-up model.",
        "For each resolution, incident trace 0 is the same physical TMA truth. Incident traces",
        "1 and 2 are generated separately by the matched discrete C6/C4 lifting and therefore",
        "are intentionally grid/time-step dependent.",
        "",
        "## Run",
        "",
        f"- Mode: `{metrics['mode']}`.",
        f"- Requested/actual duration: `{metrics['config']['requested_duration_s']:.3f}` / "
        f"`{metrics['actual_final_time_s']:.3f}` s.",
        f"- Resolution selection: `{metrics['resolution_selection']}`.",
        f"- TMA seed: `{metrics['config']['random_seed']}`; `Tp=15 s`, `Hs=0.3 m`.",
        f"- Reproduction command: `{command}`.",
        "",
        "## Grid results",
        "",
    ]
    for name, run in metrics["runs"].items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"- `N4={run['n4_physical']}`, `N10={run['n10_computational']}`, "
                f"`dt={run['dimensionless_dt']}`; runtime `{run['runtime_s']:.2f} s`.",
                f"- max `|eta|={run['maximum_abs_eta_m']:.4e} m`; max six-row residual "
                f"`{run['maximum_six_boundary_residual']:.3e}`.",
                f"- max scaled bordered-equation residual "
                f"`{run['fixed_point']['maximum_scaled_equation_residual']:.3e}`; max fixed-point "
                f"iterations `{run['fixed_point']['maximum_iterations']}`.",
                f"- max nonlinear CFL `{run['step_audit']['maximum_nonlinear_CFL']:.3e}`; "
                f"max high-k fraction `{run['maximum_highest_20pct_wavenumber_fraction']:.3e}`.",
                "",
            ]
        )
    comparison = metrics.get("medium_fine_comparison")
    if comparison is not None:
        lines.extend(
            [
                "## Medium--fine self-convergence",
                "",
                f"- Field space-time relative L2: `{comparison['field_spacetime_relative_L2']:.3e}`.",
                f"- Hs-profile relative L2: `{comparison['Hs_profile_relative_L2']:.3e}`.",
                f"- Maximum absolute field difference: "
                f"`{comparison['maximum_absolute_field_difference_m']:.3e} m`.",
                "- The fine grid is a same-model numerical reference, not exact truth.",
                "",
            ]
        )
    lines.extend(["## Acceptance gates", ""])
    for name, value in metrics["acceptance_gates"].items():
        state = "NOT RUN" if value is None else ("PASS" if value else "FAIL")
        lines.append(f"- {state}: `{name}`")
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "A smoke bundle verifies code paths and provenance only; its medium--fine errors are",
            "not paper-level production evidence. The formal 1800 s bundle must pass the two",
            "production comparison gates before quantitative use. The vKdV approximation also",
            "does not resolve shoreline wetting/drying, breaking or run-up.",
            "Final checksum and JSON/CSV finite checks are written to the detached",
            "`postrun_audit.md` and `postrun_audit.json` after the manifest is closed.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-base",
        type=Path,
        default=DEFAULT_OUTPUT_BASE,
        help="base directory; each run is placed in a named child directory",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="child directory name (defaults to smoke or production_1800s)",
    )
    parser.add_argument(
        "--resolution",
        choices=("both", "medium", "fine"),
        default="both",
    )
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--analysis-start-s", type=float, default=None)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.duration_s is not None and args.duration_s <= 0.0:
        parser.error("--duration-s must be positive")
    duration_s = (
        float(args.duration_s)
        if args.duration_s is not None
        else (30.0 if args.smoke else 1800.0)
    )
    if args.smoke:
        analysis_start_s = (
            float(args.analysis_start_s)
            if args.analysis_start_s is not None
            else 10.0
        )
        config = ProductionConfig(
            requested_duration_s=duration_s,
            medium_n4=257,
            fine_n4=513,
            medium_dt=0.008,
            fine_dt=0.004,
            medium_output_stride=10,
            fine_output_stride=20,
            analysis_start_s=analysis_start_s,
            random_seed=args.seed,
        )
    else:
        analysis_start_s = (
            float(args.analysis_start_s)
            if args.analysis_start_s is not None
            else min(600.0, 0.55 * duration_s)
        )
        config = ProductionConfig(
            requested_duration_s=duration_s,
            analysis_start_s=analysis_start_s,
            random_seed=args.seed,
        )
    if not 0.0 <= config.analysis_start_s < config.requested_duration_s:
        parser.error("analysis start must lie in [0, duration)")
    run_name = args.run_name or ("smoke" if args.smoke else "production_1800s")
    if not run_name or Path(run_name).name != run_name:
        parser.error("--run-name must be one plain directory name")
    output = args.output_base.resolve() / run_name
    _prepare_output(output, args.overwrite)
    raw = output / RAW_DIRECTORY_NAME

    parameters = CoastalParameters()
    ramp_s = 10.0 if args.smoke else parameters.boundary_ramp_s
    parameters = replace(
        parameters,
        final_time=config.requested_duration_s / parameters.time_ref_s,
        random_seed=config.random_seed,
        boundary_ramp_s=ramp_s,
        statistics_start_s=config.analysis_start_s,
    )
    specs = _resolution_specs(config, args.resolution)
    source_paths = (
        Path(__file__).resolve(),
        PROJECT_DIR / "high_order_implicit_midpoint_candidate.py",
        PROJECT_DIR / "high_order_nonlinear_candidates.py",
        PROJECT_DIR / "high_order_nonlinear_candidate_screening.py",
        PROJECT_DIR / "high_order_variable_depth_dabc.py",
        PROJECT_DIR / "high_order_variable_depth_dabc_study.py",
        PROJECT_DIR / "high_order_matched_dabc.py",
        PROJECT_DIR / "high_order_incident_lifting.py",
        PROJECT_DIR / "sea_state_boundary.py",
        PROJECT_DIR / "pde_core.py",
        PROJECT_DIR / "transparent_boundary_vkdv.py",
    )
    source_hashes_before = {str(path): _sha256(path) for path in source_paths}
    frozen_directories = {
        "Exp14": PROJECT_DIR / "results" / "transparent_boundary" / "high_order_variable_depth_dabc",
        "Exp15": PROJECT_DIR / "results" / "transparent_boundary" / "high_order_nonlinear_vkdv",
        "Exp16_split": PROJECT_DIR / "results" / "transparent_boundary" / "nonlinear_candidate_screening_exp16",
        "Exp16_CNAB3": PROJECT_DIR / "results" / "transparent_boundary" / "nonlinear_cnab3_candidate_screening",
        "Exp17": PROJECT_DIR / "results" / "transparent_boundary" / "nonlinear_implicit_midpoint_candidate_screening",
    }
    frozen_before = {
        name: _tree_state(path) for name, path in frozen_directories.items()
    }
    started = time.perf_counter()
    runs: dict[str, dict[str, object]] = {}
    sea_state = None
    for spec in specs:
        print(
            f"Production driver: {spec.name}, N4={spec.n4}, dt={spec.dt}, "
            f"steps={spec.n_steps}, duration={config.requested_duration_s:.1f}s",
            flush=True,
        )
        result, current_sea_state = _run_resolution(spec, config, parameters)
        runs[spec.name] = result
        if sea_state is None:
            sea_state = current_sea_state
        elif not np.array_equal(
            sea_state.phases_rad, current_sea_state.phases_rad
        ):
            raise RuntimeError("medium and fine TMA phase realisations differ")
    if sea_state is None:
        raise RuntimeError("no resolution was selected")

    comparison = None
    comparison_arrays = None
    if "medium" in runs and "fine" in runs:
        comparison, comparison_arrays = _compare_grids(
            runs["medium"], runs["fine"], config
        )
        _write_csv(raw / "gauge_medium_fine_metrics.csv", comparison["gauge_metrics"])

    replot_npz = raw / "production_replot_data.npz"
    _save_replot_npz(
        replot_npz,
        runs,
        comparison_arrays,
        sea_state,
        config,
    )

    run_rows = []
    hs_rows = []
    for name, run in runs.items():
        summary = run["summary"]
        run_rows.append(
            {
                "resolution": name,
                "N4": summary["n4_physical"],
                "N10": summary["n10_computational"],
                "dimensionless_dt": summary["dimensionless_dt"],
                "physical_dt_s": summary["physical_dt_s"],
                "n_steps": summary["n_steps"],
                "saved_output_count": summary["saved_output_count"],
                "runtime_s": summary["runtime_s"],
                "maximum_abs_eta_m": summary["maximum_abs_eta_m"],
                "maximum_six_boundary_residual": summary[
                    "maximum_six_boundary_residual"
                ],
                "maximum_scaled_equation_residual": summary["fixed_point"][
                    "maximum_scaled_equation_residual"
                ],
                "maximum_fixed_point_iterations": summary["fixed_point"][
                    "maximum_iterations"
                ],
                "maximum_nonlinear_CFL": summary["step_audit"][
                    "maximum_nonlinear_CFL"
                ],
                "maximum_high_k_fraction": summary[
                    "maximum_highest_20pct_wavenumber_fraction"
                ],
            }
        )
        x_values = run["arrays"]["x_physical_m"]
        hs_values = run["arrays"]["Hs_m"]
        for index in range(x_values.size):
            hs_rows.append(
                {
                    "resolution": name,
                    "x_m": float(x_values[index]),
                    "Hs_m": float(hs_values[index]),
                }
            )
    _write_csv(raw / "run_summary.csv", run_rows)
    _write_csv(raw / "Hs_profiles.csv", hs_rows)

    source_hashes_after = {str(path): _sha256(path) for path in source_paths}
    frozen_after = {
        name: _tree_state(path) for name, path in frozen_directories.items()
    }
    sources_unchanged = source_hashes_before == source_hashes_after
    frozen_unchanged = frozen_before == frozen_after
    frozen_present_and_nonempty = all(
        int(state["file_count"]) > 0
        for collection in (frozen_before, frozen_after)
        for state in collection.values()
    )
    gates = _build_gates(
        runs,
        comparison,
        config,
        args.smoke,
        frozen_unchanged,
        frozen_present_and_nonempty,
        sources_unchanged,
    )
    _write_csv(
        raw / "acceptance_gates.csv",
        [
            {
                "gate": name,
                "status": (
                    "NOT_RUN" if value is None else ("PASS" if value else "FAIL")
                ),
            }
            for name, value in gates.items()
        ],
    )
    actual_final_time_s = min(
        float(run["summary"]["actual_final_time_s"]) for run in runs.values()
    )
    runtime_s = time.perf_counter() - started
    command = " ".join([Path(__file__).name, *sys.argv[1:]])
    metrics: dict[str, object] = {
        "study": "final physical-time entropy-C6 implicit-midpoint coastal vKdV",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "smoke_workflow_only" if args.smoke else "production",
        "command": command,
        "resolution_selection": args.resolution,
        "direct_solver_class": (
            "high_order_implicit_midpoint_candidate."
            "CoastalHighOrderImplicitMidpointDABCSolver"
        ),
        "equation": (
            "(I-dt L/2)v[n+1]=(I+dt L/2)v[n]+dt*"
            "N((v[n]+v[n+1])/2)"
        ),
        "nonlinearity": "S[-gamma/3*(u D1u + D1(u^2))]",
        "coordinate_contract": {
            "shoreward_solver_coordinate": "y",
            "reported_physical_coordinate": "x=4000 m-y",
            "TMA_truth_input": "x=4 km (y=0)",
            "reported_interval_m": [0.0, 4000.0],
            "computational_interval_y_m": [0.0, 10000.0],
            "numerical_guard_y_m": [4000.0, 10000.0],
            "x_zero_statement": (
                "h=5 m reporting limit followed by a constant-depth numerical "
                "shelf; not a shoreline and not a run-up boundary"
            ),
            "plot_contract": "the y=4--10 km numerical guard is not plotted",
            "stored_array_order": (
                "x_physical_m is descending from 4000 to 0 m and matches eta "
                "columns; x_physical_ascending_m is also stored for plotting"
            ),
        },
        "incident_trace_contract": {
            "trace_0": (
                "common physical TMA truth eta(x=4 km,t), identical between grids"
            ),
            "traces_1_and_2": (
                "resolution-specific matched discrete C6/C4 incident lifting; "
                "not expected to be numerically identical between medium and fine"
            ),
        },
        "config": asdict(config),
        "parameters": asdict(parameters),
        "actual_final_time_s": actual_final_time_s,
        "TMA_sea_state": sea_state.metadata(),
        "runs": {name: run["summary"] for name, run in runs.items()},
        "medium_fine_comparison": comparison,
        "acceptance_gates": gates,
        "all_executed_gates_passed": all(
            value for value in gates.values() if value is not None
        ),
        "runtime_s": runtime_s,
        "raw_replot_npz": str(replot_npz),
        "provenance": {
            "source_hashes_before": source_hashes_before,
            "source_hashes_after": source_hashes_after,
            "sources_unchanged_during_run": sources_unchanged,
            "frozen_Exp14_to_Exp17_before": frozen_before,
            "frozen_Exp14_to_Exp17_after": frozen_after,
            "frozen_outputs_unchanged": frozen_unchanged,
            "frozen_trees_present_and_nonempty": frozen_present_and_nonempty,
        },
    }
    metrics_path = output / "metrics.json"
    _write_json(metrics_path, metrics)
    report_path = output / "production_run_report.md"
    report_path.write_text(_report_text(metrics, command), encoding="utf-8")
    artifact_hashes = _tree_hashes(output, exclude={"manifest.json"})
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": metrics["mode"],
        "command": command,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "source_hashes": source_hashes_after,
        "frozen_Exp14_to_Exp17_tree_state": frozen_after,
        "artifact_hashes": artifact_hashes,
        "detached_postrun_audit": {
            "files": ["postrun_audit.json", "postrun_audit.md"],
            "included_in_artifact_hashes": False,
            "reason": (
                "the detached audit records this final manifest hash and is "
                "excluded to avoid circular checksum dependency"
            ),
        },
        "manifest_self_hash": "excluded by construction",
    }
    manifest_path = output / "manifest.json"
    _write_json(manifest_path, manifest)
    postrun_audit, audit_json_path, audit_markdown_path = _postrun_bundle_audit(
        output
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "mode": metrics["mode"],
                "all_executed_gates_passed": metrics[
                    "all_executed_gates_passed"
                ],
                "runtime_s": runtime_s,
                "metrics_sha256": _sha256(metrics_path),
                "manifest_sha256": _sha256(manifest_path),
                "postrun_integrity_audit_passed": postrun_audit[
                    "integrity_audit_passed"
                ],
                "postrun_bundle_accepted": postrun_audit["bundle_accepted"],
                "postrun_audit_json": str(audit_json_path),
                "postrun_audit_markdown": str(audit_markdown_path),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
