"""Shared infrastructure for the final stochastic-closure chain.
Holds the coarse-grid model context (grid, bathymetry, boundary lifting,
solver factory), the closure run configuration, and the official-baseline
loader.  All final entry scripts (EKI, validation, cross-seed, sweep,
scoring, comparison) import from here; the deterministic core settings come
from :mod:`sde_closure_config`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pde_core import CoastalParameters
from high_order_variable_depth_dabc import coastal_depth_ratio_y
from high_order_variable_depth_dabc_study import (
    StudyConfig as LinearStudyConfig,
    _aligned_steps,
    _grid_for_length,
    _tma_inputs,
)
from sde_closure_core import StochasticImplicitMidpointDABCSolver
import sde_closure_config as closure_config

PROJECT_DIR = Path(__file__).resolve().parent
BASELINE_NPZ = closure_config.BASELINE_NPZ


@dataclass(frozen=True)
class ModelAConfig:
    """Coarse-closure run controls (name kept for script compatibility)."""

    requested_duration_s: float = 1800.0
    physical_length_m: float = 4000.0
    computational_length_m: float = 10000.0
    coarse_n4: int = closure_config.COARSE_N4
    coarse_dt: float = closure_config.COARSE_DT
    output_stride: int = closure_config.OUTPUT_STRIDE
    analysis_start_s: float = 600.0
    boundary_seed: int = 20260718
    noise_seed_base: int = 20260725
    n_paths: int = 8
    n_modes: int = 12
    frequency_min_hz: float = 0.035
    frequency_max_hz: float = 0.100
    initial_global_scale: float = 0.016
    calibration_shape_gauge_x_m: float = 1000.0
    gauge_x_m: tuple[float, ...] = (
        0.0,
        500.0,
        1000.0,
        2000.0,
        3000.0,
        4000.0,
    )


CoarseClosureConfig = ModelAConfig


def _write_json(path: Path, payload: object) -> None:
    def default(value: object) -> object:
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"unserialisable type {type(value)!r}")

    path.write_text(
        json.dumps(payload, indent=2, default=default), encoding="utf-8"
    )


class CoarseModelContext:
    """Shared grid, bathymetry, lifting, and solver factory."""

    def __init__(self, config: ModelAConfig) -> None:
        self.config = config
        self.parameters = CoastalParameters()
        self.y, self.dy, self.n_computational = _grid_for_length(
            config.computational_length_m,
            self.parameters,
            short_n=config.coarse_n4,
        )
        self.depth_ratio = coastal_depth_ratio_y(
            self.y,
            length_ref_m=self.parameters.lambda_ref_m,
            depth_ref_m=self.parameters.h_ref_m,
            offshore_depth_m=self.parameters.h_offshore_m,
            nearshore_depth_m=self.parameters.nearshore_depth_m,
            transition_start_m=self.parameters.transition_start_m,
            transition_end_m=self.parameters.transition_end_m,
            kind="beta_C7",
        )
        self.n_steps = _aligned_steps(
            config.requested_duration_s, self.parameters, config.coarse_dt
        )
        with closure_config.template():
            _, _, self.traces, self.lifting_metadata = _tma_inputs(
                self.parameters,
                LinearStudyConfig(random_seed=config.boundary_seed),
                self.dy,
                config.coarse_dt,
            )
        self.y_physical_m = (
            self.y[: config.coarse_n4] * self.parameters.lambda_ref_m
        )
        self.x_physical_m = config.physical_length_m - self.y_physical_m
        # Green normalisation scale S = d^{1/4} mapping surface -> state.
        self.surface_to_green = self.depth_ratio**0.25

    def make_solver(self) -> StochasticImplicitMidpointDABCSolver:
        with closure_config.template():
            return StochasticImplicitMidpointDABCSolver(
                self.y,
                self.depth_ratio,
                self.parameters.epsilon,
                self.parameters.mu,
                self.config.coarse_dt,
                self.n_steps,
            )

    def run_path(self, noise) -> dict[str, object]:
        solver = self.make_solver()
        started = time.perf_counter()
        times, surface, _, residuals = solver.run_stochastic(
            np.zeros_like(self.y),
            self.config.output_stride,
            self.traces,
            noise_increment=noise,
        )
        runtime_s = time.perf_counter() - started
        times_s = np.asarray(times) * self.parameters.time_ref_s
        eta_m = (
            np.asarray(surface[:, : self.config.coarse_n4], dtype=float)
            * self.parameters.a_ref_m
        )
        if not np.all(np.isfinite(eta_m)):
            raise FloatingPointError("stochastic path went non-finite")
        energy_m3 = np.trapezoid(eta_m**2, self.y_physical_m, axis=1)
        return {
            "times_s": times_s,
            "eta_m": eta_m,
            "energy_m3": energy_m3,
            "runtime_s": runtime_s,
            "max_boundary_residual": float(np.max(np.abs(residuals))),
            "fixed_point": solver.fixed_point_summary(),
        }


def load_baseline(config: ModelAConfig) -> dict[str, np.ndarray]:
    """Load the official coarse deterministic baseline bundle."""

    if not BASELINE_NPZ.exists():
        raise FileNotFoundError(
            f"official baseline missing: {BASELINE_NPZ}; run "
            "sde_closure_v3_baseline.py first"
        )
    data = np.load(BASELINE_NPZ, allow_pickle=True)
    required = (
        "times_s",
        "y_physical_m",
        "eta_coarse_m",
        "gauge_frequency_hz",
        "gauge_psd_error_m2_hz",
        "error_std_profile_fine_m",
        "y_ref_fine_m",
        "hs_fine_ref_m",
        "gauge_x_m",
        "energy_m3",
    )
    missing = [key for key in required if key not in data.files]
    if missing:
        raise KeyError(f"baseline bundle lacks {missing}")
    baseline = {key: np.asarray(data[key]) for key in required}
    if baseline["y_physical_m"].size != config.coarse_n4:
        raise ValueError(
            "baseline grid does not match the configured coarse grid"
        )
    return baseline


def gauge_columns(
    y_physical_m: np.ndarray,
    gauge_x_m: np.ndarray,
    physical_length_m: float,
) -> np.ndarray:
    gauge_y = physical_length_m - np.asarray(gauge_x_m, dtype=float)
    return np.asarray(
        [int(np.argmin(np.abs(y_physical_m - value))) for value in gauge_y],
        dtype=int,
    )
