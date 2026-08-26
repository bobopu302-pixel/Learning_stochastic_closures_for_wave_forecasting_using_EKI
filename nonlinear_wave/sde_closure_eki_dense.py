"""Dense-observation EKI experiment (frozen chain untouched).

Origin: 3. KDV_nonlinear_case/sde_closure_eki_dense.py
Changes vs origin (numerics untouched):
* the legacy standalone main(), its CONFIGS table and the __main__
  hook DELETED: they contained figure code and are never called by
  the 2026-08-23 spec chain (the sw_* drivers import this module as
  a library); imports and colour constants used only by main()
  removed with it;
* every symbol the drivers import is kept: DENSE_X_M, N_DENSE,
  CONCENTRATED_KNOT_Y_M, the SDE_DENSE_KNOTS import-time knot patch,
  compute_statistics_dense, truth_statistics_and_blocks_dense,
  build_gamma_dense, _init_worker_dense, forward_statistics_dense.

Tests whether the closure's spatial information bottleneck is the
observation vector: the standard 44-dim vector samples the Hs profile at
8 stations and the baseline-deviation rms at 5 gauges (500--1000 m
spacing), while the sigma(x) field it is supposed to identify lives on a
~200-500 m error-production structure.  This driver enriches BOTH
profile blocks to one station every 100 m (40 each, q=111) with
block-normalised Gamma (each dense block's total precision mass equals
the original block's, so denser sampling refines the spatial SHAPE
information without out-voting the other blocks).

Concentrated knot placement is injected by patching the frozen module's
N_SPLINE_KNOTS / SPLINE_KNOT_Y_M at import time, driven by the
SDE_DENSE_KNOTS env var so spawned workers apply the same patch.

This module is a library in this release: the sw_* EKI drivers import its
dense statistics / Gamma / forward helpers; it is not run as a script.
"""

from __future__ import annotations

import os

_KNOTS_MODE = os.environ.get("SDE_DENSE_KNOTS", "uniform8")
os.environ.setdefault("SDE_SPLINE_KNOTS", "8")

from pathlib import Path

import numpy as np

import sde_closure_config as closure_config
import sde_closure_eki as base

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS = PROJECT_DIR / "results" / "sde_closure"


# Dense profile stations: every 100 m over the physical domain.
DENSE_X_M = np.arange(0.0, 4000.0, 100.0)
N_DENSE = DENSE_X_M.size  # 40

# 16 knots concentrated where the error lives (y = 4000 - x metres):
# sparse on the offshore flat, ~300 m on the slope, ~150-200 m nearshore.
CONCENTRATED_KNOT_Y_M = np.asarray(
    [0.0, 500.0, 1000.0, 1300.0, 1600.0, 1900.0, 2200.0, 2500.0,
     2750.0, 3000.0, 3200.0, 3400.0, 3600.0, 3750.0, 3900.0, 4000.0]
)

if _KNOTS_MODE == "concentrated16":
    base.N_SPLINE_KNOTS = CONCENTRATED_KNOT_Y_M.size
    base.SPLINE_KNOT_Y_M = CONCENTRATED_KNOT_Y_M


# ----------------------------------------------------------------------
# Dense statistics (blocks: Hs(40) | bands(15) | skew(5) | kurt(5) |
# deviation-rms(40) | ACF(6)  ->  q = 111)
# ----------------------------------------------------------------------


def compute_statistics_dense(
    times_s: np.ndarray,
    dense_series_paths: list[np.ndarray],
    gauge_series_paths: list[np.ndarray],
    dense_baseline_series: np.ndarray,
    analysis: np.ndarray,
) -> np.ndarray:
    dense_pooled = np.concatenate(
        [series[analysis] for series in dense_series_paths], axis=0
    )
    hs_values = 4.0 * np.std(dense_pooled, axis=0, ddof=0)

    psd_accumulator = None
    for series in gauge_series_paths:
        frequency_hz, psd, _ = base._one_sided_psd(
            times_s, series, analysis
        )
        psd_accumulator = (
            psd if psd_accumulator is None else psd_accumulator + psd
        )
    psd_mean = psd_accumulator / float(len(gauge_series_paths))
    band_values = []
    for gauge_index in range(psd_mean.shape[0]):
        for low, high in base.BANDS_HZ:
            mask = (frequency_hz >= low) & (frequency_hz <= high)
            power = float(
                np.trapezoid(
                    psd_mean[gauge_index][mask], frequency_hz[mask]
                )
            )
            band_values.append(np.log10(max(power, 1.0e-12)))

    gauge_pooled = np.concatenate(
        [series[analysis] for series in gauge_series_paths], axis=0
    )
    mean = np.mean(gauge_pooled, axis=0)
    centred = gauge_pooled - mean
    std = np.maximum(np.std(gauge_pooled, axis=0, ddof=0), 1.0e-12)
    skew = np.mean(centred**3, axis=0) / std**3
    kurt = np.mean(centred**4, axis=0) / std**4 - 3.0

    deviation = np.concatenate(
        [
            (series - dense_baseline_series)[analysis]
            for series in dense_series_paths
        ],
        axis=0,
    )
    deviation_rms = np.sqrt(np.mean(deviation**2, axis=0))

    acf_values = []
    for gauge_index in base.ACF_GAUGE_INDICES:
        for lag in base.ACF_LAG_STEPS:
            per_path = [
                base.demeaned_acf(series[analysis][:, gauge_index], lag)
                for series in gauge_series_paths
            ]
            acf_values.append(float(np.mean(per_path)))

    return np.concatenate(
        [
            hs_values,
            np.asarray(band_values),
            skew,
            kurt,
            deviation_rms,
            np.asarray(acf_values),
        ]
    )


def truth_statistics_and_blocks_dense() -> tuple[np.ndarray, np.ndarray]:
    production = np.load(base.PRODUCTION_NPZ, allow_pickle=True)
    times_s = np.asarray(production["times_s"], dtype=float)
    eta_fine = np.asarray(production["eta_fine_m"], dtype=float)
    x_fine = np.asarray(production["x_physical_m_fine"], dtype=float)
    y_fine = 4000.0 - x_fine
    dense_cols = np.asarray(
        [
            int(np.argmin(np.abs(y_fine - (4000.0 - x))))
            for x in DENSE_X_M
        ],
        dtype=int,
    )
    gauge_cols = np.asarray(
        [
            int(np.argmin(np.abs(y_fine - (4000.0 - x))))
            for x in base.GAUGE_X_M
        ],
        dtype=int,
    )
    dense_series = eta_fine[:, dense_cols]
    gauge_series = eta_fine[:, gauge_cols]

    baseline = np.load(base.BASELINE_NPZ, allow_pickle=True)
    y_coarse = np.asarray(baseline["y_physical_m"], dtype=float)
    dense_coarse = np.asarray(
        [
            int(np.argmin(np.abs(y_coarse - (4000.0 - x))))
            for x in DENSE_X_M
        ],
        dtype=int,
    )
    baseline_dense = np.asarray(
        baseline["eta_coarse_m"], dtype=float
    )[:, dense_coarse]
    baseline_times = np.asarray(baseline["times_s"], dtype=float)
    common = min(times_s.size, baseline_times.size) - 1
    if not np.allclose(
        times_s[:common], baseline_times[:common], atol=1.0e-6
    ):
        raise RuntimeError("fine and baseline saved times differ")

    times_c = times_s[:common]
    dense_c = dense_series[:common]
    gauges_c = gauge_series[:common]
    baseline_c = baseline_dense[:common]
    analysis, _ = base._analysis_mask(times_c, base.ANALYSIS_START_S)
    observation = compute_statistics_dense(
        times_c, [dense_c], [gauges_c], baseline_c, analysis
    )
    indices = np.nonzero(analysis)[0]
    blocks = np.array_split(indices, base.N_BLOCKS)
    block_stats = []
    for block in blocks:
        mask = np.zeros_like(analysis)
        mask[block] = True
        block_stats.append(
            compute_statistics_dense(
                times_c, [dense_c], [gauges_c], baseline_c, mask
            )
        )
    return observation, np.asarray(block_stats)


def build_gamma_dense(
    observation: np.ndarray,
    block_stats: np.ndarray,
    forward_stats: np.ndarray,
) -> np.ndarray:
    truth_variance = np.var(block_stats, axis=0, ddof=1) / base.N_BLOCKS
    forward_variance = np.var(forward_stats, axis=0, ddof=1)
    floors = np.concatenate(
        [
            np.maximum(0.004, 0.05 * np.abs(observation[:N_DENSE])),
            np.full(15, 0.05),
            np.full(5, 0.03),
            np.full(5, 0.06),
            np.maximum(
                0.003,
                0.10
                * np.abs(observation[N_DENSE + 25 : 2 * N_DENSE + 25]),
            ),
            np.full(6, 0.05),
        ]
    )
    gamma_diagonal = truth_variance + forward_variance + floors**2
    # Block normalisation: the dense Hs and deviation blocks carry
    # strongly correlated entries; inflate their per-entry variance by
    # the densification ratio so each block's TOTAL precision mass under
    # the diagonal Gamma matches the original 8-/5-station design and
    # the dense blocks refine spatial shape without out-voting the
    # moment/ACF blocks.
    gamma_diagonal[:N_DENSE] *= N_DENSE / 8.0
    gamma_diagonal[N_DENSE + 25 : 2 * N_DENSE + 25] *= N_DENSE / 5.0
    return np.diag(gamma_diagonal)


# ----------------------------------------------------------------------
# Worker
# ----------------------------------------------------------------------


def _init_worker_dense(
    duration_s: float,
    boundary_seed: int,
    variance_preserving: bool = False,
) -> None:
    base._init_worker_with(duration_s, boundary_seed, variance_preserving)
    parameters = base._CTX["parameters"]
    y_physical_m = (
        base._CTX["y"][: base.COARSE_N4] * parameters.lambda_ref_m
    )
    dense_columns = np.asarray(
        [
            int(np.argmin(np.abs(y_physical_m - (4000.0 - x))))
            for x in DENSE_X_M
        ],
        dtype=int,
    )
    baseline = np.load(base.BASELINE_NPZ, allow_pickle=True)
    y_coarse = np.asarray(baseline["y_physical_m"], dtype=float)
    dense_coarse = np.asarray(
        [
            int(np.argmin(np.abs(y_coarse - (4000.0 - x))))
            for x in DENSE_X_M
        ],
        dtype=int,
    )
    base._CTX["dense_columns"] = dense_columns
    base._CTX["dense_baseline"] = np.asarray(
        baseline["eta_coarse_m"], dtype=float
    )[:, dense_coarse]


def forward_statistics_dense(
    task: tuple[np.ndarray, str, str, bool, bool, bool, tuple[int, ...]],
) -> np.ndarray | None:
    (
        theta,
        mode,
        envelope,
        colored,
        flat_sigma,
        no_damping,
        seed_root,
    ) = task
    if mode != "B":
        raise ValueError("the dense-obs driver is grid-noise only")
    from sde_closure_core import (
        GridWhiteNoise,
        GridWhiteNoiseParameters,
        StochasticImplicitMidpointDABCSolver,
    )

    parameters = base._CTX["parameters"]
    decoded = base.decode_theta(
        theta, mode, envelope, colored, flat_sigma, no_damping
    )
    weight = base.build_envelope_weight(decoded, envelope)
    damping = np.exp(
        -decoded["lambda"] * base.COARSE_DT * np.asarray(base._CTX["taper"])
    )

    dense_series_paths: list[np.ndarray] = []
    gauge_series_paths: list[np.ndarray] = []
    times_s: np.ndarray | None = None
    for path in range(base.FORWARD_PATHS):
        rng = np.random.default_rng(list(seed_root) + [path])
        noise = GridWhiteNoise(
            GridWhiteNoiseParameters(
                phi_amplitude=decoded["phi"],
                correlation_length_cells=3.0,
                correlation_time_nd=decoded.get("tau", 0.0),
                variance_preserving=bool(
                    base._CTX.get("variance_preserving", False)
                ),
            ),
            base._CTX["y"],
            parameters.lambda_ref_m,
            base._CTX["surface_to_green"],
            base.COARSE_DT,
            rng,
            spatial_weight=weight,
        )
        with closure_config.template():
            solver = StochasticImplicitMidpointDABCSolver(
                base._CTX["y"],
                base._CTX["depth_ratio"],
                parameters.epsilon,
                parameters.mu,
                base.COARSE_DT,
                base._CTX["n_steps"],
            )
        try:
            times, surface, _, _ = solver.run_stochastic(
                np.zeros_like(base._CTX["y"]),
                base.OUTPUT_STRIDE,
                base._CTX["traces"],
                noise_increment=noise,
                damping_factor=damping,
            )
        except FloatingPointError:
            return None
        times_s = np.asarray(times) * parameters.time_ref_s
        eta = (
            np.asarray(surface[:, : base.COARSE_N4], dtype=float)
            * parameters.a_ref_m
        )
        if not np.all(np.isfinite(eta)):
            return None
        dense_series_paths.append(eta[:, base._CTX["dense_columns"]])
        gauge_series_paths.append(eta[:, base._CTX["gauge_columns"]])
    baseline_times = base._CTX["baseline_times_s"]
    count = times_s.size
    if baseline_times.size < count:
        raise RuntimeError("baseline record shorter than the forward run")
    if not np.allclose(
        times_s[: count - 1], baseline_times[: count - 1], atol=1.0e-6
    ):
        raise RuntimeError("forward saved times differ from the baseline")
    analysis, _ = base._analysis_mask(times_s, base.ANALYSIS_START_S)
    return compute_statistics_dense(
        times_s,
        dense_series_paths,
        gauge_series_paths,
        np.asarray(base._CTX["dense_baseline"])[:count],
        analysis,
    )
