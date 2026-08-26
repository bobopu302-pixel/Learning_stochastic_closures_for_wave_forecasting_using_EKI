"""EKI v2: upgraded calibration of the coarse-grid stochastic vKdV closure.

Origin: 3. KDV_nonlinear_case/sde_closure_eki.py
Changes vs origin (numerics untouched):
* the legacy standalone main() and its __main__ hook DELETED: they
  contained figure code and are never called by the 2026-08-23 spec
  chain (the sw_* drivers import this module as a library); the
  imports and colour constants used only by main() removed with it;
* every symbol the drivers import is kept: theta_layout, decode_theta,
  clip_ensemble, initial_ensemble, build_envelope_weight, demeaned_acf,
  compute_statistics, _init_worker_with, forward_statistics,
  truth_statistics_and_blocks, build_gamma, module constants, _CTX.

Upgrades over the retired v1 EKI driver (this v2 content now carries the
`sde_closure_eki.py` name), addressing improvement points 3 (observation
design), 4 (forward noise / collapse control), and 2 (scheme-A envelope
decoupling):

* common random numbers (CRN): all members within one iteration share the
  same noise seeds, so G-differences reflect theta-differences, not seed
  luck -- the direct cure for the v1 best-member overfit;
* 4 forward paths per evaluation (was 2);
* two new observation blocks: deviation-from-frozen-baseline std at the 5
  interior gauges (directly constrains ensemble spread) and demeaned gauge
  autocorrelation at lags {3.46 s, 12.12 s} x gauges {0, 1000, 3000} m
  (constrains temporal coherence; expected to expose the white-in-time
  limitation) -> q_obs = 44;
* parameter-ensemble spread recording plus an early-stopping rule
  (discrepancy level or stagnating objective);
* selectable noise envelope: `terrain` (honest (d_min/d)^q), `oracle`
  (frozen N1 error-production weight; scheme-A decoupling control), and
  `bump` (learnable plateau+Gaussian local shape, no fine information).

This module is a library in this release: it is imported by the sw_* EKI
drivers (statistics, forward map, worker init, theta encode/decode, Gamma)
and is not run as a script.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from coastal_entropy_midpoint_production import _analysis_mask, _one_sided_psd
import sde_closure_config as closure_config

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS = PROJECT_DIR / "results" / "sde_closure"
PRODUCTION_NPZ = (
    PROJECT_DIR
    / "results"
    / "physical_time_final"
    / "production_1800s"
    / "raw_data"
    / "production_replot_data.npz"
)
BASELINE_NPZ = closure_config.BASELINE_NPZ

EKI_SEED = 42
COARSE_N4 = closure_config.COARSE_N4
COARSE_DT = closure_config.COARSE_DT
OUTPUT_STRIDE = closure_config.OUTPUT_STRIDE
ANALYSIS_START_S = 600.0
N_BLOCKS = 6
FORWARD_PATHS = 4
FORWARD_REPEATS = 12

HS_X_M = np.arange(0.0, 4000.0, 500.0)
GAUGE_X_M = np.asarray([0.0, 500.0, 1000.0, 2000.0, 3000.0])
BANDS_HZ = ((0.03, 0.06), (0.06, 0.085), (0.085, 0.115))
ACF_LAG_STEPS = (2, 7)  # x saved cadence 1.731 s = 3.46 s, 12.12 s
ACF_GAUGE_INDICES = (0, 2, 4)  # x = 0, 1000, 3000 m

N_MODES = 12
FREQUENCIES_HZ = np.linspace(0.035, 0.100, N_MODES)

BOUNDS_COMMON = {
    "sqrt_sigma": (1.0e-4, 0.5),
    "phi": (1.0e-4, 0.5),
    "lambda": (0.0, 0.3),
    "q": (0.0, 6.0),
    "bump_center_m": (200.0, 3800.0),
    "bump_width_m": (100.0, 1500.0),
    "bump_plateau": (0.0, 1.0),
    "tau": (0.0, 4.0),
    "knot": (1.0e-4, 0.5),
}

import os as _os

N_SPLINE_KNOTS = int(_os.environ.get("SDE_SPLINE_KNOTS", "12"))
SPLINE_KNOT_Y_M = np.linspace(0.0, 4000.0, N_SPLINE_KNOTS)


_CTX: dict[str, object] = {}


# ----------------------------------------------------------------------
# Parameter layout
# ----------------------------------------------------------------------


def theta_layout(
    mode: str,
    envelope: str,
    colored: bool = False,
    flat_sigma: bool = False,
    no_damping: bool = False,
) -> list[str]:
    if mode == "A":
        amplitude = (
            ["sqrt_sigma_flat"]
            if flat_sigma
            else [f"sqrt_sigma_{j}" for j in range(N_MODES)]
        )
    elif envelope == "spline":
        # User-proposed scheme: learn the amplitude FIELD sigma(x) directly
        # (12 spline knots over the physical region); the global phi is
        # absorbed into the knot values.
        amplitude = [f"knot_{i}" for i in range(N_SPLINE_KNOTS)]
    else:
        amplitude = ["phi"]
    names = amplitude + ([] if no_damping else ["lambda"])
    if envelope == "terrain":
        names.append("q")
    elif envelope == "bump":
        names.extend(["bump_center_m", "bump_width_m", "bump_plateau"])
    if colored:
        names.append("tau")
    return names


def decode_theta(
    theta: np.ndarray,
    mode: str,
    envelope: str,
    colored: bool = False,
    flat_sigma: bool = False,
    no_damping: bool = False,
) -> dict[str, object]:
    theta = np.asarray(theta, dtype=float)
    cursor = 0
    decoded: dict[str, object] = {}
    if mode == "A":
        if flat_sigma:
            value = float(
                np.clip(theta[0], *BOUNDS_COMMON["sqrt_sigma"])
            )
            decoded["sqrt_sigma"] = np.full(N_MODES, value)
            cursor = 1
        else:
            decoded["sqrt_sigma"] = np.clip(
                theta[:N_MODES], *BOUNDS_COMMON["sqrt_sigma"]
            )
            cursor = N_MODES
    elif envelope == "spline":
        decoded["spline_knots"] = np.clip(
            theta[:N_SPLINE_KNOTS], *BOUNDS_COMMON["knot"]
        )
        decoded["phi"] = 1.0
        cursor = N_SPLINE_KNOTS
    else:
        decoded["phi"] = float(np.clip(theta[0], *BOUNDS_COMMON["phi"]))
        cursor = 1
    if no_damping:
        decoded["lambda"] = 0.0
    else:
        decoded["lambda"] = float(
            np.clip(theta[cursor], *BOUNDS_COMMON["lambda"])
        )
        cursor += 1
    if envelope == "terrain":
        decoded["q"] = float(np.clip(theta[cursor], *BOUNDS_COMMON["q"]))
        cursor += 1
    elif envelope == "bump":
        decoded["bump_center_m"] = float(
            np.clip(theta[cursor], *BOUNDS_COMMON["bump_center_m"])
        )
        decoded["bump_width_m"] = float(
            np.clip(theta[cursor + 1], *BOUNDS_COMMON["bump_width_m"])
        )
        decoded["bump_plateau"] = float(
            np.clip(theta[cursor + 2], *BOUNDS_COMMON["bump_plateau"])
        )
        cursor += 3
    if colored:
        decoded["tau"] = float(
            np.clip(theta[cursor], *BOUNDS_COMMON["tau"])
        )
    return decoded


def clip_ensemble(
    thetas: np.ndarray,
    mode: str,
    envelope: str,
    colored: bool = False,
    flat_sigma: bool = False,
    no_damping: bool = False,
) -> np.ndarray:
    names = theta_layout(mode, envelope, colored, flat_sigma, no_damping)
    clipped = thetas.copy()
    for column, name in enumerate(names):
        if name.startswith("sqrt_sigma"):
            key = "sqrt_sigma"
        elif name.startswith("knot"):
            key = "knot"
        else:
            key = name
        low, high = BOUNDS_COMMON[key]
        clipped[:, column] = np.clip(clipped[:, column], low, high)
    return clipped


def initial_ensemble(
    mode: str,
    envelope: str,
    members: int,
    rng: np.random.Generator,
    colored: bool = False,
    flat_sigma: bool = False,
    no_damping: bool = False,
) -> np.ndarray:
    columns = []
    if mode == "A":
        width = 1 if flat_sigma else N_MODES
        columns.append(rng.uniform(0.005, 0.10, size=(members, width)))
    elif envelope == "spline":
        # Smooth correlated prior (SE kernel over knot index, length 1.5
        # knots): keeps the EKI update subspace biased toward smooth
        # sigma(x) fields, tightening the pointwise underdetermination.
        index = np.arange(N_SPLINE_KNOTS, dtype=float)
        kernel = np.exp(
            -0.5 * ((index[:, None] - index[None, :]) / 1.5) ** 2
        )
        draws = rng.multivariate_normal(
            np.full(N_SPLINE_KNOTS, 0.02),
            (0.012**2) * kernel,
            size=members,
        )
        columns.append(np.clip(draws, 1.0e-4, None))
    else:
        columns.append(rng.uniform(0.005, 0.10, size=(members, 1)))
    if not no_damping:
        columns.append(rng.uniform(0.0, 0.15, size=(members, 1)))
    if envelope == "terrain":
        columns.append(rng.uniform(0.0, 5.0, size=(members, 1)))
    elif envelope == "bump":
        columns.append(rng.uniform(800.0, 3200.0, size=(members, 1)))
        columns.append(rng.uniform(200.0, 1000.0, size=(members, 1)))
        columns.append(rng.uniform(0.0, 0.5, size=(members, 1)))
    if colored:
        columns.append(rng.uniform(0.0, 2.0, size=(members, 1)))
    return np.hstack(columns)


def build_envelope_weight(
    decoded: dict[str, object],
    envelope: str,
) -> np.ndarray:
    """Spatial weight on the full computational grid (worker side)."""

    from sde_closure_core import terrain_weight

    if envelope == "spline":
        y_m = (
            np.asarray(_CTX["y"], dtype=float)
            * _CTX["parameters"].lambda_ref_m
        )
        field = np.interp(
            y_m,
            SPLINE_KNOT_Y_M,
            np.asarray(decoded["spline_knots"], dtype=float),
            left=0.0,
            right=0.0,
        )
        return field
    if envelope == "uniform":
        return np.ones_like(
            np.asarray(_CTX["depth_ratio"], dtype=float)
        )
    if envelope == "terrain":
        return terrain_weight(_CTX["depth_ratio"], decoded["q"])
    if envelope == "oracle":
        return _CTX["oracle_weight"]
    y_m = np.asarray(_CTX["y"], dtype=float) * _CTX[
        "parameters"
    ].lambda_ref_m
    plateau = decoded["bump_plateau"]
    shape = plateau + (1.0 - plateau) * np.exp(
        -0.5
        * ((y_m - decoded["bump_center_m"]) / decoded["bump_width_m"]) ** 2
    )
    return shape / float(np.max(shape))


# ----------------------------------------------------------------------
# Statistics (shared master/worker)
# ----------------------------------------------------------------------


def demeaned_acf(series: np.ndarray, lag: int) -> float:
    values = np.asarray(series, dtype=float)
    values = values - np.mean(values)
    denominator = float(np.dot(values, values))
    if denominator <= 0.0:
        return 0.0
    return float(
        np.dot(values[:-lag], values[lag:]) / denominator
    )


def compute_statistics(
    times_s: np.ndarray,
    hs_series_paths: list[np.ndarray],
    gauge_series_paths: list[np.ndarray],
    baseline_gauge_series: np.ndarray,
    analysis: np.ndarray,
) -> np.ndarray:
    """44-dim statistics vector.

    Blocks: Hs(8) | log10 band power (5x3) | skew(5) | kurt(5) |
    deviation-from-baseline rms (5) | demeaned ACF (3 gauges x 2 lags).
    ``baseline_gauge_series`` must be on the same saved time grid.
    """

    hs_pooled = np.concatenate(
        [series[analysis] for series in hs_series_paths], axis=0
    )
    hs_values = 4.0 * np.std(hs_pooled, axis=0, ddof=0)

    psd_accumulator = None
    for series in gauge_series_paths:
        frequency_hz, psd, _ = _one_sided_psd(times_s, series, analysis)
        psd_accumulator = (
            psd if psd_accumulator is None else psd_accumulator + psd
        )
    psd_mean = psd_accumulator / float(len(gauge_series_paths))
    band_values = []
    for gauge_index in range(psd_mean.shape[0]):
        for low, high in BANDS_HZ:
            mask = (frequency_hz >= low) & (frequency_hz <= high)
            power = float(
                np.trapezoid(psd_mean[gauge_index][mask], frequency_hz[mask])
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
            (series - baseline_gauge_series)[analysis]
            for series in gauge_series_paths
        ],
        axis=0,
    )
    deviation_rms = np.sqrt(np.mean(deviation**2, axis=0))

    acf_values = []
    for gauge_index in ACF_GAUGE_INDICES:
        for lag in ACF_LAG_STEPS:
            per_path = [
                demeaned_acf(series[analysis][:, gauge_index], lag)
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


# ----------------------------------------------------------------------
# Worker
# ----------------------------------------------------------------------


def _init_worker_with(
    duration_s: float,
    boundary_seed: int,
    variance_preserving: bool = False,
) -> None:
    _CTX["variance_preserving"] = bool(variance_preserving)
    import numpy as _np
    import sde_closure_config as _closure_config

    from pde_core import CoastalParameters
    from high_order_variable_depth_dabc import coastal_depth_ratio_y
    from high_order_variable_depth_dabc_study import (
        StudyConfig as LinearStudyConfig,
        _aligned_steps,
        _grid_for_length,
        _tma_inputs,
    )
    from sde_closure_core import (
        error_production_weight,
        smooth_physical_taper,
    )

    parameters = CoastalParameters()
    y, dy, _ = _grid_for_length(10000.0, parameters, short_n=COARSE_N4)
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
    n_steps = _aligned_steps(duration_s, parameters, COARSE_DT)
    with _closure_config.template():
        _, _, traces, _ = _tma_inputs(
            parameters,
            LinearStudyConfig(random_seed=boundary_seed),
            dy,
            COARSE_DT,
        )
    y_physical_m = y[:COARSE_N4] * parameters.lambda_ref_m
    hs_columns = _np.asarray(
        [
            int(_np.argmin(_np.abs(y_physical_m - (4000.0 - x))))
            for x in HS_X_M
        ],
        dtype=int,
    )
    gauge_columns = _np.asarray(
        [
            int(_np.argmin(_np.abs(y_physical_m - (4000.0 - x))))
            for x in GAUGE_X_M
        ],
        dtype=int,
    )
    baseline = _np.load(BASELINE_NPZ, allow_pickle=True)
    baseline_gauges = _np.asarray(
        baseline["eta_coarse_m"], dtype=float
    )[:, gauge_columns]
    oracle_weight = error_production_weight(
        y * parameters.lambda_ref_m,
        depth_ratio * parameters.h_ref_m,
        _np.asarray(baseline["y_ref_fine_m"], dtype=float),
        _np.asarray(baseline["error_std_profile_fine_m"], dtype=float),
    )
    _CTX.update(
        parameters=parameters,
        y=y,
        depth_ratio=depth_ratio,
        n_steps=n_steps,
        traces=traces,
        taper=smooth_physical_taper(y * parameters.lambda_ref_m),
        surface_to_green=depth_ratio**0.25,
        hs_columns=hs_columns,
        gauge_columns=gauge_columns,
        baseline_gauges=baseline_gauges,
        baseline_times_s=_np.asarray(baseline["times_s"], dtype=float),
        oracle_weight=oracle_weight,
    )


def forward_statistics(
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
    from sde_closure_core import (
        GridWhiteNoise,
        GridWhiteNoiseParameters,
        ModalNoise,
        ModalNoiseParameters,
        StochasticImplicitMidpointDABCSolver,
    )

    parameters = _CTX["parameters"]
    decoded = decode_theta(
        theta, mode, envelope, colored, flat_sigma, no_damping
    )
    weight = build_envelope_weight(decoded, envelope)
    damping = np.exp(
        -decoded["lambda"] * COARSE_DT * np.asarray(_CTX["taper"])
    )

    hs_series_paths: list[np.ndarray] = []
    gauge_series_paths: list[np.ndarray] = []
    times_s: np.ndarray | None = None
    for path in range(FORWARD_PATHS):
        rng = np.random.default_rng(list(seed_root) + [path])
        if mode == "A":
            noise = ModalNoise(
                ModalNoiseParameters(
                    frequencies_hz=tuple(FREQUENCIES_HZ),
                    sqrt_sigma=tuple(decoded["sqrt_sigma"]),
                    correlation_time_nd=decoded.get("tau", 0.0),
                    variance_preserving=bool(
                        _CTX.get("variance_preserving", False)
                    ),
                ),
                _CTX["y"],
                parameters.lambda_ref_m,
                parameters.time_ref_s,
                parameters.mu,
                _CTX["surface_to_green"],
                COARSE_DT,
                rng,
                spatial_weight=weight,
            )
        else:
            noise = GridWhiteNoise(
                GridWhiteNoiseParameters(
                    phi_amplitude=decoded["phi"],
                    correlation_length_cells=3.0,
                    correlation_time_nd=decoded.get("tau", 0.0),
                    variance_preserving=bool(
                        _CTX.get("variance_preserving", False)
                    ),
                ),
                _CTX["y"],
                parameters.lambda_ref_m,
                _CTX["surface_to_green"],
                COARSE_DT,
                rng,
                spatial_weight=weight,
            )
        with closure_config.template():
            solver = StochasticImplicitMidpointDABCSolver(
                _CTX["y"],
                _CTX["depth_ratio"],
                parameters.epsilon,
                parameters.mu,
                COARSE_DT,
                _CTX["n_steps"],
            )
        try:
            times, surface, _, _ = solver.run_stochastic(
                np.zeros_like(_CTX["y"]),
                OUTPUT_STRIDE,
                _CTX["traces"],
                noise_increment=noise,
                damping_factor=damping,
            )
        except FloatingPointError:
            return None
        times_s = np.asarray(times) * parameters.time_ref_s
        eta = (
            np.asarray(surface[:, :COARSE_N4], dtype=float)
            * parameters.a_ref_m
        )
        if not np.all(np.isfinite(eta)):
            return None
        hs_series_paths.append(eta[:, _CTX["hs_columns"]])
        gauge_series_paths.append(eta[:, _CTX["gauge_columns"]])
    baseline_times = _CTX["baseline_times_s"]
    count = times_s.size
    if baseline_times.size < count:
        raise RuntimeError("baseline record shorter than the forward run")
    # All saved grids share the 0.14 nondimensional cadence; only the final
    # forced save may differ, so align on the common prefix.
    if not np.allclose(
        times_s[: count - 1], baseline_times[: count - 1], atol=1.0e-6
    ):
        raise RuntimeError("forward saved times differ from the baseline")
    analysis, _ = _analysis_mask(times_s, ANALYSIS_START_S)
    return compute_statistics(
        times_s,
        hs_series_paths,
        gauge_series_paths,
        np.asarray(_CTX["baseline_gauges"])[:count],
        analysis,
    )


# ----------------------------------------------------------------------
# Master
# ----------------------------------------------------------------------


def truth_statistics_and_blocks() -> tuple[np.ndarray, np.ndarray]:
    production = np.load(PRODUCTION_NPZ, allow_pickle=True)
    times_s = np.asarray(production["times_s"], dtype=float)
    eta_fine = np.asarray(production["eta_fine_m"], dtype=float)
    x_fine = np.asarray(production["x_physical_m_fine"], dtype=float)
    y_fine = 4000.0 - x_fine
    hs_columns = np.asarray(
        [int(np.argmin(np.abs(y_fine - (4000.0 - x)))) for x in HS_X_M],
        dtype=int,
    )
    gauge_columns = np.asarray(
        [int(np.argmin(np.abs(y_fine - (4000.0 - x)))) for x in GAUGE_X_M],
        dtype=int,
    )
    hs_series = eta_fine[:, hs_columns]
    gauge_series = eta_fine[:, gauge_columns]

    baseline = np.load(BASELINE_NPZ, allow_pickle=True)
    y_coarse = np.asarray(baseline["y_physical_m"], dtype=float)
    coarse_cols = np.asarray(
        [int(np.argmin(np.abs(y_coarse - (4000.0 - x)))) for x in GAUGE_X_M],
        dtype=int,
    )
    baseline_gauges = np.asarray(baseline["eta_coarse_m"], dtype=float)[
        :, coarse_cols
    ]
    baseline_times = np.asarray(baseline["times_s"], dtype=float)
    common = min(times_s.size, baseline_times.size) - 1
    if not np.allclose(
        times_s[:common], baseline_times[:common], atol=1.0e-6
    ):
        raise RuntimeError("fine and baseline saved times differ")

    times_c = times_s[:common]
    hs_c = hs_series[:common]
    gauges_c = gauge_series[:common]
    baseline_c = baseline_gauges[:common]
    analysis, _ = _analysis_mask(times_c, ANALYSIS_START_S)
    observation = compute_statistics(
        times_c, [hs_c], [gauges_c], baseline_c, analysis
    )
    indices = np.nonzero(analysis)[0]
    blocks = np.array_split(indices, N_BLOCKS)
    block_stats = []
    for block in blocks:
        mask = np.zeros_like(analysis)
        mask[block] = True
        block_stats.append(
            compute_statistics(
                times_c, [hs_c], [gauges_c], baseline_c, mask
            )
        )
    return observation, np.asarray(block_stats)


def build_gamma(
    observation: np.ndarray,
    block_stats: np.ndarray,
    forward_stats: np.ndarray,
    kurtosis_weight: float = 1.0,
) -> np.ndarray:
    truth_variance = np.var(block_stats, axis=0, ddof=1) / N_BLOCKS
    forward_variance = np.var(forward_stats, axis=0, ddof=1)
    floors = np.concatenate(
        [
            np.maximum(0.004, 0.05 * np.abs(observation[:8])),
            np.full(15, 0.05),
            np.full(5, 0.03),
            np.full(5, 0.06),
            np.maximum(0.003, 0.10 * np.abs(observation[33:38])),
            np.full(6, 0.05),
        ]
    )
    gamma_diagonal = truth_variance + forward_variance + floors**2
    if kurtosis_weight != 1.0:
        # Kurtosis block occupies entries 28..32 (after Hs 8, bands 15,
        # skew 5); increasing the weight shrinks Gamma there.
        gamma_diagonal[28:33] /= float(kurtosis_weight) ** 2
    return np.diag(gamma_diagonal)
