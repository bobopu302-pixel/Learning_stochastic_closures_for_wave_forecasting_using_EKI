"""Step 1 (S1) twin inversion driver: recover (phi, q) by EKI -- spec chain.

Origin: 3. KDV_nonlinear_case/sw_eki_s1.py
Changes vs origin:
- ENGINE REWIRE: the hand-rolled EKI loop in main() is replaced by
  algorithms.eki.run_eki.  The ensemble evaluation (process pool + CRN
  spec_seed_root + N_G forward averaging) moved into an ensemble_evaluator
  callable; failed-forward handling uses sentinel_row_fn
  (= observation + 10*sqrt(diag(Gamma))) with the spec
  all-realisations-failed rule in failed_mask_fn; the spec stopping rule is
  stop_rel_tol=0.01 / stop_patience=3; the spec latent clip is a post_update
  hook.  jitter=0.0 reproduces the origin's un-regularised Kalman-gain solve
  (for Phi it is redundant: the engine always computes Phi with the exact
  Gamma, by Cholesky whitening -- last-ulp different from the origin loop's
  explicit-inverse arithmetic, which is why the audit files below recompute
  Phi with the origin formula).  The
  per-iteration audit files iter_NNN.npz (pre-update thetas, G matrix,
  per-member Phi, ensemble Phi) are written from inside the evaluator, so
  their layout is identical to the origin's.  One behavioural addition of
  the engine: when the iteration cap is reached (stop rule never fired) the
  final post-update ensemble is evaluated once more, producing one extra
  iter_NNN.npz; the summary reports it separately and iterations_run keeps
  the origin meaning (number of in-loop evaluations).
- SPEC-ONLY: SW_ALGO / SW_STOP_RULE / SW_GAMMA_MODE env branches deleted;
  the 2026-08-23 spec behaviour is hard-set (SPEC=True, unified Gamma via
  sw_gamma_unified with SW_GAMMA_TERMS=var_ref_only, log-phi coordinates,
  N_G forward averaging, spec stop, final-ensemble-mean reporting).
  Deleted with them: joint_stop / spec_stop (the engine implements the
  stop rule), the legacy Gamma builders build_gamma_incr / the
  build_gamma_dense and base.build_gamma call sites, the legacy per-member
  seed path (v3_world.member_seed_root), the non-spec clip via
  base.clip_ensemble, and the linear-phi (non-log) prior draw.
- deleted the S1b spline variant (--config accepts only S1a; the spline
  recovery metrics and the spline branch of run_validation are gone) and
  the legacy --classic (q=44) and --incr (q=151) observation options: the
  spec S1 observes the dense q=111 vector.  compute_statistics_incr and
  increment_block are KEPT because the S2 driver builds its q=151 vector
  from them; forward_statistics_incr (an S1-only legacy forward) is gone.
- deleted the frozen coarse-grid forward branch (forward_statistics_dense):
  the spec chain is fine-grid only and main() now requires SW_FINE=1
  (forward_statistics_dense_fine honours CORR_CELLS on any grid, but the
  spec prior draw is defined in log space for the fine grid).
- deleted the matplotlib convergence/recovery figure at the end of main()
  (this release ships no plotting); the data behind it is all in
  iter_NNN.npz + summary.json.
- run_validation no longer returns the private "_spread"/"_hs_post"/
  "_weight" arrays (they only fed the deleted figure).
- removed the never-read history_spread list.
- comments/docstrings translated and polished.

The forward model, worker context, statistics and CRN conventions are the
FROZEN closure machinery (imported, not copied); only the observation
source (twin bundle) and the baseline (twin deterministic run, via
SDE_BASELINE_NPZ) differ from the production chain.  After calibration the
driver runs an 8-path fresh-seed validation ([20260810, m]) and writes the
recovery report.

Environment is read AT IMPORT TIME (see v3_world): always launch through
v3_spec_chain.sh or replicate its exports.

Usage (spec chain)::

    SW_FINE=1 SW_VERSION_DIR=v3spec SW_DURATION_S=6600 \
    SW_SYNTH_PERIOD_S=6600 SW_FORWARD_PATHS=1 SW_S1_VARIANT=S1a \
    SW_S1_SUFFIX=_fine SW_GAMMA_TYPE=diag SW_N_G=10 \
    python sw_eki_s1.py --config S1a --members 100 --iterations 20 \
        --processes 100 --overwrite
"""

from __future__ import annotations

import os
from pathlib import Path as _Path

_HERE = _Path(__file__).resolve().parent
SW_FINE = os.environ.get("SW_FINE", "") == "1"
if SW_FINE:
    os.environ["SDE_COARSE_N4"] = "3073"
    os.environ["SDE_COARSE_DT"] = "0.002"
    os.environ["SDE_OUTPUT_STRIDE"] = "70"
    os.environ["SDE_CLOSURE_V3"] = "0"  # standard C6/C4 template on fine
# Spec configuration is hard-set (2026-08-23): Gamma = diag/full of var_ref
# ONLY -- sw_gamma_unified reads this env at call time.
os.environ["SW_GAMMA_TERMS"] = "var_ref_only"
_VARIANT = os.environ.get("SW_S1_VARIANT", "")
_SUFFIX = os.environ.get("SW_S1_SUFFIX", "")
# The baseline env must point at the matching twin baseline BEFORE the
# frozen modules load; the chain exports SW_S1_VARIANT=S1a (and
# SW_S1_SUFFIX=_fine for the fine-grid twin world).
import v3_world  # noqa: E402  (v3 twin world patches; no-op for v2)
_VERSION_DIR = v3_world.version_dir("v2_fine" if SW_FINE else "v1_coarse")
if _VARIANT:
    os.environ["SDE_BASELINE_NPZ"] = str(
        _HERE / "results" / "stepwise" / _VERSION_DIR
        / f"truth_{_VARIANT}{_SUFFIX}" / "baseline_data.npz"
    )
os.environ.setdefault("SDE_SPLINE_KNOTS", "8")
v3_world.ensure_patched()

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Pool

import numpy as np

# code_rp root on sys.path so the shared algorithm layer imports.
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from algorithms.eki import run_eki  # noqa: E402

import sde_closure_eki as base  # noqa: E402

FINE_MODEL_KW = (
    {"coarse_n4": 3073, "coarse_dt": 0.002, "output_stride": 70}
    if SW_FINE else {}
)
CORR_CELLS = 24.0 if SW_FINE else 3.0  # 31.25 m physical
if SW_FINE:
    assert base.COARSE_N4 == 3073, "fine env switch failed to propagate"
# Unified convention (UNIFY_y_G_Gamma_spec 1.2, author decision
# 2026-08-16): G(theta) uses the SAME number of paths as the truth y.
# The v3 spec world sets SW_FORWARD_PATHS=1 (one path per realisation).
FORWARD_PATHS = int(os.environ.get("SW_FORWARD_PATHS", "16"))
# N_F forward repeats for the Gamma-stage diagnostics (SW_GAMMA_NF;
# default = frozen base.FORWARD_REPEATS = 12).  Under the spec these
# repeats only feed the measured var_fwd/var_ref ratio (the N_G check),
# not Gamma itself.
N_FWD_REPEATS = int(os.environ.get("SW_GAMMA_NF", str(base.FORWARD_REPEATS)))
from sde_closure_eki_dense import (  # noqa: E402
    DENSE_X_M,
    _init_worker_dense,
    compute_statistics_dense,
)
from sde_closure_context import CoarseModelContext, ModelAConfig  # noqa: E402
from sde_closure_core import GridWhiteNoise, GridWhiteNoiseParameters  # noqa: E402
from sde_closure_core import terrain_weight  # noqa: E402

OUT_ROOT = _HERE / "results" / "stepwise" / _VERSION_DIR
VALIDATION_SEED = 20260810
N_DENSE = DENSE_X_M.size

# ----------------------------------------------------------------------
# EKI_algorithm_spec_2026-08-23 (hard-set; the only shipped configuration)
#   Gamma    = diag(var_ref) (or shrunk full Cov_ref, SW_GAMMA_TYPE=full)
#   G(theta) = mean of N_G forward realisations (SW_N_G), common random
#              numbers: seed S(n, a) shared by every member in iteration n
#   Phi_n    = 0.5 || Gamma^-1/2 (y - mean_j G_hat[j]) ||^2
#   phi      = positive parameter -> log coordinates inside EKI
#   stop     = relative change of Phi < 1 % for 3 consecutive iterations
#   report   = final-iteration ensemble mean +/- ensemble sd
# ----------------------------------------------------------------------
SPEC = True   # kept as a module constant: v3_* scripts test s1.SPEC
N_G = int(os.environ.get("SW_N_G", "1"))


def spec_seed_root(eki_seed, iteration, realisation):
    """CRN: same seeds for every member, new ones at each iteration."""
    return (eki_seed, 3000 + iteration, realisation)


def model_theta(theta):
    """EKI coordinates -> model coordinates (exponentiate log-phi)."""
    t = np.array(theta, dtype=float, copy=True)
    t[0] = np.exp(t[0])
    return t


def forward_statistics_dense_fine(task):
    """Dense-station forward map with the noise correlation length held at
    its PHYSICAL value (24 fine cells = 31.25 m, the plan's stated intent);
    the frozen forward_statistics_dense hardcodes 3.0 cells, which on the
    fine grid is a different SPDE.  Returns the q=111 statistic vector, or
    None when the solver failed / produced non-finite fields."""

    import sde_closure_config as closure_config
    from sde_closure_core import (
        GridWhiteNoise,
        GridWhiteNoiseParameters,
        StochasticImplicitMidpointDABCSolver,
    )
    from sde_closure_eki_dense import compute_statistics_dense

    (theta, mode, envelope, colored, flat_sigma, no_damping,
     seed_root) = task
    theta = model_theta(theta)
    parameters = base._CTX["parameters"]
    decoded = base.decode_theta(
        theta, mode, envelope, colored, flat_sigma, no_damping
    )
    weight = base.build_envelope_weight(decoded, envelope)
    damping = np.exp(
        -decoded["lambda"] * base.COARSE_DT
        * np.asarray(base._CTX["taper"])
    )
    dense_series_paths, gauge_series_paths = [], []
    times_s = None
    for path in range(FORWARD_PATHS):
        # Path seed = [*seed_root, path]: the CRN root is shared by every
        # member within one iteration; only theta differs between members.
        rng = np.random.default_rng(list(seed_root) + [path])
        noise = GridWhiteNoise(
            GridWhiteNoiseParameters(
                phi_amplitude=decoded["phi"],
                correlation_length_cells=CORR_CELLS,
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
    count = times_s.size
    analysis, _ = base._analysis_mask(times_s, base.ANALYSIS_START_S)
    return compute_statistics_dense(
        times_s,
        dense_series_paths,
        gauge_series_paths,
        np.asarray(base._CTX["dense_baseline"])[:count],
        analysis,
    )


# ----------------------------------------------------------------------
# Path-parallel evaluation (SW_H_PATH_PARALLEL=1, shared switch with the
# S2 driver): each of the FORWARD_PATHS closure-noise paths of a member
# is its own pool task, seeded exactly as in forward_statistics_dense_fine
# ([*seed_root, path]); the pooled statistics are then computed from the
# same per-path series with the same function -> bit-identical result.
# ----------------------------------------------------------------------
PATH_PARALLEL = os.environ.get("SW_H_PATH_PARALLEL", "0") == "1"


def forward_path_dense_fine(task):
    """One (theta, path) task -> (times_s, dense_series, gauge_series)."""
    import sde_closure_config as closure_config
    from sde_closure_core import (
        GridWhiteNoise,
        GridWhiteNoiseParameters,
        StochasticImplicitMidpointDABCSolver,
    )

    (theta, mode, envelope, colored, flat_sigma, no_damping,
     seed_root, path) = task
    theta = model_theta(theta)
    parameters = base._CTX["parameters"]
    decoded = base.decode_theta(
        theta, mode, envelope, colored, flat_sigma, no_damping
    )
    weight = base.build_envelope_weight(decoded, envelope)
    damping = np.exp(
        -decoded["lambda"] * base.COARSE_DT
        * np.asarray(base._CTX["taper"])
    )
    rng = np.random.default_rng(list(seed_root) + [path])
    noise = GridWhiteNoise(
        GridWhiteNoiseParameters(
            phi_amplitude=decoded["phi"],
            correlation_length_cells=CORR_CELLS,
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
    return (times_s, eta[:, base._CTX["dense_columns"]],
            eta[:, base._CTX["gauge_columns"]])


def forward_batch(pool, forward, tasks):
    """pool.map(forward, tasks), path-parallel when enabled for the
    official fine dense forward map; other forward maps stay serial."""
    if not (PATH_PARALLEL and forward is forward_statistics_dense_fine):
        return pool.map(forward, tasks)
    from sde_closure_eki_dense import compute_statistics_dense
    path_tasks = [tuple(t) + (path,) for t in tasks
                  for path in range(FORWARD_PATHS)]
    raw = pool.map(forward_path_dense_fine, path_tasks, chunksize=1)
    baseline = np.asarray(base._CTX["dense_baseline"])
    out = []
    for i in range(len(tasks)):
        chunk = raw[i * FORWARD_PATHS:(i + 1) * FORWARD_PATHS]
        if any(c is None for c in chunk):
            out.append(None)
            continue
        times_s = chunk[0][0]
        count = times_s.size
        analysis, _ = base._analysis_mask(times_s, base.ANALYSIS_START_S)
        out.append(compute_statistics_dense(
            times_s, [c[1] for c in chunk], [c[2] for c in chunk],
            baseline[:count], analysis,
        ))
    return out


def increment_block(
    dense_series_paths: list[np.ndarray],
    dense_baseline_series: np.ndarray,
    analysis: np.ndarray,
) -> np.ndarray:
    """One-save-step deviation-increment std at the dense stations.

    Var[Delta d] = 2(1-rho) Var_accumulated + q_local: an independent
    mixture of the transported and the LOCALLY injected variance, so
    together with the deviation-rms block it spans the local direction
    that plain accumulated statistics cannot see.  Used by the S2 (q=151)
    observation vector.
    """

    mask = np.asarray(analysis, dtype=bool)
    incr_sq = None
    count = 0
    for series in dense_series_paths:
        deviation = series - dense_baseline_series
        delta = np.diff(deviation, axis=0)
        pair_mask = mask[1:] & mask[:-1]
        block = np.sum(delta[pair_mask] ** 2, axis=0)
        incr_sq = block if incr_sq is None else incr_sq + block
        count += int(np.sum(pair_mask))
    return np.sqrt(incr_sq / max(count, 1))


INCR_SUB = int(os.environ.get("SW_INCR_SUB", "1"))


def compute_statistics_incr(
    times_s: np.ndarray,
    dense_series_paths: list[np.ndarray],
    gauge_series_paths: list[np.ndarray],
    dense_baseline_series: np.ndarray,
    analysis: np.ndarray,
) -> np.ndarray:
    """q=151 statistic vector: dense q=111 core + 40-station increment std.

    Controlled design: the core blocks are evaluated on the STANDARD save
    cadence (subsample by SW_INCR_SUB when the run saves faster), so their
    definitions match the dense layout exactly; only the increment block
    sees the fast cadence.
    """
    s = INCR_SUB
    core = compute_statistics_dense(
        times_s[::s],
        [series[::s] for series in dense_series_paths],
        [series[::s] for series in gauge_series_paths],
        dense_baseline_series[::s],
        analysis[::s],
    )
    return np.concatenate(
        [
            core,
            increment_block(
                dense_series_paths, dense_baseline_series, analysis
            ),
        ]
    )


def twin_columns(y_physical: np.ndarray, dense: bool):
    """(hs_cols, gauge_cols) for the twin statistics on a grid."""
    profile_x = DENSE_X_M if dense else np.asarray(base.HS_X_M)
    hs_cols = np.asarray(
        [int(np.argmin(np.abs(y_physical - (4000.0 - x))))
         for x in profile_x],
        dtype=int,
    )
    gauge_cols = np.asarray(
        [int(np.argmin(np.abs(y_physical - (4000.0 - x))))
         for x in base.GAUGE_X_M],
        dtype=int,
    )
    return hs_cols, gauge_cols


def truth_stats_from_eta(
    times_s: np.ndarray,
    eta: np.ndarray,
    baseline_eta: np.ndarray,
    y_physical: np.ndarray,
    dense: bool,
    incr: bool,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """stats(record) for a (paths, T, n4) eta record -- the SAME pipeline
    the official observation y uses (spec 1.1/1.2), reusable for the
    independent reference records of the unified Gamma (spec 1.3a)."""
    hs_cols, gauge_cols = twin_columns(y_physical, dense)
    hs_series = [eta[m][:, hs_cols] for m in range(eta.shape[0])]
    gauge_series = [eta[m][:, gauge_cols] for m in range(eta.shape[0])]
    baseline_profile = baseline_eta[:, hs_cols]
    baseline_gauges = baseline_eta[:, gauge_cols]
    if mask is None:
        mask, _ = base._analysis_mask(times_s, base.ANALYSIS_START_S)
    if incr:
        return compute_statistics_incr(
            times_s, hs_series, gauge_series, baseline_profile, mask)
    if dense:
        return compute_statistics_dense(
            times_s, hs_series, gauge_series, baseline_profile, mask)
    return base.compute_statistics(
        times_s, hs_series, gauge_series, baseline_gauges, mask)


def truth_statistics_and_blocks_twin(
    bundle_path: _Path,
    dense: bool,
    incr: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Observation y (single full analysis window) + per-block stats + aux.

    The block-split statistics are retained for diagnostics only: under the
    spec, Gamma comes from the independent reference records
    (sw_ref_records.py), not from window blocks.
    """
    bundle = np.load(bundle_path, allow_pickle=True)
    times_s = np.asarray(bundle["times_s"], dtype=float)
    eta = np.asarray(bundle["eta_paths_m"], dtype=float)
    y_physical = np.asarray(bundle["y_physical_m"], dtype=float)
    profile_x = DENSE_X_M if dense else np.asarray(base.HS_X_M)
    hs_cols = np.asarray(
        [
            int(np.argmin(np.abs(y_physical - (4000.0 - x))))
            for x in profile_x
        ],
        dtype=int,
    )
    gauge_cols = np.asarray(
        [
            int(np.argmin(np.abs(y_physical - (4000.0 - x))))
            for x in base.GAUGE_X_M
        ],
        dtype=int,
    )
    baseline = np.load(
        bundle_path.parent / "baseline_data.npz", allow_pickle=True
    )
    baseline_eta = np.asarray(baseline["eta_coarse_m"], dtype=float)
    baseline_gauges = baseline_eta[:, gauge_cols]
    baseline_profile = baseline_eta[:, hs_cols]

    hs_series = [eta[m][:, hs_cols] for m in range(eta.shape[0])]
    gauge_series = [eta[m][:, gauge_cols] for m in range(eta.shape[0])]
    analysis, _ = base._analysis_mask(times_s, base.ANALYSIS_START_S)
    if incr:
        stat = lambda mask: compute_statistics_incr(
            times_s, hs_series, gauge_series, baseline_profile, mask
        )
    elif dense:
        stat = lambda mask: compute_statistics_dense(
            times_s, hs_series, gauge_series, baseline_profile, mask
        )
    else:
        stat = lambda mask: base.compute_statistics(
            times_s, hs_series, gauge_series, baseline_gauges, mask
        )
    observation = stat(analysis)
    indices = np.nonzero(analysis)[0]
    blocks = np.array_split(indices, base.N_BLOCKS)
    block_stats = []
    for block in blocks:
        mask = np.zeros_like(analysis)
        mask[block] = True
        block_stats.append(stat(mask))
    aux = {
        "times_s": times_s,
        "y_physical_m": y_physical,
        "gauge_cols": gauge_cols,
        "gauge_eta": eta[:, :, gauge_cols],
        "hs_profile": np.asarray(bundle["hs_profile_m"], dtype=float),
        "target_spread": np.asarray(
            bundle["target_spread_m"], dtype=float
        ),
        "sigma_true": np.asarray(bundle["sigma_amp_true"], dtype=float),
    }
    return observation, np.asarray(block_stats), aux


def run_validation(
    decoded: dict[str, object],
    envelope: str,
    aux: dict[str, np.ndarray],
    truth_dir: _Path,
) -> dict[str, object]:
    """8 fresh-seed posterior paths vs the twin truth (same grid)."""

    from scipy.stats import gaussian_kde

    config = ModelAConfig(boundary_seed=20260801, **FINE_MODEL_KW)
    context = CoarseModelContext(config)
    # Terrain envelope only (the S1b spline branch is not shipped).
    weight = float(decoded["phi"]) * terrain_weight(
        context.depth_ratio, float(decoded["q"])
    )
    n4 = config.coarse_n4
    paths = []
    for m in range(8):
        rng = np.random.default_rng([VALIDATION_SEED, m])
        noise = GridWhiteNoise(
            GridWhiteNoiseParameters(
                phi_amplitude=1.0,
                correlation_length_cells=CORR_CELLS,
            ),
            context.y,
            context.parameters.lambda_ref_m,
            context.surface_to_green,
            config.coarse_dt,
            rng,
            spatial_weight=weight,
        )
        solver = context.make_solver()
        _, surface, _, _ = solver.run_stochastic(
            np.zeros_like(context.y),
            config.output_stride,
            context.traces,
            noise_increment=noise,
        )
        paths.append(
            (
                np.asarray(surface[:, :n4], dtype=float)
                * context.parameters.a_ref_m
            ).astype(np.float32)
        )
    ensemble = np.stack(paths)
    times_s = aux["times_s"]
    analysis = times_s >= 600.0

    hs_post = np.mean(
        4.0 * np.std(ensemble[:, analysis, :], axis=1), axis=0
    )
    hs_rel = float(
        np.linalg.norm(hs_post - aux["hs_profile"])
        / np.linalg.norm(aux["hs_profile"])
    )
    baseline = np.load(
        truth_dir / "baseline_data.npz", allow_pickle=True
    )
    base_eta = np.asarray(baseline["eta_coarse_m"], dtype=float)
    deviation = ensemble[:, analysis, :] - base_eta[None, analysis, :]
    spread = np.sqrt(np.mean(deviation**2, axis=(0, 1)))
    target = aux["target_spread"]
    mask = target > 0.005
    spread_ratio = spread[mask] / target[mask]

    grid = np.linspace(-0.45, 0.55, 400)
    kde_l1 = []
    for gi in range(5):
        col = aux["gauge_cols"][gi]
        truth_samples = aux["gauge_eta"][:, analysis, gi].reshape(-1)
        post_samples = ensemble[:, analysis, col].reshape(-1)
        kde_l1.append(
            float(
                np.trapezoid(
                    np.abs(
                        gaussian_kde(post_samples)(grid)
                        - gaussian_kde(truth_samples)(grid)
                    ),
                    grid,
                )
            )
        )
    return {
        "hs_profile_relative_l2_vs_twin_truth": hs_rel,
        "spread_over_target_p10_p50_p90": [
            float(np.percentile(spread_ratio, p)) for p in (10, 50, 90)
        ],
        "kde_l1_at_gauges_x0_500_1000_2000_3000": kde_l1,
        "validation_seed": VALIDATION_SEED,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=("S1a",), required=True)
    parser.add_argument("--members", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--processes", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    if not SW_FINE:
        raise SystemExit(
            "the spec chain is fine-grid only: launch with SW_FINE=1 "
            "(the spec prior for phi is defined in log space on the fine "
            "grid)"
        )

    variant = arguments.config
    if _VARIANT != variant:
        raise SystemExit(
            f"launch with SW_S1_VARIANT={variant} in the environment so "
            "the twin baseline is wired before module import "
            f"(got {_VARIANT!r})"
        )
    envelope = "terrain"
    mode, colored, flat_sigma, no_damping = "B", False, False, True
    truth_dir = OUT_ROOT / f"truth_{variant}{_SUFFIX}"
    outtag = os.environ.get("SW_S1_OUTTAG", "")
    output = OUT_ROOT / f"{variant}_eki_dense{_SUFFIX}{outtag}"
    if output.exists() and any(output.iterdir()) and not arguments.overwrite:
        raise SystemExit(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)

    members = arguments.members or 60
    max_iterations = arguments.iterations or 16
    duration_s = v3_world.DURATION_S
    if arguments.smoke:
        members, max_iterations, duration_s = 8, 2, 240.0
    tag = f"SW_{variant}_dense"

    names = base.theta_layout(mode, envelope, colored, flat_sigma,
                              no_damping)
    print(
        f"[{tag}] twin inversion: J={members} dims={len(names)} "
        f"N_G={N_G} envelope={envelope} "
        f"baseline={os.environ['SDE_BASELINE_NPZ']}"
    )
    observation, block_stats, aux = truth_statistics_and_blocks_twin(
        truth_dir / "truth_bundle.npz", dense=True, incr=False
    )
    q_obs = observation.size
    print(f"[{tag}] q_obs={q_obs}")
    forward = forward_statistics_dense_fine
    initializer = _init_worker_dense

    rng = np.random.default_rng(
        [base.EKI_SEED, 72, ord(mode), len(envelope)]
    )
    thetas = base.initial_ensemble(
        mode, envelope, members, rng, colored, flat_sigma, no_damping
    )
    # Pre-declared deviation (fine grid): the coarse-grid phi prior
    # U(0.005, 0.1) relies on coarse-grid numerical dissipation for
    # stability -- on 3073/0.002 most paths at phi ~ 0.05 blow up.
    # The fine prior U(0.003, 0.03) still spans an order of magnitude
    # around the truth 0.0124.  phi lives in log coordinates inside EKI,
    # so the prior interval is drawn in log space.
    thetas[:, 0] = rng.uniform(np.log(0.003), np.log(0.03), size=members)

    pool = Pool(
        processes=arguments.processes,
        initializer=initializer,
        initargs=(duration_s, 20260801, False),
    )
    if PATH_PARALLEL:
        initializer(duration_s, 20260801, False)   # main-process ctx
        print(f"[{tag}] path-parallel evaluation: {FORWARD_PATHS} tasks "
              f"per member, {arguments.processes} processes")
    try:
        # ------------------------------------------------ Gamma (spec)
        import sw_gamma_unified as gu
        theta_repeat = np.mean(thetas, axis=0)
        started = time.perf_counter()
        repeat_stats = None
        fwd_cache = gu.fwd_cache_path(
            OUT_ROOT, tag, theta_repeat, base.EKI_SEED, FORWARD_PATHS,
            N_FWD_REPEATS, "dense")
        cached_fwd = gu.load_fwd_cache(fwd_cache, q_obs)
        if cached_fwd is not None:
            repeat_stats = list(cached_fwd)
            print(f"[{tag}] forward repeats loaded from cache "
                  f"({len(repeat_stats)} repeats, {fwd_cache.name}); "
                  f"Gamma rebuilt with current settings")
        if repeat_stats is None:
            repeat_tasks = [
                (theta_repeat, mode, envelope, colored, flat_sigma,
                 no_damping, (base.EKI_SEED, 2999, repeat))
                for repeat in range(N_FWD_REPEATS)
            ]
            repeat_stats = [
                s for s in forward_batch(pool, forward, repeat_tasks)
                if s is not None
            ]
            if len(repeat_stats) < max(3, N_FWD_REPEATS // 2):
                raise RuntimeError("too many failed Gamma repeats")
            gu.save_fwd_cache(fwd_cache, np.asarray(repeat_stats),
                              theta_repeat, note="S1 prior-mean repeats")
        ref_stats = gu.load_ref_stats(truth_dir, "dense", q_obs)
        gamma, gamma_diag_info = gu.build_gamma_unified(
            observation, ref_stats, np.asarray(repeat_stats),
            "dense", n_dense=N_DENSE,
        )
        print(gu.acceptance_line(gamma_diag_info))
        gamma_inverse = np.linalg.inv(gamma)
        print(
            f"[{tag}] Gamma (spec, var_ref only) from "
            f"{np.asarray(ref_stats).shape[0]} reference records, "
            f"N_F={len(repeat_stats)} diagnostic repeats "
            f"({time.perf_counter()-started:.0f}s)"
        )

        # ------------------------------------------ engine injections
        sentinel = 10.0 * np.sqrt(np.diag(gamma))

        # Audit bookkeeping shared between the evaluator and the summary:
        # phis are recomputed here with the exact origin formula
        # (0.5 r^T Gamma^-1 r via the explicit inverse) so the audit files
        # stay bit-identical to the origin's.  The engine computes its own
        # Phi by Cholesky-whitening the exact Gamma (same mathematics, no
        # regularisation either), which differs from this inv-based
        # arithmetic at the last-ulp level only -- so the engine-internal
        # stop-rule comparison is last-ulp different from the origin loop,
        # while every saved number comes from this audit path.
        audit = {"phi_member": [], "phi_ens": [], "best_phi": np.inf,
                 "best_theta": None}

        def evaluator(theta_ens, iteration):
            """One EKI evaluation: J x N_G forward tasks, CRN seeds shared
            across members, mean over the N_G realisations per member.
            Members whose realisations ALL failed return a NaN row (the
            engine's failed_mask_fn/sentinel_row_fn replace it)."""
            started = time.perf_counter()
            n_members = theta_ens.shape[0]
            tasks = [
                (theta_ens[m], mode, envelope, colored, flat_sigma,
                 no_damping, spec_seed_root(base.EKI_SEED, iteration, a))
                for m in range(n_members) for a in range(N_G)
            ]
            flat = forward_batch(pool, forward, tasks)
            rows = np.empty((n_members, q_obs), dtype=float)
            n_lost, lost_members = 0, 0
            for m in range(n_members):
                ok = [flat[m * N_G + a] for a in range(N_G)
                      if flat[m * N_G + a] is not None]
                n_lost += N_G - len(ok)
                if ok:
                    rows[m] = np.mean(np.asarray(ok), axis=0)
                else:
                    rows[m] = np.nan
                    lost_members += 1
            if n_lost:
                print(f"[{tag}] {n_lost}/{n_members * N_G} realisations "
                      f"failed ({lost_members} members lost entirely)")
            # Audit copy with the declared penalty rows swapped in (the
            # same replacement the engine applies afterwards).
            g_matrix = rows.copy()
            failed = ~np.all(np.isfinite(g_matrix), axis=1)
            g_matrix[failed] = observation + sentinel
            residuals = g_matrix - observation
            phis = 0.5 * np.einsum(
                "ij,jk,ik->i", residuals, gamma_inverse, residuals
            )
            r_bar = g_matrix.mean(axis=0) - observation
            phi_ensemble = float(0.5 * r_bar @ gamma_inverse @ r_bar)
            np.savez(output / f"iter_{iteration:03d}.npz",
                     thetas=theta_ens, g_matrix=g_matrix, phis=phis,
                     phi_ensemble=phi_ensemble)
            audit["phi_member"].append(phis.copy())
            audit["phi_ens"].append(phi_ensemble)
            for m in np.nonzero(~failed)[0]:
                if phis[m] < audit["best_phi"]:
                    audit["best_phi"] = float(phis[m])
                    audit["best_theta"] = theta_ens[m].copy()
            valid_phis = phis[~failed] if (~failed).any() else phis
            print(
                f"[{tag}] iter {iteration}: Phi(mean G)={phi_ensemble:.2f} "
                f"mean Phi={float(np.mean(valid_phis)):.2f} "
                f"min={float(np.min(valid_phis)):.2f} "
                f"({time.perf_counter()-started:.0f}s)"
            )
            return rows

        def failed_mask(outputs):
            """Spec rule: a member fails when ALL its realisations failed
            (the evaluator marks that with a NaN row)."""
            return ~np.all(np.isfinite(outputs), axis=1)

        def sentinel_row(y, gamma_diag):
            """Declared penalty row for failed members."""
            return y + 10.0 * np.sqrt(gamma_diag)

        def spec_clip(theta_ens, iteration):
            """Safety bounds only (log-phi in [log 1e-4, log 0.5],
            q in [0, 6]); the spec has no other clipping."""
            clipped = np.array(theta_ens, dtype=float, copy=True)
            clipped[:, 0] = np.clip(clipped[:, 0],
                                    np.log(1e-4), np.log(0.5))
            if clipped.shape[1] > 1:
                clipped[:, 1] = np.clip(clipped[:, 1], 0.0, 6.0)
            return clipped

        # jitter=0.0 reproduces the origin's un-regularised Kalman-gain
        # solve.  For the objective it is redundant-but-harmless: the
        # engine always computes Phi with the exact Gamma (Cholesky
        # whitening), independent of the jitter setting.
        result = run_eki(
            thetas,
            None,
            observation,
            gamma,
            n_iter=max_iterations,
            rng=rng,
            perturb_observations=True,
            jitter=0.0,
            verbose=True,
            ensemble_evaluator=evaluator,
            stop_rel_tol=0.01,
            stop_patience=3,
            sentinel_row_fn=sentinel_row,
            failed_mask_fn=failed_mask,
            post_update=spec_clip,
            jitter_mode="absolute",
        )
        print(f"[{tag}] engine stop: {result.stop_reason} "
              f"({result.n_updates} updates applied)")
    finally:
        pool.close()
        pool.join()

    thetas = result.final_ensemble
    # In-loop evaluations (origin meaning): the engine adds one extra
    # evaluation of the final ensemble when the iteration cap is reached.
    iterations_run = int(min(len(audit["phi_ens"]), max_iterations))

    model_ensemble = np.array([model_theta(t) for t in thetas])
    if audit["best_theta"] is None:
        raise SystemExit(f"[{tag}] every member failed at every iteration")
    decoded_best = base.decode_theta(
        model_theta(audit["best_theta"]), mode, envelope, colored,
        flat_sigma, no_damping
    )
    # Spec reporting: final ensemble mean +/- sd, in MODEL coordinates
    # (the log-phi ensemble is exponentiated first), no selection.
    decoded_mean = base.decode_theta(
        np.mean(model_ensemble, axis=0),
        mode, envelope, colored, flat_sigma, no_damping,
    )
    final_mean_model = np.mean(model_ensemble, axis=0)
    final_sd_model = np.std(model_ensemble, axis=0, ddof=1)

    # ---- recovery vs truth ----
    metadata = json.loads(
        (truth_dir / "metadata.json").read_text(encoding="utf-8")
    )
    recovery = {
        "phi_true": metadata["weight"]["phi_true"],
        "phi_best": decoded_best["phi"],
        "phi_mean": decoded_mean["phi"],
        "phi_rel_error_best": abs(
            decoded_best["phi"] - metadata["weight"]["phi_true"]
        ) / metadata["weight"]["phi_true"],
        "q_true": metadata["weight"]["q_true"],
        "q_best": decoded_best["q"],
        "q_mean": decoded_mean["q"],
        "q_abs_error_best": abs(
            decoded_best["q"] - metadata["weight"]["q_true"]
        ),
    }
    validation = {}
    if not arguments.smoke:
        try:
            # Spec: the reported parameter set is the final ensemble
            # mean, so the validation paths use it as well.
            validation = run_validation(decoded_mean, envelope, aux,
                                        truth_dir)
        except Exception as error:  # noqa: BLE001 - never lose the EKI
            validation = {"validation_error": repr(error)}
            print(f"[{tag}] WARNING validation failed: {error!r}")

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "driver": "sw_eki_s1",
        "config": variant,
        "dense": True,
        "incr": False,
        "envelope": envelope,
        "members": members,
        "iterations_run": iterations_run,
        "n_updates": int(result.n_updates),
        "stop_reason": result.stop_reason,
        "phi_best": audit["best_phi"],
        "q_obs": int(q_obs),
        "decoded_best": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in decoded_best.items()
        },
        "decoded_final_mean": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in decoded_mean.items()
        },
        "final_parameter_spread": final_sd_model.tolist(),
        "final_parameter_mean": final_mean_model.tolist(),
        "algorithm": "spec_2026-08-23",
        "parameter_choice": "final_ensemble_mean",
        "N_G": N_G,
        "phi_ensemble_history":
            [float(p) for p in audit["phi_ens"][:iterations_run]],
        "phi_ensemble_final_evaluation": (
            float(audit["phi_ens"][-1])
            if len(audit["phi_ens"]) > iterations_run else None
        ),
        "log_coordinates": ["phi"],
        "parameter_names": names,
        "recovery": recovery,
        "validation": {
            k: v for k, v in validation.items()
            if not k.startswith("_")
        },
        "truth_metadata": metadata,
        "forward_paths": FORWARD_PATHS,
        "gamma_mode": "unified",
    }
    if gamma_diag_info is not None:
        import sw_gamma_unified as gu
        summary.update(gu.summary_fields(gamma_diag_info))
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"[{tag}] best Phi={audit['best_phi']:.2f} "
          f"recovery={json.dumps(recovery)[:200]}")
    if validation and "hs_profile_relative_l2_vs_twin_truth" in validation:
        print(
            f"[{tag}] validation: Hs rel "
            f"{validation['hs_profile_relative_l2_vs_twin_truth']:.4f}, "
            f"spread ratio p10/50/90 "
            f"{validation['spread_over_target_p10_p50_p90']}, "
            f"KDE L1 {validation['kde_l1_at_gauges_x0_500_1000_2000_3000']}"
        )


if __name__ == "__main__":
    main()
