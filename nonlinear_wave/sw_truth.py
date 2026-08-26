"""Twin-truth generation for the Chapter-5 vKdV spec chain (stage 1).
Generates the KNOWN-phi stochastic vKdV truth bundle that both steps of
the stepwise ladder observe:

  S1-a  terrain-law amplitude  sigma(y) = phi* (d_min/d)^q*,
        phi* = 0.0124, q* = 2.35  (scheme-B posterior values).

The bundle holds ``--paths`` stochastic paths (the v3 spec world uses 1)
on the requested grid, plus ONE deterministic (no-noise) baseline run that
defines the twin "deviation-from-baseline" observable and the spread
target.

Seeds (pre-registered): boundary phase 20260801; path m uses
[20260801, m]; nothing overlaps the reference-record seed space.

Outputs (under results/stepwise/<version dir>/truth_S1a<suffix>/):
  truth_bundle.npz    times, grids, eta paths, gauge series, Hs profile,
                      spread target, true sigma(y)
  baseline_data.npz   deterministic baseline in the key layout consumed by
                      the frozen worker init (SDE_BASELINE_NPZ)
  metadata.json       seeds, grid, template, truth parameters

Usage (spec chain)::

    SW_FINE=1 SW_VERSION_DIR=v3spec SW_DURATION_S=6600 \
    SW_SYNTH_PERIOD_S=6600 SW_FORWARD_PATHS=1 \
    python sw_truth.py --variant S1a --suffix _fine --paths 1
"""

from __future__ import annotations

import argparse
import os
import json

# SW_FINE=1 -> production fine grid via the OFFICIAL closure_config env
# switches (they must be set BEFORE the frozen modules import, because
# ModelAConfig dataclass defaults are read at class-definition time).
# The DRP template is coarse-tuned; fine runs use the frozen standard
# template (SDE_CLOSURE_V3=0) per closure_config's own doctrine.
SW_FINE = os.environ.get("SW_FINE", "") == "1"
# SW_GRID=1537 -> workhorse-certification tier (results/stepwise/cert_1537)
SW_GRID = os.environ.get("SW_GRID", "")
if SW_GRID == "1537":
    os.environ["SDE_COARSE_N4"] = "1537"
    os.environ["SDE_COARSE_DT"] = "0.004"
    os.environ["SDE_OUTPUT_STRIDE"] = "35"
    os.environ["SDE_CLOSURE_V3"] = "0"
elif SW_FINE:
    os.environ["SDE_COARSE_N4"] = "3073"
    os.environ["SDE_COARSE_DT"] = "0.002"
    os.environ["SDE_OUTPUT_STRIDE"] = "70"
    os.environ["SDE_CLOSURE_V3"] = "0"  # standard C6/C4 template on fine

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import v3_world  # noqa: E402  (v3 twin world patches; no-op for v2)
v3_world.ensure_patched()
import sde_closure_config as closure_config
from sde_closure_context import CoarseModelContext, ModelAConfig
from sde_closure_core import (
    GridWhiteNoise,
    GridWhiteNoiseParameters,
    terrain_weight,
)

PROJECT_DIR = Path(__file__).resolve().parent
OUT_ROOT = PROJECT_DIR / "results" / "stepwise" / v3_world.version_dir(
    "cert_1537" if SW_GRID == "1537"
    else ("v2_fine" if SW_FINE else "v1_coarse")
)

if SW_GRID == "1537":
    FINE_MODEL_KW = {"coarse_n4": 1537, "coarse_dt": 0.004,
                     "output_stride": 35}
elif SW_FINE:
    FINE_MODEL_KW = {"coarse_n4": 3073, "coarse_dt": 0.002,
                     "output_stride": 70}
else:
    FINE_MODEL_KW = {}
# Noise spatial correlation is defined in GRID CELLS in the frozen
# core; the plan's physical intent is "3 cells ~ 31 m" ON THE COARSE
# grid.  On the fine grid 3 cells = 3.9 m -- an 8x weaker
# Hilbert--Schmidt regularisation (a DIFFERENT SPDE) and exactly the
# documented grid-scale blow-up regime.  The correlation length is held
# PHYSICAL (31.25 m); cells = 31.25 / dx.
if SW_GRID == "1537":
    CORR_CELLS = 12.0
elif SW_FINE:
    CORR_CELLS = 24.0
else:
    CORR_CELLS = 3.0

BOUNDARY_SEED = 20260801
N_PATHS = 16
PHI_TRUE = 0.0124
Q_TRUE = 2.35
GAUGE_X_M = np.asarray([0.0, 500.0, 1000.0, 2000.0, 3000.0, 4000.0])


def truth_weight(variant: str, context: CoarseModelContext) -> tuple[
    np.ndarray, dict[str, object]
]:
    """Return sigma_amp on the full grid and its metadata (S1a only)."""

    if variant != "S1a":
        raise ValueError(f"only the S1a terrain-law truth is shipped "
                         f"(got {variant!r})")
    weight = PHI_TRUE * terrain_weight(context.depth_ratio, Q_TRUE)
    meta = {"form": "terrain", "phi_true": PHI_TRUE, "q_true": Q_TRUE}
    return weight, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("S1a",), required=True)
    parser.add_argument("--paths", type=int, default=N_PATHS)
    parser.add_argument(
        "--suffix", type=str, default="",
        help="output dir suffix, e.g. _fine for the fine-grid twin world",
    )
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()

    output = OUT_ROOT / f"truth_{arguments.variant}{arguments.suffix}"
    if output.exists() and any(output.iterdir()) and not arguments.overwrite:
        raise SystemExit(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)

    config = ModelAConfig(boundary_seed=BOUNDARY_SEED, **FINE_MODEL_KW)
    context = CoarseModelContext(config)
    weight, weight_meta = truth_weight(arguments.variant, context)
    n4 = config.coarse_n4
    y_physical_m = context.y_physical_m
    gauge_cols = np.asarray(
        [
            int(np.argmin(np.abs(y_physical_m - (4000.0 - x))))
            for x in GAUGE_X_M
        ],
        dtype=int,
    )

    # Deterministic twin baseline (no noise): defines the deviation
    # observable and the spread target of this twin world.
    started = time.perf_counter()
    solver = context.make_solver()
    times, surface, _, residuals = solver.run(
        np.zeros_like(context.y), config.output_stride, context.traces
    )
    times_s = np.asarray(times) * context.parameters.time_ref_s
    baseline_eta = (
        np.asarray(surface[:, :n4], dtype=float)
        * context.parameters.a_ref_m
    )
    print(
        f"[{arguments.variant}] baseline {time.perf_counter()-started:.0f}s "
        f"max|eta|={np.max(np.abs(baseline_eta)):.4f} m"
    )

    # Stochastic truth paths.
    eta_paths = np.empty(
        (arguments.paths, times_s.size, n4), dtype=np.float32
    )
    for path in range(arguments.paths):
        rng = np.random.default_rng([BOUNDARY_SEED, path])
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
        eta = (
            np.asarray(surface[:, :n4], dtype=float)
            * context.parameters.a_ref_m
        )
        if not np.all(np.isfinite(eta)):
            raise FloatingPointError(f"truth path {path} non-finite")
        eta_paths[path] = eta.astype(np.float32)
        print(f"[{arguments.variant}] path {path+1}/{arguments.paths} done")

    analysis = times_s >= 600.0
    hs_truth = np.mean(
        4.0 * np.std(eta_paths[:, analysis, :], axis=1), axis=0
    )
    # Twin spread target: std over paths of (path - baseline), pooled in
    # time -- what a perfectly calibrated ensemble should reproduce.
    deviation = eta_paths[:, analysis, :] - baseline_eta[None, analysis, :]
    target_spread = np.sqrt(np.mean(deviation**2, axis=(0, 1)))

    np.savez_compressed(
        output / "truth_bundle.npz",
        times_s=times_s,
        y_physical_m=y_physical_m,
        x_physical_m=context.x_physical_m,
        gauge_x_m=GAUGE_X_M,
        gauge_columns=gauge_cols,
        eta_paths_m=eta_paths,
        gauge_eta_m=eta_paths[:, :, gauge_cols],
        hs_profile_m=hs_truth,
        target_spread_m=target_spread,
        sigma_amp_true=weight,
    )
    # Baseline bundle in the standard key layout consumed by the frozen
    # worker init (SDE_BASELINE_NPZ): the "oracle" keys carry the twin
    # error profile on the coarse grid.
    np.savez_compressed(
        output / "baseline_data.npz",
        times_s=times_s,
        y_physical_m=y_physical_m,
        x_physical_m=context.x_physical_m,
        eta_coarse_m=baseline_eta,
        energy_m3=np.trapezoid(
            baseline_eta**2, y_physical_m, axis=1
        ),
        y_ref_fine_m=y_physical_m,
        error_std_profile_fine_m=target_spread,
        hs_fine_ref_m=hs_truth,
    )
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "command": " ".join(sys.argv),
                "variant": arguments.variant,
                "noise_correlation_cells": CORR_CELLS,
                "template": (
                    "DRP band-optimised" if closure_config.V3
                    else "standard"
                ),
                "weight": weight_meta,
                "boundary_seed": BOUNDARY_SEED,
                "path_seeds": f"[{BOUNDARY_SEED}, m] m=0..{arguments.paths-1}",
                "n_paths": arguments.paths,
                "grid": {
                    "coarse_n4": n4,
                    "coarse_dt": config.coarse_dt,
                    "output_stride": config.output_stride,
                },
                "analysis_start_s": 600.0,
                "max_boundary_residual": float(np.max(np.abs(residuals))),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[{arguments.variant}] bundle written to {output}")


if __name__ == "__main__":
    main()
