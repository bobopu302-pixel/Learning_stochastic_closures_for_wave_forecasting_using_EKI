"""GPR reference track: nonlinear-tendency regression from twin fields.
Turns the nonlinear-term learning into a REGRESSION problem (only
possible in the twin setting where full fields are stored):

    r_n(y) = (u_{n+1} - E_dt[u_n]) / dt  ~  N(u_n, du_n/dy)(y) + noise

where E_dt is the EXACT linear evolution over one save interval,
realised by running the frozen linear solver (GP drift disabled) for
one save step from each snapshot -- this strips dispersion from the
residual so the target is nonlinearity + zero-mean noise increment.

The (u, s = u_y/s0) samples are binned on a grid, bin means computed
(the ~10^6-10^7 samples crush the noise).  Outputs:

  results/stepwise/<ver>/GPR_reference/
      calibration.json        t0 (tendency scale), s0 (slope scale) --
                              consumed by sw_eki_h.load_calibration
      surface.npz             bin grid, bin means, counts, sigma of mean
      reference_bundle.npz    full-grid stochastic reference paths
      reference_baseline.npz  full-grid deterministic run (machinery test)

Usage (spec chain)::

    SW_FINE=1 SW_VERSION_DIR=v3spec SW_DURATION_S=6600 \
    SW_SYNTH_PERIOD_S=6600 SW_FORWARD_PATHS=1 \
    python sw_gpr_reference.py --paths 1
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
    os.environ["SDE_CLOSURE_V3"] = "0"

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np

import v3_world  # noqa: E402  (v3 twin world patches; no-op for v2)
v3_world.ensure_patched()
import sde_closure_config as closure_config
from sde_closure_context import CoarseModelContext, ModelAConfig
from sw_linear_core import LambdaFluxSolver  # zero-lambda = linear core

_VERSION_DIR = v3_world.version_dir("v2_fine" if SW_FINE else "v1_coarse")
_TRUTH_NAME = "truth_S1a_fine" if SW_FINE else "truth_S1a"
OUT_ROOT = _HERE / "results" / "stepwise" / _VERSION_DIR
TRUTH_DIR = OUT_ROOT / _TRUTH_NAME
OUT_DIR = OUT_ROOT / "GPR_reference"
EPSILON = 0.01


def far_tail_taper(state: np.ndarray, n_taper: int = 300) -> np.ndarray:
    """Zero the far guard end smoothly (DABC precondition).

    The frozen solver requires ~zero exterior initial data at the
    outflow.  The tapered zone is 6 km from the physical window; even
    the fastest upstream numerical branch (|c_g| ~ 1.5e2 nondim)
    covers only ~1.5 km within one half-cadence interval, so the
    taper cannot influence the physical window.
    """

    out = state.copy()
    ramp = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, n_taper)))
    out[-n_taper:] = out[-n_taper:] * ramp[::-1]
    out[-6:] = 0.0
    return out


def linear_step_operator(context, config, n_substeps: int):
    """Return a function u -> exact linear evolution over one save."""

    with closure_config.template():
        solver = LambdaFluxSolver(
            context.y, context.depth_ratio,
            context.parameters.epsilon, context.parameters.mu,
            config.coarse_dt, n_substeps,
        )
    solver.set_lambda(np.zeros(5))

    n4 = config.coarse_n4
    a_ref = context.parameters.a_ref_m

    t_ref = context.parameters.time_ref_s

    def evolve(eta_m: np.ndarray, t_start_s: float) -> np.ndarray:
        """One save-interval of linear dynamics from a physical field.

        Uses the TRUE incident traces time-shifted to the snapshot
        instant: the frozen chain reconstructs a GLOBAL incident field
        from the boundary record (three-trace lifting), so a zero-trace
        evolution is a different linear operator (measured: amplitude-
        proportional residual contamination ~6.7x the nonlinear
        signal, uniform in x).  Sharing the truth's traces makes the
        linear step identical up to the O(dt^2) fixed-point coupling.
        Guard content is still zero-initialised (outgoing only).
        """

        t0_nd = t_start_s / t_ref
        shifted = tuple(
            (lambda t, f=f: f(np.asarray(t, dtype=float) + t0_nd))
            for f in context.traces
        )
        if eta_m.size == context.y.size:
            state = eta_m / a_ref          # full computational state
        else:
            state = np.zeros_like(context.y)
            state[:n4] = eta_m / a_ref     # legacy physical-slice mode
        state = far_tail_taper(state)
        # run() takes the SURFACE field and normalises internally --
        # passing to_normalized(state) double-applies S = d^{1/4}
        # (no error on the offshore flat where S = 1, growing down
        # the slope: the exact geography of the contamination).
        _, surface, _, _ = solver.run(
            state, n_substeps, shifted
        )
        return np.asarray(surface[-1, :n4], dtype=float) * a_ref

    return evolve


def base_traces_zero(context):
    """Zero boundary traces (tuple of callables t -> 0)."""

    def zero(t):
        return 0.0 * np.asarray(t, dtype=float)

    return tuple(zero for _ in context.traces)


def generate_reference_bundle(n_paths: int = 4,
                              stride: int = 35):
    """Dedicated full-grid reference bundle for residual extraction.

    The EKI twin bundle stores only the physical window; the vKdV
    short-wave branch has NEGATIVE group velocity (any wavelength
    below ~21 m travels upstream, |c_g| up to ~10^2 nondim), so guard
    content causally influences the whole slope within one save
    interval and residual extraction REQUIRES the full computational
    state.  Same seeds as the twin ([20260801, m]) so the physical
    window is the identical realization; finer save cadence halves the
    in-interval dispersion smear.
    """

    from sde_closure_context import CoarseModelContext, ModelAConfig
    from sde_closure_core import (
        GridWhiteNoise,
        GridWhiteNoiseParameters,
        terrain_weight,
    )

    out = OUT_DIR / "reference_bundle.npz"
    if out.exists():
        return out
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    corr = 24.0 if SW_FINE else 3.0
    config = ModelAConfig(boundary_seed=20260801)
    context = CoarseModelContext(config)
    weight = 0.0124 * terrain_weight(context.depth_ratio, 2.35)
    fields = []
    times_s = None
    for m in range(n_paths):
        rng = np.random.default_rng([20260801, m])
        noise = GridWhiteNoise(
            GridWhiteNoiseParameters(
                phi_amplitude=1.0, correlation_length_cells=corr
            ),
            context.y, context.parameters.lambda_ref_m,
            context.surface_to_green, config.coarse_dt, rng,
            spatial_weight=weight,
        )
        solver = context.make_solver()
        t, surface, _, _ = solver.run_stochastic(
            np.zeros_like(context.y), stride, context.traces,
            noise_increment=noise,
        )
        times_s = np.asarray(t) * context.parameters.time_ref_s
        full = (np.asarray(surface, dtype=float)
                * context.parameters.a_ref_m).astype(np.float32)
        fields.append(full)
        print(f"[gpr-ref] reference path {m+1}/{n_paths} done")
    np.savez_compressed(
        out, times_s=times_s,
        eta_full_m=np.stack(fields),
        stride=stride, n_grid=context.y.size,
        note="full computational grid incl. guard; seeds [20260801,m]",
    )
    print(f"[gpr-ref] reference bundle written: {out}")
    return out


def generate_reference_baseline(stride: int = 35):
    """Deterministic full-grid run (no noise) for the machinery test."""

    from sde_closure_context import CoarseModelContext, ModelAConfig

    out = OUT_DIR / "reference_baseline.npz"
    if out.exists():
        return out
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = ModelAConfig(boundary_seed=20260801)
    context = CoarseModelContext(config)
    solver = context.make_solver()
    t, surface, _, _ = solver.run(
        np.zeros_like(context.y), stride, context.traces
    )
    times_s = np.asarray(t) * context.parameters.time_ref_s
    np.savez_compressed(
        out, times_s=times_s,
        eta_full_m=(np.asarray(surface, dtype=float)
                    * context.parameters.a_ref_m
                    ).astype(np.float32)[None, :, :],
        stride=stride,
    )
    print(f"[gpr-ref] deterministic baseline written: {out}")
    return out


def machinery_test(context, config, ref: dict, n_checks: int = 3):
    """Re-evolve snapshots with the FULL NONLINEAR solver.

    If state injection + shifted traces + DABC restart are sound, the
    re-evolution must reproduce the next stored snapshot almost
    exactly (residual limits: lost DABC memory at the far guard end
    and fixed-point tolerance).  This validates the measurement
    apparatus independently of the linear-strip question.
    """

    times_s = ref["times_s"]
    eta_full = ref["eta_full_m"]
    a_ref = context.parameters.a_ref_m
    t_ref = context.parameters.time_ref_s
    stride = int(ref["stride"])
    # Short-horizon nonlinear solver: n_steps must equal ONE save
    # interval (make_solver()'s horizon is the full run --
    # output_stride only controls saving, not the march length).
    from sde_closure_core import StochasticImplicitMidpointDABCSolver
    with closure_config.template():
        solver = StochasticImplicitMidpointDABCSolver(
            context.y, context.depth_ratio,
            context.parameters.epsilon, context.parameters.mu,
            config.coarse_dt, stride,
        )
    n4 = config.coarse_n4
    # Only pairs spanning a FULL stride interval (the run's total
    # step count need not divide by the stride; the final saved
    # interval can be partial).
    dt_save = float(np.median(np.diff(times_s)))
    regular = np.where(
        np.abs(np.diff(times_s) - dt_save) < 1e-6
    )[0]
    regular = regular[times_s[regular] >= 600.0]
    idx = regular[np.linspace(0, regular.size - 1, n_checks)
                  .astype(int)]
    worst = 0.0
    for n in idx:
        t0_nd = float(times_s[n]) / t_ref
        shifted = tuple(
            (lambda t, f=f: f(np.asarray(t, dtype=float) + t0_nd))
            for f in context.traces
        )
        state = far_tail_taper(
            eta_full[0, n, :].astype(float) / a_ref
        )
        _, surface, _, _ = solver.run(state, stride, shifted)
        got = np.asarray(surface[-1, :n4], dtype=float) * a_ref
        want = eta_full[0, n + 1, :n4].astype(float)
        rel = float(
            np.linalg.norm(got - want) / np.linalg.norm(want)
        )
        worst = max(worst, rel)
        print(f"[gpr-ref] machinery test t={times_s[n]:.1f}s: "
              f"nonlinear re-evolve rel L2 {rel:.2e}")
    return worst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=4)
    parser.add_argument("--u-bins", type=int, default=25)
    parser.add_argument("--s-bins", type=int, default=21)
    parser.add_argument("--skip-test", action="store_true")
    arguments = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = ModelAConfig(boundary_seed=20260801)
    context = CoarseModelContext(config)
    n4 = config.coarse_n4
    bundle_path = generate_reference_bundle(arguments.paths)
    ref = dict(np.load(bundle_path, allow_pickle=True))
    times_s = np.asarray(ref["times_s"], dtype=float)
    eta_full = np.asarray(ref["eta_full_m"], dtype=np.float32)
    stride = int(ref["stride"])
    a_ref = context.parameters.a_ref_m
    t_ref = context.parameters.time_ref_s
    dt_nd = float(times_s[1] - times_s[0]) / t_ref
    y_nd = np.asarray(context.y[:n4], dtype=float)
    dy_nd = float(y_nd[1] - y_nd[0])

    if not arguments.skip_test:
        base_ref = dict(np.load(generate_reference_baseline(),
                                allow_pickle=True))
        worst = machinery_test(
            context, config,
            {"times_s": np.asarray(base_ref["times_s"], dtype=float),
             "eta_full_m": np.asarray(base_ref["eta_full_m"],
                                      dtype=np.float32),
             "stride": int(base_ref["stride"])})
        if worst > 0.02:
            raise SystemExit(
                f"machinery test failed (worst rel {worst:.3e}) -- "
                "fix the apparatus before extracting residuals"
            )

    evolve = linear_step_operator(context, config, stride)
    dt_save_chk = float(np.median(np.diff(times_s)))
    regular = np.abs(np.diff(times_s) - dt_save_chk) < 1e-6
    idx_analysis = np.where((times_s[:-1] >= 600.0) & regular)[0]

    sample_u = eta_full[0, idx_analysis[:20], :n4].astype(float) / a_ref
    s0 = float(np.std(np.gradient(sample_u, dy_nd, axis=1)))
    print(f"[gpr-ref] s0 (slope scale) = {s0:.4f}")

    u_edges = np.linspace(-2.8, 2.8, arguments.u_bins + 1)
    s_edges = np.linspace(-2.0, 2.0, arguments.s_bins + 1)
    sums = np.zeros((arguments.u_bins, arguments.s_bins))
    sums_sq = np.zeros_like(sums)
    counts = np.zeros_like(sums)

    started = time.perf_counter()
    total = 0
    for path in range(eta_full.shape[0]):
        for n in idx_analysis:
            full_now = eta_full[path, n, :].astype(float)
            u_next = eta_full[path, n + 1, :n4].astype(float)
            linear_next = evolve(full_now, float(times_s[n]))
            residual_nd = (
                (u_next - linear_next) / a_ref
            ) / dt_nd
            u_nd = full_now[:n4] / a_ref
            s_nd = np.gradient(u_nd, dy_nd) / s0
            sl = slice(30, n4 - 30)
            iu = np.digitize(u_nd[sl], u_edges) - 1
            js = np.digitize(s_nd[sl], s_edges) - 1
            ok = (
                (iu >= 0) & (iu < arguments.u_bins)
                & (js >= 0) & (js < arguments.s_bins)
                & np.isfinite(residual_nd[sl])
            )
            np.add.at(sums, (iu[ok], js[ok]), residual_nd[sl][ok])
            np.add.at(sums_sq, (iu[ok], js[ok]),
                      residual_nd[sl][ok] ** 2)
            np.add.at(counts, (iu[ok], js[ok]), 1.0)
            total += int(ok.sum())
        print(f"[gpr-ref] path {path+1} done "
              f"({time.perf_counter()-started:.0f}s, "
              f"{total/1e6:.1f}M samples)")

    means = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    u_centres = 0.5 * (u_edges[1:] + u_edges[:-1])
    s_centres = 0.5 * (s_edges[1:] + s_edges[:-1])
    good = counts > 50
    t0_raw = float(np.nanpercentile(np.abs(means[good]), 90)) if \
        np.any(good) else float("nan")
    # Centre-region scale: the extreme-|u| bins carry the documented
    # crest-transient estimator bias and would inflate t0 (0.70 vs
    # 0.27); the tendency scale must reflect where the mass lives.
    centre = (
        good
        & (np.abs(u_centres)[:, None] <= 2.0)
        & (np.abs(s_centres)[None, :] <= 1.5)
    )
    t0 = float(np.nanpercentile(np.abs(means[centre]), 90)) if \
        np.any(centre) else t0_raw
    print(f"[gpr-ref] t0 (tendency scale, centre) = {t0:.5f} "
          f"(raw all-bin 90pct {t0_raw:.5f})")

    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "paths": int(eta_full.shape[0]),
        "save_cadence_s": float(times_s[1] - times_s[0]),
        "samples": total,
        "s0_slope_scale": s0,
        "t0_tendency_scale": t0,
        "t0_raw_90pct": t0_raw,
        "note": "full-grid reference bundle; exact-linear-propagator "
        "residuals with shifted true traces; surface in nondim "
        "u-tendency units on (u, u_y/s0)",
        "note2": "t0_tendency_scale uses centre-region bins (|u|<=2, "
        "|s|<=1.5, counts>50): extreme-|u| bins are inflated by the "
        "crest-transient estimator bias and excluded from the scale",
    }
    (OUT_DIR / "calibration.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    variance = np.where(
        counts > 1,
        (sums_sq - sums**2 / np.maximum(counts, 1))
        / np.maximum(counts - 1, 1),
        np.nan,
    )
    mean_sigma = np.sqrt(variance / np.maximum(counts, 1))
    np.savez_compressed(
        OUT_DIR / "surface.npz",
        u_centres=u_centres, s_centres=s_centres,
        bin_means=means, bin_counts=counts,
        bin_mean_sigma=mean_sigma, s0=s0, t0=t0,
    )
    print(f"[gpr-ref] outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
