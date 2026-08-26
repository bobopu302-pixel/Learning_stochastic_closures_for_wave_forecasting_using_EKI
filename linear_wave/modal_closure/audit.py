"""Post-hoc audit of a frozen modal closure run.

This module answers three questions that the headline metrics in
``derived_metrics.json`` cannot answer on their own.

1.  **Which objective.**  The reported ``Phi`` is computed from ``G_hat``,
    the mean of N_G = 100 forward runs, so it is smoother than a single
    realisation by sqrt(N_G) = 10 and is *not* comparable with an objective
    built from single runs.  ``out_of_sample_objective`` re-evaluates the
    reported parameter vector with fresh noise using single, unaveraged runs,
    which is the like-for-like quantity.

2.  **Calibration anchor.**  ``Gamma`` is the reference-side sampling
    covariance and nothing else, so ``Phi`` is on a chi^2 scale and q/2 is the
    level a correct model should reach.  ``truth_reference_objective`` supplies
    the empirical anchor: independent *truth* records scored against the same
    target, which is the irreducible level that no model can beat.

3.  **Velocity-channel definition.**  The second observation channel is the
    modal-skeleton functional of Eq. (5), not the pathwise derivative of the
    model elevation.  ``velocity_readout_audit`` measures the gap and checks it
    against the exact Ito prediction.

Nothing here writes to ``bundle.npz``, ``summary.json`` or
``derived_metrics.json``.  Outputs go to ``audit_metrics.json`` (machine
readable) and ``audit_arrays.npz`` (every raw sample, for later tables).

Origin: 2.Linear_wave_case/modal_closure/audit.py
Changes vs origin:
- the three figure builders (_figure_objective_calibration,
  _figure_velocity_readout, _figure_spectral_leakage), their calls, the
  report_figures/ directory handling and the matplotlib import are deleted
  (no figures in this release); audit_metrics.json and audit_arrays.npz are
  saved exactly as before;
- the plot-style import from the old diagnostics module (colour constants and
  _style) is deleted with the figures;
- the diagnostic objective() is DE-regularised (release decision 2026-08-25,
  uniform across the release): Phi = 0.5 r^T Gamma^-1 r with the EXACT Gamma,
  no jitter anywhere in the objective.  The origin inflated the diagonal by a
  relative 1e-8; the change is <= 1e-8 relative, far below the audit's
  tolerances and every reported precision.  Regularisation now lives only in
  the engine's Kalman-gain solve (algorithms.eki);
- comments/docstrings polished; all other retained numerics untouched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from . import experiment as case


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "modal_closure"
DEFAULT_BUNDLE = DEFAULT_RESULTS_DIR / "bundle.npz"

# Audit-only seeds.  Disjoint from every seed recorded in summary.json so that
# no out-of-sample evaluation can accidentally reuse a training realisation.
OOS_SEED = 90_000
REF_SEED = 300_000
VELOCITY_SEED = 777_001
N_OUT_OF_SAMPLE = 200
N_TRUTH_REFERENCE = 200
VELOCITY_STEPS = (0.1, 0.05, 0.025, 0.0125, 0.00625)
VELOCITY_RECORD_S = 120.0


# The statistic order is fixed by the experiment module, so the names live
# there too; re-exported here because the audit's own callers import it.
statistic_labels = case.statistic_labels


def statistic_blocks() -> list[str]:
    """Coarse grouping of the 38 statistics, used for reporting."""

    return (
        ["variance"] * 2
        + ["autocorrelation"] * len(case.AUTO_LAGS)
        + ["cross-correlation"] * (2 * len(case.CROSS_LAGS))
        + ["band energy"] * case.M_MODES
    )


def objective(prediction, observation, gamma) -> float:
    """Diagnostic objective ``0.5 * r^T Gamma^-1 r`` with the EXACT Gamma.

    No regularisation enters the objective anywhere in this release (uniform
    convention): Phi is the reported physical discrepancy measure, so it may
    not carry a numerical conditioning term.  Regularisation exists only in
    the engine's Kalman-gain solve (``algorithms.eki``).

    ``gamma`` may be the diagonal as a vector or the full covariance as a
    matrix.  Both shapes occur: the spec's Gamma is ``diag(var_ref)``, but the
    full reference covariance is available as a switch, and quietly taking only
    its diagonal would report a number that is not the objective EKI minimised.
    """

    residual = np.asarray(prediction, dtype=float) - np.asarray(observation, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    if gamma.ndim == 1:
        return 0.5 * float(np.sum(residual**2 / gamma))
    return 0.5 * float(residual @ np.linalg.solve(gamma, residual))


def rebuild_error_model() -> dict[str, np.ndarray]:
    """Rebuild y and Gamma exactly as the calibration built them.

    This delegates to ``experiment.build_observation_and_gamma`` rather than
    reimplementing the formula: an audit that re-derives the error model
    independently silently stops describing the run as soon as the two copies
    drift apart.  The truth generator is analytic and seeded, so the result
    reproduces the archived y and Gamma exactly -- which ``run_audit`` asserts.
    """

    y, gamma_used, parts = case.build_observation_and_gamma()
    return {
        "y": y,
        "gamma_used": gamma_used,
        "gamma_sampling": parts["gamma_sampling"],
        "var_ref": parts["var_ref"],
        "var_fwd": parts["var_fwd"],
        "floor": parts["floor"],
        "floor_binds": parts["floor_dominates"],
        "per_record_stats": parts["reference_stats"],
        "forward_stats": parts["forward_stats"],
        "neff_correction": bool(parts.get("neff_correction", False)),
        "neff_factors_by_family": dict(parts.get("neff_factors_by_family", {})),
    }


def out_of_sample_objective(
    theta,
    y,
    gamma_used,
    gamma_sampling,
    *,
    n_repeats: int = N_OUT_OF_SAMPLE,
    seed: int = OOS_SEED,
    n_workers: int = case.N_WORKERS,
) -> dict[str, np.ndarray]:
    """Re-evaluate one parameter vector with fresh noise, n_repeats times.

    Each evaluation uses the *same* forward map as EKI -- one simulation over
    the analysis window -- so it is directly comparable with the stored
    objective.
    """

    q = np.asarray(y).size
    map_fn = case._mapper(n_workers)
    outputs = np.asarray(map_fn(
        case._forward_job,
        [(theta, (seed, 0, i), q, 1, case.T_RECORD)
         for i in range(n_repeats)]))
    return {
        "outputs": outputs,
        "phi_used": np.array([objective(g, y, gamma_used) for g in outputs]),
        "phi_sampling": np.array([objective(g, y, gamma_sampling) for g in outputs]),
    }


def truth_reference_objective(
    y,
    gamma_used,
    gamma_sampling,
    *,
    n_repeats: int = N_TRUTH_REFERENCE,
    seed: int = REF_SEED,
    n_workers: int = case.N_WORKERS,
) -> dict[str, np.ndarray]:
    """Score independent TRUTH records against the same target.

    This is the irreducible objective: it contains no model error at all, only
    the Monte Carlo noise of a single-record statistic, exactly as the forward
    map has.  Any model must be judged against this level, not against zero.
    """

    map_fn = case._mapper(n_workers)
    outputs = np.asarray(map_fn(
        case._reference_job, [(seed + i, case.N_DATA) for i in range(n_repeats)]))
    return {
        "outputs": outputs,
        "phi_used": np.array([objective(g, y, gamma_used) for g in outputs]),
        "phi_sampling": np.array([objective(g, y, gamma_sampling) for g in outputs]),
    }


def spectral_like_for_like(model_outputs, reference_outputs, theta) -> dict[str, Any]:
    """Compare the spectrum the way the objective actually constrains it.

    The last M entries of the statistic vector are periodogram band energies, so
    averaging them over the out-of-sample model draws and over the independent
    reference records gives the same estimator applied to both sides.  The
    analytic modal energy Var(q_j) is reported alongside to expose the leakage:
    once a mode's Lorentzian is wide enough to spill across band edges, EKI
    distorts Var(q_j) so that the periodogram matches, and comparing Var(q_j)
    with the binned truth then reports that distortion as spectral error.
    """

    model_bands = np.asarray(model_outputs)[:, -case.M_MODES:].mean(axis=0)
    truth_bands = np.asarray(reference_outputs)[:, -case.M_MODES:].mean(axis=0)
    modal_energy = case.recovered_spectrum(*case.unpack(theta))
    grid = case.F_GRID
    truth_total = float(truth_bands.sum())
    model_total = float(model_bands.sum())
    delta, _ = case.unpack(theta)
    # Lorentzian half-width in Hz against the band width; > ~0.5 means the tails
    # cross band edges and the two spectral comparisons must diverge.
    half_width_hz = delta / (4.0 * np.pi)
    return {
        "n_model_draws": int(np.asarray(model_outputs).shape[0]),
        "n_reference_records": int(np.asarray(reference_outputs).shape[0]),
        "truth_total_elevation_energy_m2": truth_total,
        "model_total_elevation_energy_m2": model_total,
        "total_energy_relative_error": model_total / truth_total - 1.0,
        "l1_error_fraction": float(np.abs(model_bands - truth_bands).sum() / truth_total),
        "normalized_rmse": float(
            np.sqrt(np.mean((model_bands - truth_bands) ** 2))
            / np.sqrt(np.mean(truth_bands**2))
        ),
        "truth_centroid_hz": float(np.sum(grid * truth_bands) / truth_total),
        "model_centroid_hz": float(np.sum(grid * model_bands) / model_total),
        "truth_bands_m2": truth_bands.tolist(),
        "model_bands_m2": model_bands.tolist(),
        "modal_energy_m2": modal_energy.tolist(),
        "leakage_modal_minus_periodogram_m2": (modal_energy - model_bands).tolist(),
        "max_abs_leakage_m2": float(np.max(np.abs(modal_energy - model_bands))),
        "lorentzian_half_width_hz": half_width_hz.tolist(),
        "half_width_over_band_width": (half_width_hz / case.SPEC_DF).tolist(),
        "note": (
            "Quote l1_error_fraction from this block as spectral accuracy. The "
            "modal-energy-versus-binned-truth figure in derived_metrics.json "
            "measures a different pair and is inflated by leakage."
        ),
    }


def _summarise(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)),
        "min": float(values.min()),
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "q95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
        "n": int(values.size),
    }


def velocity_readout_audit(
    theta,
    *,
    steps=VELOCITY_STEPS,
    t_record: float = VELOCITY_RECORD_S,
    seed: int = VELOCITY_SEED,
) -> dict[str, Any]:
    """Compare each velocity channel with a central difference of its elevation.

    For the truth the two agree to O(dt^2): v is the analytic time derivative.
    For the model they cannot agree.  Ito's formula applied to
    eta_i = sum_j [q_j cos(k_j x_i) - (p_j/omega_j) sin(k_j x_i)] gives

        d eta_i = [v_i + sum_j (delta_j/omega_j) p_j sin(k_j x_i)] dt
                  - sum_j (sqrt(sigma_j)/omega_j) sin(k_j x_i) dW_j,

    so eta has a martingale part and is not differentiable.  A central
    difference over 2*dt therefore has excess variance S_i/(2 dt) with
    S_i = sum_j (sigma_j/omega_j^2) sin^2(k_j x_i), i.e. the mismatch grows like
    dt^{-1/2} as the step is refined.  This function checks that prediction.
    """

    delta, sqrt_sigma = case.unpack(theta)
    sigma = np.asarray(sqrt_sigma, dtype=float) ** 2
    kx = np.outer(case.GAUGES, case.K_GRID)
    excess = (np.sin(kx) ** 2) @ (sigma / case.OMEGA2_GRID)  # S_i, one per gauge

    def mismatch(eta, v, dt):
        derivative = (eta[2:] - eta[:-2]) / (2.0 * dt)
        return float(np.sqrt(np.mean((derivative - v[1:-1]) ** 2)) / v.std())

    rows = []
    for dt in steps:
        n = int(round(t_record / dt))
        times = np.arange(n) * dt

        eta_m, v_m = case.simulate_grid(
            delta, sqrt_sigma, rng=np.random.default_rng(seed), t_record=t_record, t_burn=30.0, dt=dt
        )
        eta_list, v_list = case.truth_fields_at(times, seed=seed + 1)

        predicted = float(np.sqrt(np.mean(excess) / (2.0 * dt)) / v_m.std())
        rows.append(
            {
                "dt_s": float(dt),
                "model_relative_rmse": mismatch(eta_m, v_m, dt),
                "model_ito_prediction": predicted,
                "truth_relative_rmse": mismatch(eta_list, v_list, dt),
            }
        )

    model = np.array([r["model_relative_rmse"] for r in rows])
    truth = np.array([r["truth_relative_rmse"] for r in rows])
    log_dt = np.log(np.array([r["dt_s"] for r in rows]))
    return {
        "rows": rows,
        "model_slope_loglog": float(np.polyfit(log_dt, np.log(model), 1)[0]),
        "truth_slope_loglog": float(np.polyfit(log_dt, np.log(truth), 1)[0]),
        "analysis_dt_s": float(case.DT_DATA),
        "model_relative_rmse_at_analysis_dt": float(
            model[[abs(r["dt_s"] - case.DT_DATA) < 1e-12 for r in rows]][0]
        ),
        "excess_variance_per_gauge": excess,
    }


def run_audit(
    bundle_path: str | Path = DEFAULT_BUNDLE,
    output_dir: str | Path | None = None,
    *,
    n_out_of_sample: int = N_OUT_OF_SAMPLE,
    n_truth_reference: int = N_TRUTH_REFERENCE,
    n_workers: int = case.N_WORKERS,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run every audit and save the raw samples and metrics."""

    bundle_path = Path(bundle_path)
    results_dir = Path(output_dir) if output_dir is not None else bundle_path.parent
    results_dir.mkdir(parents=True, exist_ok=True)

    with np.load(bundle_path, allow_pickle=False) as bundle:
        y_stored = np.array(bundle["target_statistics"])
        gamma_stored = np.array(bundle["gamma_diag"])
        # Bundle naming trap: `sde_best` holds the FINAL-ENSEMBLE MEAN in
        # physical units (spec reporting), not a selected best member.
        theta = np.array(bundle["sde_best"])
        stored_objective = float(bundle["sde_objective"])
        members = np.array(bundle["sde_members"])

    if verbose:
        print("[audit] rebuilding the error model from the truth generator ...",
              flush=True)
    error_model = rebuild_error_model()
    # Not bit equality: the archived run was computed on the Linux server and
    # this rebuild runs wherever the audit is invoked, and different libm/BLAS
    # builds move the last one or two bits (measured Windows vs Ubuntu: 1.1e-15
    # relative).  1e-11 is four orders tighter than any physical difference and
    # still catches a genuinely different y or Gamma at once.
    rebuild_rtol = 1e-11
    reproduces_y = bool(np.allclose(error_model["y"], y_stored,
                                    rtol=rebuild_rtol, atol=0.0))
    reproduces_gamma = bool(np.allclose(error_model["gamma_used"], gamma_stored,
                                        rtol=rebuild_rtol, atol=0.0))
    if not (reproduces_y and reproduces_gamma):
        raise RuntimeError(
            "the rebuilt error model does not match the archived bundle to "
            f"rtol={rebuild_rtol:g}; the audit would not describe the frozen run"
        )

    if verbose:
        print(f"[audit] re-evaluating the reported theta {n_out_of_sample} times ...", flush=True)
    model_phi = out_of_sample_objective(
        theta, y_stored, gamma_stored, error_model["gamma_sampling"],
        n_repeats=n_out_of_sample, n_workers=n_workers
    )
    if verbose:
        print(f"[audit] scoring {n_truth_reference} independent truth records ...", flush=True)
    truth_phi = truth_reference_objective(
        y_stored, gamma_stored, error_model["gamma_sampling"],
        n_repeats=n_truth_reference, n_workers=n_workers
    )
    if verbose:
        print("[audit] velocity readout consistency ...", flush=True)
    velocity = velocity_readout_audit(theta)
    spectral = spectral_like_for_like(model_phi["outputs"], truth_phi["outputs"], theta)

    # The inversion runs in log coordinates, so the prior interval is a
    # sampling range for the initial ensemble and not a constraint: a member is
    # free to leave it and nothing is clipped.  What is worth recording is how
    # far the final ensemble ranged outside it, since a large excursion would
    # mean the prior was placed badly.  (The yardstick is the legacy parameter
    # box DELTA_BOUNDS/SIGMA_BOUNDS_SDE, exactly as in the frozen run.)
    lower = np.array([b[0] for b in case.DELTA_BOUNDS + case.SIGMA_BOUNDS_SDE])
    upper = np.array([b[1] for b in case.DELTA_BOUNDS + case.SIGMA_BOUNDS_SDE])
    below = members < lower[None, :]
    above = members > upper[None, :]

    used = _summarise(model_phi["phi_used"])
    reference_used = _summarise(truth_phi["phi_used"])
    sampling = _summarise(model_phi["phi_sampling"])
    reference_sampling = _summarise(truth_phi["phi_sampling"])

    metrics: dict[str, Any] = {
        "purpose": (
            "Post-hoc audit of the frozen run. Does not modify bundle.npz, "
            "summary.json or derived_metrics.json."
        ),
        "error_model_reproduced_from_truth": {
            "y_matches_bundle": reproduces_y,
            "gamma_matches_bundle": reproduces_gamma,
        },
        "reported_objective": {
            "value": stored_objective,
            "convention": "final ensemble mean of G_hat, averaged over N_G runs",
            "chi2_expectation_q_over_2": float(y_stored.size / 2.0),
            "ratio_to_chi2_expectation": float(stored_objective / (y_stored.size / 2.0)),
        },
        "out_of_sample": {
            "description": (
                "The reported theta re-evaluated with fresh noise, using SINGLE "
                "unaveraged forward runs."
            ),
            "distribution": used,
            "ratio_median_over_reported": float(used["median"] / stored_objective),
            "note": (
                "This is not evidence of selection bias: no member is selected, the "
                "reported value is the ensemble mean. The gap is mostly construction "
                "-- G_hat averages N_G = 100 runs and is smoother than a single run by "
                "sqrt(N_G) = 10. Compare this median with irreducible_reference, which "
                "is built the same way, not with the reported value."
            ),
        },
        "irreducible_reference": {
            "description": (
                "Independent truth records scored against the same target over the same "
                "single analysis window. Contains no model error."
            ),
            "under_reported_gamma": reference_used,
            "under_sampling_gamma": reference_sampling,
            "model_excess_over_reference_used_gamma": float(
                used["median"] / reference_used["median"]
            ),
            "model_excess_over_reference_sampling_gamma": float(
                sampling["median"] / reference_sampling["median"]
            ),
        },
        "error_model_scale": {
            "gamma": "diag(var_ref); no forward term, no discrepancy floor, no n_eff",
            "chi2_expectation_q_over_2": float(y_stored.size / 2.0),
            "max_abs_relative_gap_used_vs_sampling": float(
                np.max(np.abs(np.asarray(error_model["gamma_used"], dtype=float)
                              / np.asarray(error_model["gamma_sampling"], dtype=float) - 1.0))
            ),
            "note": (
                "Gamma IS the sampling covariance under this spec, so Phi is on a "
                "sampling-noise chi^2 scale and its absolute value is meaningful. The "
                "gap above is identically zero by construction and is recorded as a "
                "guard: a nonzero value would mean a floor had returned."
            ),
        },
        "spectral_like_for_like": spectral,
        "velocity_readout": {
            "rows": velocity["rows"],
            "model_slope_loglog": velocity["model_slope_loglog"],
            "truth_slope_loglog": velocity["truth_slope_loglog"],
            "analysis_dt_s": velocity["analysis_dt_s"],
            "model_relative_rmse_at_analysis_dt": velocity["model_relative_rmse_at_analysis_dt"],
            "note": (
                "Truth converges at order dt^2 (v is the analytic derivative). The model "
                "diverges like dt^-0.5 because eta contains the white-noise-driven p and is "
                "not differentiable; v is the modal-skeleton readout of Eq. (5). Stationary "
                "second moments, which are what the report compares, are unaffected."
            ),
        },
        "prior_box_excursions": {
            "final_ensemble_components_below_prior": int(below.sum()),
            "final_ensemble_components_above_prior": int(above.sum()),
            "final_ensemble_components_total": int(members.size),
            "below_prior_delta_block": int(below[:, : case.M_MODES].sum()),
            "below_prior_sqrt_sigma_block": int(below[:, case.M_MODES :].sum()),
            "note": (
                "The inversion runs in log coordinates, so the prior interval samples "
                "the initial ensemble and does not constrain the update: nothing is "
                "clipped and no component can sit on a bound. These counts say how far "
                "the final ensemble ranged outside the prior box, which is a check on "
                "prior placement, not on the fit."
            ),
        },
        "seeds": {
            "out_of_sample": OOS_SEED,
            "truth_reference": REF_SEED,
            "velocity": VELOCITY_SEED,
        },
    }

    metrics_path = results_dir / "audit_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    # Every raw sample, so tables and figures can be rebuilt without recomputing.
    arrays_path = results_dir / "audit_arrays.npz"
    np.savez(
        arrays_path,
        statistic_labels=np.array(statistic_labels()),
        statistic_blocks=np.array(statistic_blocks()),
        target_statistics=y_stored,
        gamma_used=error_model["gamma_used"],
        gamma_sampling=error_model["gamma_sampling"],
        gamma_var_ref=error_model["var_ref"],
        truth_per_record_statistics=error_model["per_record_stats"],
        selected_theta=theta,
        stored_objective=np.array(stored_objective),
        oos_outputs=model_phi["outputs"],
        oos_phi_used=model_phi["phi_used"],
        oos_phi_sampling=model_phi["phi_sampling"],
        reference_outputs=truth_phi["outputs"],
        reference_phi_used=truth_phi["phi_used"],
        reference_phi_sampling=truth_phi["phi_sampling"],
        velocity_dt=np.array([r["dt_s"] for r in velocity["rows"]]),
        velocity_model_rmse=np.array([r["model_relative_rmse"] for r in velocity["rows"]]),
        velocity_model_prediction=np.array([r["model_ito_prediction"] for r in velocity["rows"]]),
        velocity_truth_rmse=np.array([r["truth_relative_rmse"] for r in velocity["rows"]]),
        velocity_excess_variance_per_gauge=velocity["excess_variance_per_gauge"],
        spectral_truth_bands=np.array(spectral["truth_bands_m2"]),
        spectral_model_bands=np.array(spectral["model_bands_m2"]),
        spectral_modal_energy=np.array(spectral["modal_energy_m2"]),
        ensemble_below_prior=below,
        ensemble_above_prior=above,
    )

    if verbose:
        print(f"[audit] reported objective        {stored_objective:.4f}")
        print(f"[audit] out-of-sample objective   {used['median']:.3f} "
              f"(90% interval {used['q05']:.3f}-{used['q95']:.3f})")
        print(f"[audit] irreducible truth level   {reference_used['median']:.3f} "
              f"(90% interval {reference_used['q05']:.3f}-{reference_used['q95']:.3f})")
        print(f"[audit] spectrum, like for like    L1 = "
              f"{spectral['l1_error_fraction']:.2%}  "
              f"(modal-energy metric reports "
              f"{np.abs(np.array(spectral['modal_energy_m2']) - np.array(spectral['truth_bands_m2'])).sum() / np.array(spectral['truth_bands_m2']).sum():.2%})")
        print(f"[audit] Phi / (q/2)                {stored_objective / (y_stored.size / 2.0):.2f}"
              f"   (single-run model {used['median']:.1f} vs truth "
              f"{reference_used['median']:.1f}, ratio "
              f"{used['median'] / reference_used['median']:.1f}x)")
        print(f"[audit] wrote {metrics_path}")
        print(f"[audit] wrote {arrays_path}")

    return metrics
