"""Bundle validation and derived validation metrics for a completed modal
closure run (no figures).

Origin: 2.Linear_wave_case/modal_closure/diagnostics.py
Changes vs origin:
- renamed from diagnostics.py; only the computation half is kept (load_bundle,
  validate_bundle, band_energy_spectrum_like, compute_metrics, write_metrics
  and their private helpers).  The matplotlib import, the plot style/colour
  constants, the histogram-smoothing helpers and every figure function
  (make_report_figures and the eleven _*_figure builders) are deleted --
  this release ships computation and data only;
- the statistic estimators (band_energy_spectrum, cross_corr, gauge_acf) are
  imported from the shared algorithms.statistics instead of the local
  numerics module;
- the derived_metrics.json key "objective.selected_member" is renamed to
  "objective.reported_final_ensemble_mean": the stored value is the objective
  of the FINAL-ENSEMBLE-MEAN output (2026-08-23 spec reporting, no member
  selection), and the old name was a leftover misnomer.  No shipped code read
  the old key, so the rename is safe; a JSON written by the original tree
  differs only in this key name.

Note the related bundle-array naming trap: the bundle key ``sde_best`` also
holds the final-ensemble mean, not a selected member.  Bundle keys are NOT
renamed, so the archived frozen bundle still validates; see the README.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import ks_2samp, kurtosis, skew, wasserstein_distance

from algorithms.statistics import band_energy_spectrum, cross_corr, gauge_acf


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "modal_closure"
DEFAULT_BUNDLE = DEFAULT_RESULTS_DIR / "bundle.npz"

AUTO_LAGS = np.array([1, 2, 3, 4, 6, 8, 12, 16], dtype=int)
CROSS_LAGS = np.array([-8, -6, -4, -2, 0, 2, 4, 6, 8], dtype=int)


def load_bundle(path: str | Path = DEFAULT_BUNDLE) -> dict[str, np.ndarray]:
    bundle_path = Path(path)
    with np.load(bundle_path, allow_pickle=False) as bundle:
        return {key: np.array(bundle[key]) for key in bundle.files}


def validate_bundle(data: dict[str, np.ndarray]) -> dict[str, Any]:
    required = {
        "experiment",
        "gauges",
        "dt_data",
        "truth_eta_long",
        "truth_v_long",
        "sde_eta_long",
        "sde_v_long",
        "grid_freq",
        "truth_freq",
        "truth_energy",
        "truth_binned",
        "sde_best_spectrum",
        "sde_member_spectrum",
        "sde_best",
        "sde_objective",
        "sde_objective_history",
    }
    missing = sorted(required.difference(data))
    if missing:
        raise ValueError(f"Bundle is missing required arrays: {missing}")

    checks: dict[str, Any] = {}
    checks["experiment"] = str(data["experiment"])
    checks["n_gauges"] = int(data["gauges"].size)
    checks["n_truth_components"] = int(data["truth_freq"].size)
    checks["n_model_modes"] = int(data["grid_freq"].size)
    checks["long_record_samples"] = int(data["truth_eta_long"].shape[0])
    checks["long_record_seconds"] = float(
        data["truth_eta_long"].shape[0] * float(data["dt_data"])
    )

    expected_shape = data["truth_eta_long"].shape
    for key in (
        "truth_v_long",
        "sde_eta_long",
        "sde_v_long",
    ):
        if data[key].shape != expected_shape:
            raise ValueError(f"{key} has shape {data[key].shape}, expected {expected_shape}")
        if not np.all(np.isfinite(data[key])):
            raise ValueError(f"{key} contains non-finite values")

    if data["sde_best"].size != 2 * data["grid_freq"].size:
        raise ValueError("The reported SDE parameter vector does not contain delta and sqrt_sigma")
    if not np.isclose(data["truth_energy"].sum(), data["target_var"], rtol=1e-12):
        raise ValueError("Truth component energy does not satisfy the target variance")
    if not np.isclose(data["truth_binned"].sum(), data["truth_energy"].sum(), rtol=1e-12):
        raise ValueError("Binned truth energy does not conserve total energy")

    modes = data["grid_freq"].size
    delta = data["sde_best"][:modes]
    sqrt_sigma = data["sde_best"][modes:]
    omega = 2.0 * np.pi * data["grid_freq"]
    derived = sqrt_sigma**2 / (2.0 * delta * omega**2)
    spectrum_error = float(np.max(np.abs(derived - data["sde_best_spectrum"])))
    if spectrum_error > 1e-12:
        raise ValueError(f"Stored SDE spectrum is inconsistent with parameters: {spectrum_error}")
    checks["maximum_spectrum_identity_error"] = spectrum_error
    checks["status"] = "passed"
    return checks


def band_energy_spectrum_like(data: dict[str, np.ndarray], key: str) -> np.ndarray:
    """Periodogram band energy of one stored long record, on the model grid.

    The same estimator is applied to whichever field is named, so truth and
    model numbers from this function are directly comparable.  Exposed for
    inspection; the reportable like-for-like spectral comparison is built in
    ``audit.spectral_like_for_like``, which averages many full-window records
    instead of using one 300 s validation record.
    """

    grid = data["grid_freq"]
    spec_df = float(grid[1] - grid[0])
    return band_energy_spectrum(data[key], grid, spec_df, float(data["dt_data"]))


def _distribution_summary(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=float).ravel()
    return {
        "mean": float(flat.mean()),
        "standard_deviation": float(flat.std()),
        "variance": float(flat.var()),
        "skewness": float(skew(flat)),
        "excess_kurtosis": float(kurtosis(flat)),
    }


def _field_metrics(
    truth: np.ndarray, model: np.ndarray
) -> dict[str, float]:
    truth_flat = truth.ravel()
    model_flat = model.ravel()
    return {
        "variance_relative_error": float(model_flat.var() / truth_flat.var() - 1.0),
        "normalized_wasserstein_distance": float(
            wasserstein_distance(truth_flat, model_flat) / truth_flat.std()
        ),
        "kolmogorov_smirnov_distance": float(
            ks_2samp(truth_flat, model_flat).statistic
        ),
    }


def _binned_truth_velocity_energy(data: dict[str, np.ndarray]) -> np.ndarray:
    grid = data["grid_freq"]
    truth_freq = data["truth_freq"]
    truth_velocity_energy = data["truth_energy"] * (2.0 * np.pi * truth_freq) ** 2
    indices = np.argmin(np.abs(truth_freq[:, None] - grid[None, :]), axis=1)
    binned = np.zeros_like(grid, dtype=float)
    for component, index in enumerate(indices):
        binned[index] += truth_velocity_energy[component]
    return binned


def compute_metrics(data: dict[str, np.ndarray]) -> dict[str, Any]:
    checks = validate_bundle(data)
    dt = float(data["dt_data"])
    truth_eta = data["truth_eta_long"]
    truth_v = data["truth_v_long"]

    metrics: dict[str, Any] = {
        "validation": checks,
        "truth_distribution": {
            "eta": _distribution_summary(truth_eta),
            "v": _distribution_summary(truth_v),
        },
        "models": {},
    }

    truth_acf = gauge_acf(truth_eta, AUTO_LAGS)
    truth_c1 = cross_corr(truth_eta, 1, CROSS_LAGS)
    truth_c2 = cross_corr(truth_eta, 2, CROSS_LAGS)
    for model in ("sde",):
        eta = data[f"{model}_eta_long"]
        velocity = data[f"{model}_v_long"]
        model_acf = gauge_acf(eta, AUTO_LAGS)
        model_c1 = cross_corr(eta, 1, CROSS_LAGS)
        model_c2 = cross_corr(eta, 2, CROSS_LAGS)
        ensemble_eta_var = data[f"{model}_eta_ens"].var(axis=(1, 2))
        ensemble_v_var = data[f"{model}_v_ens"].var(axis=(1, 2))
        metrics["models"][model] = {
            "eta_distribution": _distribution_summary(eta),
            "v_distribution": _distribution_summary(velocity),
            "eta_fit": _field_metrics(truth_eta, eta),
            "v_fit": _field_metrics(truth_v, velocity),
            "acf_rmse": float(np.sqrt(np.mean((model_acf - truth_acf) ** 2))),
            "adjacent_cross_correlation_rmse": float(
                np.sqrt(np.mean((model_c1 - truth_c1) ** 2))
            ),
            "next_nearest_cross_correlation_rmse": float(
                np.sqrt(np.mean((model_c2 - truth_c2) ** 2))
            ),
            "adjacent_peak_lag_seconds": float(CROSS_LAGS[np.argmax(model_c1)] * dt),
            "next_nearest_peak_lag_seconds": float(CROSS_LAGS[np.argmax(model_c2)] * dt),
            "ensemble_eta_variance_q05_q50_q95": np.quantile(
                ensemble_eta_var, [0.05, 0.5, 0.95]
            ).tolist(),
            "ensemble_v_variance_q05_q50_q95": np.quantile(
                ensemble_v_var, [0.05, 0.5, 0.95]
            ).tolist(),
        }

    grid = data["grid_freq"]
    truth_spectrum = data["truth_binned"]
    sde_spectrum = data["sde_best_spectrum"]
    truth_total = float(truth_spectrum.sum())
    sde_total = float(sde_spectrum.sum())
    metrics["correlation_reference"] = {
        "adjacent_truth_peak_lag_seconds": float(CROSS_LAGS[np.argmax(truth_c1)] * dt),
        "next_nearest_truth_peak_lag_seconds": float(CROSS_LAGS[np.argmax(truth_c2)] * dt),
    }
    # Two different spectral comparisons, kept explicitly apart.
    #
    # "modal_energy_vs_binned_truth" compares the analytic modal energy
    # Var(q_j) = sigma_j/(2 delta_j omega_j^2) with the analytically binned
    # truth.  It is NOT what the calibration targets, and the two quantities
    # coincide only while each mode's Lorentzian (half-width delta_j/4pi) sits
    # well inside its band.  Once delta is large enough to leak across bands,
    # EKI distorts Var(q_j) so that the *periodogram* matches, and this
    # comparison reports that distortion as error.  Identified modal energies
    # are a deconvolution of the truth spectrum, not the truth spectrum.
    #
    # The like-for-like periodogram comparison lives in audit.py, which already
    # has 200 model draws and 200 reference records over the full analysis
    # window.  It is not computed here because a single 300 s record gives each
    # band only ~14% precision, so the comparison would be noise-dominated.
    metrics["spectrum"] = {
        "comparison": "modal_energy_vs_binned_truth",
        "caveat": (
            "Var(q_j) versus the binned truth. This is NOT the quantity the "
            "objective targets, and band leakage inflates it: quote "
            "spectral_like_for_like.l1_error_fraction from audit_metrics.json "
            "as spectral accuracy instead."
        ),
        "truth_total_elevation_energy_m2": truth_total,
        "sde_total_elevation_energy_m2": sde_total,
        "total_energy_relative_error": sde_total / truth_total - 1.0,
        "l1_error_fraction": float(np.abs(sde_spectrum - truth_spectrum).sum() / truth_total),
        "normalized_rmse": float(
            np.sqrt(np.mean((sde_spectrum - truth_spectrum) ** 2))
            / np.sqrt(np.mean(truth_spectrum**2))
        ),
        "truth_centroid_hz": float(np.sum(grid * truth_spectrum) / truth_total),
        "sde_centroid_hz": float(np.sum(grid * sde_spectrum) / sde_total),
    }
    modes = grid.size
    delta = data["sde_best"][:modes]
    sqrt_sigma = data["sde_best"][modes:]
    omega = 2.0 * np.pi * grid
    mechanical = sqrt_sigma**2 / (2.0 * delta)
    metrics["parameters"] = [
        {
            "mode": int(index + 1),
            "frequency_hz": float(grid[index]),
            "omega_rad_per_s": float(omega[index]),
            "delta_per_s": float(delta[index]),
            "damping_time_s": float(1.0 / delta[index]),
            "sqrt_sigma": float(sqrt_sigma[index]),
            "elevation_energy_m2": float(sde_spectrum[index]),
            "velocity_energy_m2_per_s2": float(mechanical[index]),
        }
        for index in range(modes)
    ]
    history = np.asarray(data["sde_objective_history"], dtype=float)
    metrics["objective"] = {
        # The objective of the FINAL-ENSEMBLE-MEAN output (spec 2026-08-23
        # reporting: no member selection).  This key was called
        # "selected_member" in the source tree -- a misnomer; it equals
        # final_ensemble_mean by construction and both are kept for clarity.
        "reported_final_ensemble_mean": float(data["sde_objective"]),
        "initial_ensemble_mean": float(history[0]),
        "final_ensemble_mean": float(history[-1]),
        "ensemble_mean_history": history.tolist(),
    }
    metrics["truth_velocity_energy_binned"] = _binned_truth_velocity_energy(data).tolist()
    return metrics


def write_metrics(
    bundle_path: str | Path = DEFAULT_BUNDLE,
    output_path: str | Path | None = None,
) -> Path:
    bundle_path = Path(bundle_path).resolve()
    destination = (
        Path(output_path).resolve()
        if output_path is not None
        else bundle_path.parent / "derived_metrics.json"
    )
    metrics = compute_metrics(load_bundle(bundle_path))
    destination.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return destination
