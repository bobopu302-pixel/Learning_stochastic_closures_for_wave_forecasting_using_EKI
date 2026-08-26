"""Physical validation of the S1 posterior: fresh-seed paths vs the truth.

Origin: 3. KDV_nonlinear_case/v3_s1_validation.py
Changes vs origin:
- deleted the three matplotlib figure blocks (invariant measure, spectra,
  profile panels) -- this release ships no plotting.  The compute behind
  them is kept, and the derived statistics that only existed inside the
  figure code are now SAVED instead of drawn:
    analysis/validation_fields.npz    (unchanged, as origin)
    analysis/validation_profiles.npz  (NEW: Hs / devrms / skew / kurt of
                                      each model path, the observed y
                                      values and the reference-record
                                      spreads used as error bars, plus
                                      the per-gauge PSDs)
- dropped the MPLBACKEND setdefault and the scipy gaussian_filter1d import
  (both only served the figures);
- comments/docstrings translated and polished.

Runs `--paths` SINGLE realisations at the reported parameters (final
ensemble mean) with seeds disjoint from y, Gamma and the EKI, keeps the
gauge time series, and compares against the truth record.  Where the
quantity is one of the 111 observed statistics, the natural spread of the
N_Gamma reference records is stored as the error bar -- that is the
tolerance the inversion was asked to meet.

Usage (spec chain)::

    SW_FINE=1 SW_VERSION_DIR=v3spec SW_DURATION_S=6600 \
    SW_SYNTH_PERIOD_S=6600 SW_FORWARD_PATHS=1 \
    python v3_s1_validation.py --branch diag --paths 8 --processes 8
"""
from __future__ import annotations

import os
from pathlib import Path as _Path

_HERE = _Path(__file__).resolve().parent
if os.environ.get("SW_FINE", "") == "1":
    os.environ["SDE_COARSE_N4"] = "3073"
    os.environ["SDE_COARSE_DT"] = "0.002"
    os.environ["SDE_OUTPUT_STRIDE"] = "70"
    os.environ["SDE_CLOSURE_V3"] = "0"
os.environ.setdefault("SW_S1_VARIANT", "S1a")
os.environ.setdefault("SW_S1_SUFFIX", "_fine")

import argparse
import json
from multiprocessing import Pool

import numpy as np

import sw_eki_s1 as s1                                    # noqa: E402
import v3_world                                           # noqa: E402
import sde_closure_eki as base                            # noqa: E402
import sw_gamma_unified as gu                             # noqa: E402
from sde_closure_eki_dense import _init_worker_dense      # noqa: E402

ROOT = _HERE / "results" / "stepwise" / v3_world.version_dir("v2_fine")
TRUTH_DIR = ROOT / "truth_S1a_fine"
VAL_SEED = 33000              # disjoint from y / Gamma / calibration / EKI


def run_path(task):
    """One realisation at theta; returns the gauge and dense fields."""
    theta, seed = task
    import sde_closure_config as closure_config
    from sde_closure_core import (
        GridWhiteNoise, GridWhiteNoiseParameters,
        StochasticImplicitMidpointDABCSolver,
    )
    phi, q = float(theta[0]), float(theta[1])
    parameters = base._CTX["parameters"]
    weight = base.build_envelope_weight({"phi": phi, "q": q, "lambda": 0.0},
                                        "terrain")
    rng = np.random.default_rng([VAL_SEED, int(seed)])
    noise = GridWhiteNoise(
        GridWhiteNoiseParameters(phi_amplitude=phi,
                                 correlation_length_cells=s1.CORR_CELLS),
        base._CTX["y"], parameters.lambda_ref_m,
        base._CTX["surface_to_green"], base.COARSE_DT, rng,
        spatial_weight=weight,
    )
    with closure_config.template():
        solver = StochasticImplicitMidpointDABCSolver(
            base._CTX["y"], base._CTX["depth_ratio"], parameters.epsilon,
            parameters.mu, base.COARSE_DT, base._CTX["n_steps"],
        )
    times, surface, _, _ = solver.run_stochastic(
        np.zeros_like(base._CTX["y"]), base.OUTPUT_STRIDE,
        base._CTX["traces"], noise_increment=noise,
    )
    times_s = np.asarray(times) * parameters.time_ref_s
    eta = (np.asarray(surface[:, : base.COARSE_N4], dtype=float)
           * parameters.a_ref_m)
    return (times_s, eta[:, base._CTX["gauge_columns"]].astype(np.float32),
            eta[:, base._CTX["dense_columns"]].astype(np.float32))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--branch", default="diag")
    p.add_argument("--paths", type=int, default=8)
    p.add_argument("--processes", type=int, default=8)
    a = p.parse_args()

    run = ROOT / f"S1a_eki_dense_fine_{a.branch}"
    ana = run / "analysis"
    ana.mkdir(parents=True, exist_ok=True)
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    phi_rep = float(summary["decoded_final_mean"]["phi"])
    q_rep = float(summary["decoded_final_mean"]["q"])
    sd = summary["final_parameter_spread"]
    meta = summary["truth_metadata"]["weight"]
    print(f"[val] {a.branch}: reported phi = {phi_rep:.5f} +/- {sd[0]:.5f}, "
          f"q = {q_rep:.3f} +/- {sd[1]:.3f}  (truth {meta['phi_true']}, "
          f"{meta['q_true']})", flush=True)

    pool = Pool(processes=min(a.processes, a.paths),
                initializer=_init_worker_dense,
                initargs=(v3_world.DURATION_S, 20260801, False))
    try:
        _init_worker_dense(v3_world.DURATION_S, 20260801, False)
        out = pool.map(run_path, [(np.array([phi_rep, q_rep]), k)
                                  for k in range(a.paths)])
    finally:
        pool.close()
        pool.join()

    times_s = out[0][0]
    mask, _ = base._analysis_mask(times_s, base.ANALYSIS_START_S)
    gauge_model = np.stack([o[1][mask] for o in out])      # (P, T, 5)
    dense_model = np.stack([o[2][mask] for o in out])
    gauge_cols = np.asarray(base._CTX["gauge_columns"])
    dense_cols = np.asarray(base._CTX["dense_columns"])
    y_phys = np.asarray(base._CTX["y"], dtype=float)[: base.COARSE_N4] \
        * base._CTX["parameters"].lambda_ref_m
    gauge_x = 4000.0 - y_phys[gauge_cols]
    dense_x = 4000.0 - y_phys[dense_cols]

    bundle = np.load(TRUTH_DIR / "truth_bundle.npz")
    t_truth = np.asarray(bundle["times_s"], dtype=float)
    eta_truth = np.asarray(bundle["eta_paths_m"][0], dtype=float)
    mt, _ = base._analysis_mask(t_truth, base.ANALYSIS_START_S)
    gauge_truth = eta_truth[mt][:, gauge_cols]
    dense_truth = eta_truth[mt][:, dense_cols]

    ref = np.asarray(gu.load_ref_stats(TRUTH_DIR, "dense", None), dtype=float)
    sd_ref = ref.std(0, ddof=1)
    y_obs, _, _ = s1.truth_statistics_and_blocks_twin(
        TRUTH_DIR / "truth_bundle.npz", dense=True, incr=False)
    fam = dict((n, (i0, i1)) for n, i0, i1 in
               gu.family_layout("dense", n_dense=s1.N_DENSE))
    np.savez_compressed(ana / "validation_fields.npz",
                        times_s=times_s[mask], gauge_model=gauge_model,
                        dense_model=dense_model, gauge_truth=gauge_truth,
                        dense_truth=dense_truth, gauge_x=gauge_x,
                        dense_x=dense_x, theta=[phi_rep, q_rep])

    # ------------------------------------------------ derived statistics
    # (previously drawn; now saved so the run's evidence stays available)
    hs_model = 4.0 * dense_model.std(axis=1)               # (P, 40)
    base_line = np.asarray(base._CTX["dense_baseline"], dtype=float)[mask]
    dev_model = np.sqrt(((dense_model - base_line) ** 2).mean(axis=1))
    cen = gauge_model - gauge_model.mean(axis=1, keepdims=True)
    s_dev = np.maximum(gauge_model.std(axis=1), 1e-12)
    skew_model = (cen ** 3).mean(axis=1) / s_dev ** 3
    kurt_model = (cen ** 4).mean(axis=1) / s_dev ** 4 - 3.0

    ones = np.ones(int(mask.sum()), dtype=bool)
    f_hz, psd_truth, _ = base._one_sided_psd(times_s[mask], gauge_truth, ones)
    psd_model = np.stack([
        base._one_sided_psd(times_s[mask], gauge_model[i], ones)[1]
        for i in range(gauge_model.shape[0])
    ])

    np.savez_compressed(
        ana / "validation_profiles.npz",
        dense_x=dense_x, gauge_x=gauge_x,
        hs_model=hs_model, devrms_model=dev_model,
        skew_model=skew_model, kurt_model=kurt_model,
        y_obs=y_obs, sd_ref=sd_ref,
        family_slices=json.dumps({k: list(v) for k, v in fam.items()}),
        psd_freq_hz=f_hz, psd_truth=psd_truth, psd_model=psd_model,
        bands_hz=np.asarray(base.BANDS_HZ),
        theta=[phi_rep, q_rep],
    )
    print(f"[val] validation_fields.npz + validation_profiles.npz "
          f"written to {ana}", flush=True)


if __name__ == "__main__":
    main()
