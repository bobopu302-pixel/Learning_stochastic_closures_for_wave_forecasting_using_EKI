"""S2 results: learned nonlinear surface + physical validation (data only).
Usage (spec chain)::

    SW_FINE=1 SW_VERSION_DIR=v3spec SW_DURATION_S=6600 \
    SW_SYNTH_PERIOD_S=6600 SW_FORWARD_PATHS=1 SW_N_G=10 \
    python v3_s2_validation.py --branch diag --paths 8 --processes 8
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
import glob
import json
from multiprocessing import Pool

import numpy as np

import sw_eki_s1 as s1                                    # noqa: E402
import v3_world                                           # noqa: E402
import sde_closure_eki as base                            # noqa: E402
import sw_gamma_unified as gu                             # noqa: E402
import sw_eki_h as h                                      # noqa: E402

ROOT = _HERE / "results" / "stepwise" / v3_world.version_dir("v2_fine")
TRUTH_DIR = ROOT / "truth_S1a_fine"
VAL_SEED = 34000


def run_path(task):
    """One S2 realisation; returns gauge and dense fields."""
    theta, seed = task
    import sde_closure_config as closure_config
    from sde_closure_core import (
        GridWhiteNoise, GridWhiteNoiseParameters, terrain_weight,
    )
    d = h.decode_theta_h(theta)
    parameters = base._CTX["parameters"]
    surface = h.GPSurface(d["node_values"], d["obs_noise"], d["amplitude"],
                          d["l_u"], d["l_s"])
    weight = d["phi"] * terrain_weight(base._CTX["depth_ratio"], d["q"])
    rng = np.random.default_rng([VAL_SEED, int(seed)])
    noise = GridWhiteNoise(
        GridWhiteNoiseParameters(phi_amplitude=1.0,
                                 correlation_length_cells=s1.CORR_CELLS),
        base._CTX["y"], parameters.lambda_ref_m,
        base._CTX["surface_to_green"], base.COARSE_DT, rng,
        spatial_weight=weight,
    )
    with closure_config.template():
        solver = h.GPDriftSolver(
            base._CTX["y"], base._CTX["depth_ratio"], parameters.epsilon,
            parameters.mu, base.COARSE_DT, base._CTX["n_steps"],
        )
    solver.set_surface(surface, d["p"], "H2", h._CAL["s0"])
    times, out, _, _ = solver.run_stochastic(
        np.zeros_like(base._CTX["y"]), base.OUTPUT_STRIDE,
        base._CTX["traces"], noise_increment=noise,
    )
    times_s = np.asarray(times) * parameters.time_ref_s
    eta = (np.asarray(out[:, : base.COARSE_N4], dtype=float)
           * parameters.a_ref_m)
    return (times_s, eta[:, base._CTX["gauge_columns"]].astype(np.float32),
            eta[:, base._CTX["dense_columns"]].astype(np.float32))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--branch", default="diag")
    p.add_argument("--paths", type=int, default=8)
    p.add_argument("--processes", type=int, default=8)
    p.add_argument("--iteration", type=int, default=-1,
                   help="which iteration's ensemble to use (-1 = last)")
    a = p.parse_args()

    run = ROOT / f"H2_eki_{a.branch}"
    ana = run / "analysis"
    ana.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(run / "iter_*.npz")))
    if not files:
        raise SystemExit(f"no iterations in {run}")
    zs = [np.load(f) for f in files]
    thetas = [z["thetas"] for z in zs]
    phis = [z["phis"] for z in zs]
    phi_ens = [float(z["phi_ensemble"]) for z in zs if "phi_ensemble" in z]
    final = thetas[a.iteration]
    calib = h.load_calibration()
    t0, s0 = calib["t0"], calib["s0"]
    print(f"[s2-val] {len(files)} iterations, J={final.shape[0]}, "
          f"t0={t0:.4f}, s0={s0:.3f}", flush=True)

    # ------------------------------------------------ learned surface
    u_line = np.linspace(-2.5, 2.5, 121)
    s_slices = np.asarray((-1.0, -0.5, 0.5, 1.0))
    c_true = -(1.5 * h.EPSILON) * s0
    ref_surf = np.load(ROOT / "GPR_reference" / "surface.npz")
    uc, sc, bm = (ref_surf["u_centres"], ref_surf["s_centres"],
                  ref_surf["bin_means"])
    members = []
    for m in range(final.shape[0]):
        d = h.decode_theta_h(final[m])
        try:
            members.append(h.GPSurface(d["node_values"], d["obs_noise"],
                                       d["amplitude"], d["l_u"], d["l_s"]))
        except FloatingPointError:
            continue
    slice_mean = np.empty((s_slices.size, u_line.size))
    slice_sd = np.empty_like(slice_mean)
    slice_truth = np.empty_like(slice_mean)
    slice_ref = []
    for k, s_val in enumerate(s_slices):
        vals = np.stack([f(u_line, np.full_like(u_line, s_val))
                         for f in members])
        slice_mean[k] = vals.mean(0)
        slice_sd[k] = vals.std(0, ddof=1)
        slice_truth[k] = c_true * u_line * s_val
        j = int(np.argmin(np.abs(sc - s_val)))
        slice_ref.append(bm[:, j])
    np.savez_compressed(
        ana / "s2_surface_slices.npz",
        u_line=u_line, s_slices=s_slices,
        slice_mean=slice_mean, slice_sd=slice_sd, slice_truth=slice_truth,
        gpr_ref_u=uc, gpr_ref_slices=np.stack(slice_ref),
        c_true=c_true, n_members_valid=len(members),
    )

    # ------------------------------------------------------ convergence
    spread = np.array([t.std(0, ddof=1) for t in thetas])
    groups = {"gp_nodes": (0, 20), "log_hypers": (20, 24), "p": (24, 25),
              "log_phi": (25, 26), "q": (26, 27)}
    np.savez_compressed(
        ana / "s2_convergence.npz",
        phi_ensemble=np.asarray(phi_ens),
        phi_member_mean=np.asarray([np.nanmean(x) for x in phis]),
        q_over_2=0.5 * 151,
        spread_by_parameter=spread,
        group_slices=json.dumps({k: list(v) for k, v in groups.items()}),
        spread_by_group=np.stack(
            [spread[:, a0:b0].mean(axis=1) for a0, b0 in groups.values()],
            axis=1),
        group_names=list(groups.keys()),
    )

    # -------------------------------------------------- forward validation
    theta_rep = final.mean(axis=0)
    pool = Pool(processes=min(a.processes, a.paths),
                initializer=h._init_worker_h,
                initargs=(v3_world.DURATION_S, 20260801, calib))
    try:
        h._init_worker_h(v3_world.DURATION_S, 20260801, calib)
        out = pool.map(run_path, [(theta_rep, k) for k in range(a.paths)])
    finally:
        pool.close()
        pool.join()
    times_s = out[0][0]
    mask, _ = base._analysis_mask(times_s, base.ANALYSIS_START_S)
    gauge_model = np.stack([o[1][mask] for o in out])
    dense_model = np.stack([o[2][mask] for o in out])
    gauge_cols = np.asarray(base._CTX["gauge_columns"])
    dense_cols = np.asarray(base._CTX["dense_columns"])
    y_phys = (np.asarray(base._CTX["y"], dtype=float)[: base.COARSE_N4]
              * base._CTX["parameters"].lambda_ref_m)
    gauge_x, dense_x = 4000.0 - y_phys[gauge_cols], 4000.0 - y_phys[dense_cols]
    bundle = np.load(TRUTH_DIR / "truth_bundle.npz")
    t_truth = np.asarray(bundle["times_s"], dtype=float)
    eta_truth = np.asarray(bundle["eta_paths_m"][0], dtype=float)
    mt, _ = base._analysis_mask(t_truth, base.ANALYSIS_START_S)
    gauge_truth = eta_truth[mt][:, gauge_cols]
    dense_truth = eta_truth[mt][:, dense_cols]
    np.savez_compressed(ana / "validation_fields.npz", times_s=times_s[mask],
                        gauge_model=gauge_model, dense_model=dense_model,
                        gauge_truth=gauge_truth, dense_truth=dense_truth,
                        gauge_x=gauge_x, dense_x=dense_x, theta=theta_rep)

    # ------------------------------------------------ derived statistics
    ref = np.asarray(gu.load_ref_stats(TRUTH_DIR, "incr", None), dtype=float)
    sd_ref = ref.std(0, ddof=1)
    y_obs, _, _ = s1.truth_statistics_and_blocks_twin(
        TRUTH_DIR / "truth_bundle.npz", dense=True, incr=True)
    fam = dict((n, (i0, i1)) for n, i0, i1 in
               gu.family_layout("incr", n_dense=s1.N_DENSE))
    base_line = np.asarray(base._CTX["dense_baseline"], dtype=float)[mask]
    hs_model = 4.0 * dense_model.std(axis=1)
    dev_model = np.sqrt(((dense_model - base_line) ** 2).mean(axis=1))
    incr_model = np.sqrt(
        (np.diff(dense_model - base_line, axis=1) ** 2).mean(axis=1))
    cen = gauge_model - gauge_model.mean(axis=1, keepdims=True)
    sdev = np.maximum(gauge_model.std(axis=1), 1e-12)
    kurt_model = (cen ** 4).mean(axis=1) / sdev ** 4 - 3.0

    ones = np.ones(int(mask.sum()), dtype=bool)
    f_hz, psd_truth, _ = base._one_sided_psd(times_s[mask], gauge_truth, ones)
    psd_model = np.stack([
        base._one_sided_psd(times_s[mask], gauge_model[i], ones)[1]
        for i in range(gauge_model.shape[0])
    ])
    np.savez_compressed(
        ana / "validation_profiles.npz",
        dense_x=dense_x, gauge_x=gauge_x,
        hs_model=hs_model, devrms_model=dev_model, incr_model=incr_model,
        kurt_model=kurt_model,
        y_obs=y_obs, sd_ref=sd_ref,
        family_slices=json.dumps({k: list(v) for k, v in fam.items()}),
        psd_freq_hz=f_hz, psd_truth=psd_truth, psd_model=psd_model,
        bands_hz=np.asarray(base.BANDS_HZ),
        theta=theta_rep,
    )
    print(f"[s2-val] surface-slice, convergence, field and profile data "
          f"written to {ana}", flush=True)


if __name__ == "__main__":
    main()
