"""Forward validation paths at the TRUE (or zero) S2 parameters.
Companion to v3_s2_validation.py: runs the same realisations, with the
same noise seeds, but with the truth's GP nodes, p = -1/2 and the true
noise parameters.  Saves `validation_fields_truth.npz` (or `_zero.npz`)
next to the learned run's `validation_fields.npz`, so the physical
comparison can be drawn three ways -- observed record, inversion, and the
best the model class can do (v3_s2_threeway.py consumes all three).

Usage (spec chain)::

    SW_FINE=1 SW_VERSION_DIR=v3spec SW_DURATION_S=6600 \
    SW_SYNTH_PERIOD_S=6600 SW_FORWARD_PATHS=1 SW_N_G=10 \
    python v3_s2_val_truth.py --branch diag --paths 32 --processes 32
    python v3_s2_val_truth.py --branch diag --paths 32 --processes 32 \
        --mode zero
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

import sw_eki_s1 as s1                                    # noqa: E402,F401
import v3_world                                           # noqa: E402
import sde_closure_eki as base                            # noqa: E402
import sw_eki_h as h                                      # noqa: E402
import v3_s2_validation as val                            # noqa: E402

ROOT = _HERE / "results" / "stepwise" / v3_world.version_dir("v2_fine")
TRUTH_DIR = ROOT / "truth_S1a_fine"


def truth_theta(t0: float, s0: float, mode: str = "truth") -> np.ndarray:
    """theta holding m^dagger, p = -1/2 and the true noise parameters.

    `mode="zero"` keeps everything but sets the surface to m == 0, i.e.
    the S1 model with no learned nonlinear term at all -- the baseline
    S2 is supposed to improve on.
    """
    meta = json.loads((TRUTH_DIR / "metadata.json").read_text(
        encoding="utf-8"))["weight"]
    theta = np.zeros(h.THETA_DIM)
    if mode == "truth":
        theta[: h.R_NODES] = h.true_surface_values(t0, s0)
    theta[h.I_LOG[0]] = np.log(1.0e-2 * t0)      # tiny nugget: interpolate
    theta[h.I_LOG[1]] = np.log(t0)               # amplitude
    theta[h.I_LOG[2]] = np.log(1.25)             # l_u = node spacing
    theta[h.I_LOG[3]] = np.log(1.00)             # l_s = node spacing
    theta[h.I_P] = -0.5
    phi_true = float(meta["phi_true"])
    theta[h.I_PHI] = np.log(phi_true)            # log coordinates (spec)
    theta[h.I_Q] = float(meta["q_true"])
    return theta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--branch", default="diag")
    p.add_argument("--paths", type=int, default=32)
    p.add_argument("--processes", type=int, default=32)
    p.add_argument("--mode", default="truth", choices=("truth", "zero"),
                   help="truth = m^dagger; zero = no nonlinear term at all")
    a = p.parse_args()

    ana = ROOT / f"H2_eki_{a.branch}" / "analysis"
    ana.mkdir(parents=True, exist_ok=True)
    calib = h.load_calibration()
    t0, s0 = float(calib["t0"]), float(calib["s0"])
    theta = truth_theta(t0, s0, a.mode)
    d = h.decode_theta_h(theta)
    print("[val-%s] p = %.2f, phi = %.5f, q = %.3f, paths = %d"
          % (a.mode, d["p"], d["phi"], d["q"], a.paths), flush=True)

    pool = Pool(processes=min(a.processes, a.paths),
                initializer=h._init_worker_h,
                initargs=(v3_world.DURATION_S, 20260801, calib))
    try:
        h._init_worker_h(v3_world.DURATION_S, 20260801, calib)
        out = pool.map(val.run_path, [(theta, k) for k in range(a.paths)])
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
    out_path = ana / ("validation_fields_%s.npz" % a.mode)
    np.savez_compressed(
        out_path, times_s=times_s[mask],
        gauge_model=gauge_model, dense_model=dense_model,
        gauge_x=4000.0 - y_phys[gauge_cols],
        dense_x=4000.0 - y_phys[dense_cols], theta=theta, paths=a.paths,
        mode=a.mode)
    print("[val-%s] %d paths written to %s" % (a.mode, a.paths, out_path),
          flush=True)


if __name__ == "__main__":
    main()
