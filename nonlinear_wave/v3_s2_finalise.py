"""Write S2's summary.json from the per-iteration audit files.

Origin: 3. KDV_nonlinear_case/v3_s2_finalise.py
Changes vs origin:
- dropped the MPLBACKEND setdefault (nothing in this release plots);
- comments/docstrings only otherwise.

Same role as v3_s1_finalise.py: the run is stopped at a fixed iteration
count (the spec's stop rule cannot trigger under the Monte-Carlo noise of
Phi), so the summary is assembled from the audit files and records the
truncation point and its reason, plus the recovery metrics against the
bilinear truth surface.

    python v3_s2_finalise.py --branch diag --iterations 10
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
from datetime import datetime, timezone

import numpy as np

import sw_eki_s1 as s1                                    # noqa: E402,F401
import v3_world                                           # noqa: E402
import sw_eki_h as h                                      # noqa: E402

ROOT = _HERE / "results" / "stepwise" / v3_world.version_dir("v2_fine")
TRUTH_DIR = ROOT / "truth_S1a_fine"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--branch", default="diag")
    p.add_argument("--iterations", type=int, default=0)
    p.add_argument("--reason", default=(
        "stopped at a fixed iteration count: Phi had plateaued within "
        "10 % of the value attained by the TRUE parameters (measured "
        "separately, S2_truth_phi.json), and the spec's stop rule "
        "(relative change below 1 % for three iterations) cannot trigger "
        "under the Monte-Carlo fluctuation of Phi"))
    a = p.parse_args()

    run = ROOT / f"H2_eki_{a.branch}"
    files = sorted(glob.glob(str(run / "iter_*.npz")))
    if a.iterations:
        files = files[: a.iterations]
    if not files:
        raise SystemExit(f"no iteration files in {run}")
    zs = [np.load(f) for f in files]
    thetas = [z["thetas"] for z in zs]
    phis = [z["phis"] for z in zs]
    phi_ens = [float(z["phi_ensemble"]) for z in zs if "phi_ensemble" in z]
    final = thetas[-1]

    calib = h.load_calibration()
    t0, s0 = float(calib["t0"]), float(calib["s0"])
    decoded = [h.decode_theta_h(t) for t in final]
    scalars = [k for k, v in decoded[0].items()
               if not isinstance(v, np.ndarray)]
    mean = {k: float(np.mean([d[k] for d in decoded])) for k in scalars}
    sd = {k: float(np.std([d[k] for d in decoded], ddof=1)) for k in scalars}
    nodes = np.array([d["node_values"] for d in decoded])
    mean["node_values"] = nodes.mean(0).tolist()
    sd["node_values"] = nodes.std(0, ddof=1).tolist()

    # recovery against the bilinear truth surface
    c_true = -(1.5 * h.EPSILON) * s0
    uu, ss = np.meshgrid(np.linspace(-2.0, 2.0, 25),
                         np.linspace(-1.5, 1.5, 19), indexing="ij")
    m_true = c_true * uu * ss
    fields = []
    for d in decoded:
        try:
            g = h.GPSurface(d["node_values"], d["obs_noise"], d["amplitude"],
                            d["l_u"], d["l_s"])
        except FloatingPointError:
            continue
        fields.append(g(uu.ravel(), ss.ravel()).reshape(uu.shape))
    field = np.mean(fields, axis=0)
    meta = json.loads((TRUTH_DIR / "metadata.json").read_text(
        encoding="utf-8"))["weight"]
    truth_phi_path = ROOT / "S2_truth_phi.json"
    phi_at_truth = (json.loads(truth_phi_path.read_text(encoding="utf-8"))
                    ["phi_at_truth"] if truth_phi_path.exists() else None)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "driver": "sw_eki_h (finalised by v3_s2_finalise)",
        "variant": "H2", "branch": a.branch,
        "algorithm": "spec_2026-08-23",
        "parameter_choice": "final_ensemble_mean",
        "stopping": {"rule": "fixed iteration count",
                     "iterations_used": len(files), "reason": a.reason},
        "iterations_run": len(files),
        "members": int(final.shape[0]),
        "N_G": int(os.environ.get("SW_N_G", "1")),
        "q_obs": 151,
        "calibration": {"t0": t0, "s0": s0},
        "decoded_final_mean": mean,
        "decoded_final_sd": sd,
        "phi_ensemble_history": phi_ens,
        "mean_phi_history": [float(np.nanmean(x)) for x in phis],
        "phi_at_true_parameters": phi_at_truth,
        "recovery": {
            "phi_true": float(meta["phi_true"]),
            "q_true": float(meta["q_true"]),
            "p_true_reference": -0.5,
            "phi_mean": mean["phi"], "q_mean": mean["q"], "p_mean": mean["p"],
            "phi_rel_error": mean["phi"] / float(meta["phi_true"]) - 1.0,
            "q_rel_error": mean["q"] / float(meta["q_true"]) - 1.0,
            "surface_shape_correlation": float(
                np.corrcoef(field.ravel(), m_true.ravel())[0, 1]),
            "surface_amplitude_ratio": float(
                np.sqrt((field ** 2).mean()) / np.sqrt((m_true ** 2).mean())),
            "bilinear_coeff_true": float(c_true),
        },
        "world": v3_world.describe(),
    }
    diag = run / "gamma_unified_diag.json"
    if diag.exists():
        summary.update({k: v for k, v in
                        json.loads(diag.read_text(encoding="utf-8")).items()
                        if k not in summary})
    (run / "summary.json").write_text(json.dumps(summary, indent=2),
                                      encoding="utf-8")
    r = summary["recovery"]
    print(f"[finalise] {run.name}: {len(files)} iterations, Phi "
          f"{phi_ens[-1]:.1f} (truth parameters: {phi_at_truth}), "
          f"phi {r['phi_mean']:.5f} ({r['phi_rel_error']*100:+.1f} %), "
          f"q {r['q_mean']:.3f} ({r['q_rel_error']*100:+.1f} %), "
          f"p {r['p_mean']:+.2f} (truth -0.50), surface shape corr "
          f"{r['surface_shape_correlation']:+.2f}, amplitude ratio "
          f"{r['surface_amplitude_ratio']:.2f}", flush=True)
    print(f"[finalise] summary written: {run / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
