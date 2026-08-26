"""Write S1's summary.json from the per-iteration audit files.

Origin: 3. KDV_nonlinear_case/v3_s1_finalise.py
Changes vs origin:
- dropped the MPLBACKEND setdefault (nothing in this release plots);
- comments/docstrings only otherwise (spec-only sw_eki_s1 always has
  SPEC=True, so the log-phi decode branch is unconditional).

Used when the run is stopped at a fixed iteration count instead of by the
spec's stop rule: with N_G realisations per evaluation and fresh common
random numbers each iteration, Phi keeps a Monte-Carlo fluctuation of a
few tens of per cent, so "relative change of Phi below 1 % for three
consecutive iterations" can never trigger and the run would always hit
the iteration cap.  The truncation point and this reason are recorded in
the summary.

Note on iter files: the engine adds one extra audit file holding the
evaluation of the final post-update ensemble when the iteration cap is
reached; with no --iterations limit this finalise therefore reports the
true final ensemble (the one the spec's final-ensemble-mean rule refers
to).  Pass --iterations to truncate to a fixed in-loop count instead.

    python v3_s1_finalise.py --branch diag --reason "..." [--iterations 10]
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

import sw_eki_s1 as s1                                   # noqa: E402
import v3_world                                          # noqa: E402

ROOT = _HERE / "results" / "stepwise" / v3_world.version_dir("v2_fine")
TRUTH = ROOT / "truth_S1a_fine"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--branch", default="diag")
    p.add_argument("--iterations", type=int, default=0,
                   help="use only the first N iterations (0 = all present)")
    p.add_argument("--reason", default=(
        "stopped at a fixed iteration count: the spec's stop rule "
        "(relative change of Phi below 1 % for three consecutive "
        "iterations) cannot trigger because Phi carries a Monte-Carlo "
        "fluctuation of a few tens of per cent between iterations "
        "(fresh common random numbers each iteration at N_G = 10)"))
    a = p.parse_args()

    run = ROOT / f"S1a_eki_dense_fine_{a.branch}"
    files = sorted(glob.glob(str(run / "iter_*.npz")))
    if a.iterations:
        files = files[: a.iterations]
    if not files:
        raise SystemExit(f"no iteration files in {run}")
    thetas = [np.load(f)["thetas"] for f in files]
    phis = [np.load(f)["phis"] for f in files]
    phi_ens = [float(np.load(f)["phi_ensemble"]) for f in files
               if "phi_ensemble" in np.load(f)]
    final = thetas[-1]
    model = np.array(final, dtype=float, copy=True)
    model[:, 0] = np.exp(model[:, 0])          # phi lives in log coords
    mean = model.mean(axis=0)
    sd = model.std(axis=0, ddof=1)

    # Best member over all evaluated iterations (kept for reference only;
    # the spec reports the final ensemble mean).
    best_it = int(np.argmin([np.nanmin(p) for p in phis]))
    best_idx = int(np.nanargmin(phis[best_it]))
    best = np.array(thetas[best_it][best_idx], dtype=float, copy=True)
    best[0] = float(np.exp(best[0]))

    meta = json.loads((TRUTH / "metadata.json").read_text(encoding="utf-8"))
    phi_true = float(meta["weight"]["phi_true"])
    q_true = float(meta["weight"]["q_true"])
    gamma_diag = {}
    diag_path = run / "gamma_unified_diag.json"
    if diag_path.exists():
        gamma_diag = json.loads(diag_path.read_text(encoding="utf-8"))

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "driver": "sw_eki_s1 (finalised by v3_s1_finalise)",
        "config": "S1a",
        "branch": a.branch,
        "algorithm": "spec_2026-08-23" if s1.SPEC else "legacy",
        "parameter_choice": "final_ensemble_mean",
        "stopping": {"rule": "fixed iteration count",
                     "iterations_used": len(files),
                     "reason": a.reason},
        "iterations_run": len(files),
        "members": int(final.shape[0]),
        "N_G": int(os.environ.get("SW_N_G", "1")),
        "q_obs": 111,
        "log_coordinates": ["phi"] if s1.SPEC else [],
        "parameter_names": ["phi", "q"],
        "decoded_final_mean": {"phi": float(mean[0]), "lambda": 0.0,
                               "q": float(mean[1])},
        "final_parameter_mean": [float(v) for v in mean],
        "final_parameter_spread": [float(v) for v in sd],
        "decoded_best": {"phi": float(best[0]), "lambda": 0.0,
                         "q": float(best[1])},
        "phi_best": float(np.nanmin(phis[best_it])),
        "phi_ensemble_history": phi_ens,
        "mean_phi_history": [float(np.nanmean(p)) for p in phis],
        "recovery": {
            "phi_true": phi_true, "q_true": q_true,
            "phi_mean": float(mean[0]), "q_mean": float(mean[1]),
            "phi_rel_error_mean": float((mean[0] - phi_true) / phi_true),
            "q_rel_error_mean": float((mean[1] - q_true) / q_true),
            "phi_sd": float(sd[0]), "q_sd": float(sd[1]),
        },
        "truth_metadata": meta,
        "world": v3_world.describe(),
    }
    summary.update({k: v for k, v in gamma_diag.items()
                    if k not in summary})
    (run / "summary.json").write_text(json.dumps(summary, indent=2),
                                      encoding="utf-8")
    print(f"[finalise] {run.name}: {len(files)} iterations, "
          f"phi = {mean[0]:.5f} +/- {sd[0]:.5f} "
          f"({(mean[0]-phi_true)/phi_true*100:+.1f} % of truth), "
          f"q = {mean[1]:.3f} +/- {sd[1]:.3f} "
          f"({(mean[1]-q_true)/q_true*100:+.1f} %)", flush=True)
    print(f"[finalise] summary written: {run / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
