"""N_G calibration -- EKI_algorithm_spec_2026-08-23, section 1.3 (stage 2).
Procedure:

    for k = 1 .. K:  F_k = stats(forward run at theta_probe, length T_G)
    var_fwd = var(F_1..F_K, ddof=1);  ratio = var_fwd / var_ref
    N_G >= 5 * max(ratio)          (neglected var_fwd/N_G < 20 % of var_ref)

Probes at >= 2 points: the prior mean and a point near the expected
optimum; the larger result is taken.

Usage (spec chain)::

    SW_FINE=1 SW_VERSION_DIR=v3spec SW_DURATION_S=6600 \
    SW_SYNTH_PERIOD_S=6600 SW_FORWARD_PATHS=1 \
    python v3_calibrate_ng.py --k 20 --processes 100

Writes results/stepwise/<ver>/NG_calibration.json (and the raw stats).
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
import time
from datetime import datetime, timezone
from multiprocessing import Pool

import numpy as np

# sw_eki_s1 FIRST of all project imports: its preamble sets
# SDE_BASELINE_NPZ and calls v3_world.ensure_patched() before any frozen
# module is imported (those bind BASELINE_NPZ / _tma_inputs at import time).
from sw_eki_s1 import (                              # noqa: E402
    N_DENSE, forward_statistics_dense_fine,
)
import v3_world
import sw_gamma_unified as gu                        # noqa: E402
from sde_closure_eki_dense import _init_worker_dense  # noqa: E402

OUT_ROOT = _HERE / "results" / "stepwise" / v3_world.version_dir("v2_fine")
TRUTH_DIR = OUT_ROOT / "truth_S1a_fine"
CAL_SEED = 27000            # disjoint from y, Gamma records and EKI seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--processes", type=int, default=40)
    parser.add_argument("--layout", default="dense",
                        choices=("dense", "incr"))
    arguments = parser.parse_args()

    layout = arguments.layout
    ref_stats = gu.load_ref_stats(TRUTH_DIR, layout, None)
    var_ref = np.var(np.asarray(ref_stats, dtype=float), axis=0, ddof=1)
    q = var_ref.size
    fam = gu.family_layout(layout, n_dense=N_DENSE)

    # probe 1: prior mean of the S1 prior  U(0.003, 0.03) x U(0, 5)
    # probe 2: near the expected optimum (the twin truth of this case)
    metadata = json.loads(
        (TRUTH_DIR / "metadata.json").read_text(encoding="utf-8"))
    phi_true = float(metadata["weight"]["phi_true"])
    q_true = float(metadata["weight"]["q_true"])
    probes = {
        "prior_mean": (0.5 * (0.003 + 0.03), 2.5),
        "near_optimum": (phi_true, q_true),
    }

    mode, envelope = "B", "terrain"
    colored = flat_sigma = False
    no_damping = True
    duration = v3_world.DURATION_S
    print(f"[ng-cal] q={q} layout={layout} K={arguments.k} "
          f"T_G={duration:.0f}s paths/realisation={v3_world.FORWARD_PATHS} "
          f"probes={ {k: (round(v[0], 5), v[1]) for k, v in probes.items()} }",
          flush=True)

    pool = Pool(processes=arguments.processes,
                initializer=_init_worker_dense,
                initargs=(duration, 20260801, False))
    result: dict[str, object] = {}
    raw_stats: dict[str, np.ndarray] = {}
    try:
        # all probes in ONE batch: K x n_probes tasks are submitted
        # together so the pool is not left idle between probes
        all_tasks, spans = [], {}
        for probe_index, (name, (phi, q_par)) in enumerate(probes.items()):
            # phi lives in log coordinates inside the forward map (spec)
            theta = np.array([np.log(phi), q_par], dtype=float)
            spans[name] = (len(all_tasks), len(all_tasks) + arguments.k,
                           theta)
            all_tasks += [
                (theta, mode, envelope, colored, flat_sigma, no_damping,
                 (CAL_SEED, probe_index, k))
                for k in range(arguments.k)
            ]
        started_all = time.perf_counter()
        flat = pool.map(forward_statistics_dense_fine, all_tasks)
        print(f"[ng-cal] {len(all_tasks)} realisations done in "
              f"{time.perf_counter() - started_all:.0f}s", flush=True)
        for name, (i0, i1, theta) in spans.items():
            started = time.perf_counter()
            stats = [s for s in flat[i0:i1] if s is not None]
            if len(stats) < 3:
                raise RuntimeError(f"probe {name}: only {len(stats)} of "
                                   f"{arguments.k} realisations finite")
            arr = np.asarray(stats)
            raw_stats[name] = arr
            var_fwd = arr.var(axis=0, ddof=1)
            ratio = var_fwd / np.where(var_ref > 0, var_ref, np.inf)
            by_family = {n: float(np.nanmax(ratio[a:b])) for n, a, b in fam}
            result[name] = {
                "theta": theta.tolist(),
                "realisations_finite": len(stats),
                "ratio_min": float(np.nanmin(ratio)),
                "ratio_median": float(np.nanmedian(ratio)),
                "ratio_p90": float(np.nanpercentile(ratio, 90)),
                "ratio_max": float(np.nanmax(ratio)),
                "ratio_argmax": int(np.nanargmax(ratio)),
                "ratio_by_family": by_family,
                "N_G_required": int(max(1, np.ceil(5.0 * np.nanmax(ratio)))),
                "wall_s": round(time.perf_counter() - started),
            }
            print(f"[ng-cal] {name}: {len(stats)}/{arguments.k} finite, "
                  f"ratio min/med/p90/max="
                  f"{result[name]['ratio_min']:.3g}/"
                  f"{result[name]['ratio_median']:.3g}/"
                  f"{result[name]['ratio_p90']:.3g}/"
                  f"{result[name]['ratio_max']:.3g} -> N_G >= "
                  f"{result[name]['N_G_required']}  "
                  f"({result[name]['wall_s']}s)", flush=True)
            print(f"[ng-cal]   by family: "
                  + ", ".join(f"{k}={v:.2g}" for k, v in by_family.items()),
                  flush=True)
    finally:
        pool.close()
        pool.join()

    n_g = max(int(r["N_G_required"]) for r in result.values())  # type: ignore
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "spec": "EKI_algorithm_spec_2026-08-23 section 1.3",
        "T_G_s": duration,
        "analysis_start_s": v3_world.ANALYSIS_START_S,
        "paths_per_realisation": v3_world.FORWARD_PATHS,
        "K": arguments.k,
        "layout": layout,
        "q": int(q),
        "N_Gamma": int(np.asarray(ref_stats).shape[0]),
        "probes": result,
        "N_G_chosen": n_g,
        "seed_scheme": f"[{CAL_SEED}, probe, k]",
    }
    (OUT_ROOT / "NG_calibration.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(OUT_ROOT / "NG_calibration_stats.npz",
                        var_ref=var_ref, **raw_stats)
    print(f"[ng-cal] N_G chosen = {n_g}  -> "
          f"{OUT_ROOT / 'NG_calibration.json'}", flush=True)


if __name__ == "__main__":
    main()
