"""Independent truth reference records for the spec Gamma (stage 1).
Generates N_Gamma independent realisations of the S1a twin truth process,
evaluates the statistic vector on each record with the same pipeline as
the official observation y, and stores ONLY the statistics (small) under

    <truth_dir>/ref_records/ref_stats_<layout>.npz
        ref_stats  (N_Gamma, q)   ref_seeds (N_Gamma,)   layout, n_paths, ...

The eta fields themselves are NOT kept; each record's stats for ALL three
layouts (standard, dense, incr) are computed in one pass so the drivers
can share the records.  Gamma itself is assembled later by
sw_gamma_unified.build_gamma_unified as diag(var_ref) (or the shrunk full
Cov_ref) over these records.

Resumable: finished records are appended to a per-record cache and skipped
on restart (--resume); if every layout's ref_stats file already holds
>= --n-ref records nothing is recomputed.

Usage (spec chain)::

    SW_FINE=1 SW_VERSION_DIR=v3spec SW_DURATION_S=6600 \
    SW_SYNTH_PERIOD_S=6600 SW_FORWARD_PATHS=1 SW_REF_NEW_SEA_STATE=1 \
    python sw_ref_records.py --n-ref 50 --paths 1 --processes 50 --resume
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
import v3_world  # noqa: E402  (v3 twin world patches; no-op for v2)
_VERSION_DIR = v3_world.version_dir("v2_fine" if SW_FINE else "v1_coarse")
_TRUTH_NAME = "truth_S1a_fine" if SW_FINE else "truth_S1a"
os.environ.setdefault("SW_S1_VARIANT", "S1a")
if SW_FINE:
    os.environ.setdefault("SW_S1_SUFFIX", "_fine")
os.environ.setdefault(
    "SDE_BASELINE_NPZ",
    str(_HERE / "results" / "stepwise" / _VERSION_DIR / _TRUTH_NAME
        / "baseline_data.npz"),
)
v3_world.ensure_patched()

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Pool

import numpy as np

# sw_eki_s1's preamble sets SDE_BASELINE_NPZ and applies the v3 patches
# before any frozen module import; it must come first of the project
# imports below.
from sw_eki_s1 import CORR_CELLS, N_DENSE, truth_stats_from_eta  # noqa
from sw_truth import FINE_MODEL_KW, PHI_TRUE, Q_TRUE  # noqa: E402
from sde_closure_context import CoarseModelContext, ModelAConfig  # noqa
from sde_closure_core import (  # noqa: E402
    GridWhiteNoise, GridWhiteNoiseParameters, terrain_weight,
)

TRUTH_DIR = _HERE / "results" / "stepwise" / _VERSION_DIR / _TRUTH_NAME
REF_DIR = TRUTH_DIR / "ref_records"
REF_SEED_BASE = 20260901          # disjoint from truth [20260801,m],
                                  # validation [20260810,m], H2 [20260814,m]
N_PATHS_REF = v3_world.FORWARD_PATHS   # = official truth record (1 in v3)

LAYOUTS = ("standard", "dense", "incr")


# ----------------------------------------------------------------------
# Spec mode (SW_REF_NEW_SEA_STATE=1): every reference record is a fresh
# realisation of the WHOLE truth process -- its own incident phases
# (boundary seed REF_BOUNDARY_BASE + r), its own deterministic baseline
# (needed by the deviation statistics) and its own closure-noise paths.
# Baselines and paths are independent pool tasks so that a box with
# many cores finishes a record in one path-time instead of two.
# ----------------------------------------------------------------------
REF_BOUNDARY_BASE = 20260901 * 10    # 202609010 + r: disjoint from the
                                     # official boundary seed 20260801


def record_boundary_seed(record_index: int) -> int:
    return REF_BOUNDARY_BASE + int(record_index)


def _record_context(record_index: int):
    config = ModelAConfig(boundary_seed=record_boundary_seed(record_index),
                          **FINE_MODEL_KW)
    return config, CoarseModelContext(config)


def _baseline_task(record_index: int):
    """Deterministic (no-noise) run of record r's incident sea state."""
    config, context = _record_context(record_index)
    solver = context.make_solver()
    times, surface, _, _ = solver.run(
        np.zeros_like(context.y), config.output_stride, context.traces)
    n4 = config.coarse_n4
    eta = (np.asarray(surface[:, :n4], dtype=float)
           * context.parameters.a_ref_m)
    times_s = np.asarray(times) * context.parameters.time_ref_s
    return ("base", record_index, times_s, eta.astype(np.float32),
            np.asarray(context.y_physical_m, dtype=float))


def _path_task(task):
    """One closure-noise path of record r (truth envelope, fresh seed)."""
    record_index, path = task
    config, context = _record_context(record_index)
    weight = PHI_TRUE * terrain_weight(context.depth_ratio, Q_TRUE)
    rng = np.random.default_rng([REF_SEED_BASE, record_index, path])
    noise = GridWhiteNoise(
        GridWhiteNoiseParameters(
            phi_amplitude=1.0, correlation_length_cells=CORR_CELLS),
        context.y, context.parameters.lambda_ref_m,
        context.surface_to_green, config.coarse_dt, rng,
        spatial_weight=weight,
    )
    solver = context.make_solver()
    times, surface, _, _ = solver.run_stochastic(
        np.zeros_like(context.y), config.output_stride,
        context.traces, noise_increment=noise,
    )
    n4 = config.coarse_n4
    eta = (np.asarray(surface[:, :n4], dtype=float)
           * context.parameters.a_ref_m)
    if not np.all(np.isfinite(eta)):
        return ("path", record_index, path, None)
    return ("path", record_index, path, eta.astype(np.float32))


def _record_stats_v3(times_s, baseline_eta, y_physical, eta_paths):
    common = min(times_s.size, baseline_eta.shape[0], eta_paths.shape[1])
    return {
        layout: truth_stats_from_eta(
            times_s[:common], eta_paths[:, :common], baseline_eta[:common],
            y_physical, dense=(layout != "standard"),
            incr=(layout == "incr"),
        )
        for layout in LAYOUTS
    }


def _run_v3(arguments, done: dict, cache) -> dict:
    todo = [i for i in range(arguments.n_ref) if i not in done]
    print(f"[ref-records] v3 mode: new sea state + own baseline per record; "
          f"N_R={arguments.n_ref} paths/record={arguments.paths} "
          f"todo={len(todo)} processes={arguments.processes} "
          f"duration={v3_world.DURATION_S:.0f}s "
          f"synth={v3_world.SYNTH_PERIOD_S:.0f}s")
    if not todo:
        return done
    started = time.perf_counter()
    pending: dict[int, dict] = {i: {"paths": {}} for i in todo}
    with Pool(processes=arguments.processes) as pool:
        base_jobs = [pool.apply_async(_baseline_task, (i,)) for i in todo]
        path_jobs = [pool.apply_async(_path_task, ((i, p),))
                     for i in todo for p in range(arguments.paths)]
        jobs = base_jobs + path_jobs
        remaining = set(range(len(jobs)))
        while remaining:
            finished = [k for k in remaining if jobs[k].ready()]
            if not finished:
                time.sleep(5.0)
                continue
            for k in finished:
                remaining.discard(k)
                result = jobs[k].get()
                kind, idx = result[0], result[1]
                rec = pending[idx]
                if kind == "base":
                    _, _, times_s, eta, y_physical = result
                    rec.update(times_s=times_s, baseline=eta,
                               y_physical=y_physical)
                else:
                    _, _, path, eta = result
                    if eta is None:
                        print(f"[ref-records] record {idx} path {path} "
                              f"FAILED (non-finite)")
                        rec["failed"] = True
                    rec["paths"][path] = eta
                complete = ("baseline" in rec
                            and len(rec["paths"]) == arguments.paths)
                if complete and not rec.get("failed"):
                    eta_paths = np.stack(
                        [np.asarray(rec["paths"][p], dtype=float)
                         for p in range(arguments.paths)])
                    done[idx] = _record_stats_v3(
                        rec["times_s"],
                        np.asarray(rec["baseline"], dtype=float),
                        rec["y_physical"], eta_paths)
                    np.savez(cache, records=np.array(done, dtype=object))
                    print(f"[ref-records] record {idx} done "
                          f"({len(done)}/{arguments.n_ref}, "
                          f"{time.perf_counter()-started:.0f}s)")
                    pending[idx] = {"paths": {}}      # free the fields
                elif complete:
                    pending[idx] = {"paths": {}}
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-ref", type=int, default=20)
    parser.add_argument("--paths", type=int, default=N_PATHS_REF)
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()

    if not v3_world.REF_NEW_SEA_STATE:
        raise SystemExit(
            "the spec Gamma requires SW_REF_NEW_SEA_STATE=1 (every "
            "reference record gets its own incident sea state and its own "
            "deterministic baseline); the legacy shared-sea-state record "
            "mode is not shipped")
    if not TRUTH_DIR.exists():
        raise SystemExit(f"truth dir missing: {TRUTH_DIR}")
    REF_DIR.mkdir(parents=True, exist_ok=True)
    # Finished-product short-circuit: if every layout's ref_stats file
    # already holds >= n-ref records, there is nothing to compute (the
    # per-record cache may be absent when the stats were shipped to
    # another machine; the stats files ARE the deliverable).
    if arguments.resume:
        done_files = []
        for layout in LAYOUTS:
            f = REF_DIR / f"ref_stats_{layout}.npz"
            if f.exists():
                z = np.load(f)
                done_files.append(z["ref_stats"].shape[0] >= arguments.n_ref)
        if len(done_files) == len(LAYOUTS) and all(done_files):
            print(f"[ref-records] ref_stats complete for all layouts "
                  f"(>= {arguments.n_ref} records) -- nothing to do")
            return
    cache = REF_DIR / "records_cache.npz"
    done: dict[int, dict] = {}
    if arguments.resume and cache.exists():
        z = np.load(cache, allow_pickle=True)
        done = {int(k): v for k, v in z["records"].item().items()}
        print(f"[ref-records] resumed {len(done)} finished records")

    if arguments.resume and not done:
        files = {lay: REF_DIR / f"ref_stats_{lay}.npz" for lay in LAYOUTS}
        if all(f.exists() for f in files.values()):
            loaded = {lay: np.load(f) for lay, f in files.items()}
            seeds = np.asarray(loaded[LAYOUTS[0]]["ref_seeds"])
            for k, (seed_base, idx) in enumerate(seeds):
                if int(seed_base) != REF_SEED_BASE:
                    continue
                done[int(idx)] = {lay: np.asarray(loaded[lay]["ref_stats"][k])
                                  for lay in LAYOUTS}
            print(f"[ref-records] rebuilt {len(done)} finished records from "
                  f"ref_stats files (no per-record cache present)")
    todo = [i for i in range(arguments.n_ref) if i not in done]
    print(f"[ref-records] N_R={arguments.n_ref} paths/record="
          f"{arguments.paths} todo={len(todo)} processes="
          f"{arguments.processes} corr_cells={CORR_CELLS}")
    if todo:
        done = _run_v3(arguments, done, cache)

    if len(done) < 3:
        raise SystemExit("fewer than 3 records -- cannot build Gamma")
    order = sorted(done)
    for layout in LAYOUTS:
        ref_stats = np.stack([done[i][layout] for i in order])
        np.savez_compressed(
            REF_DIR / f"ref_stats_{layout}.npz",
            ref_stats=ref_stats,
            ref_seeds=np.array([[REF_SEED_BASE, i] for i in order]),
            layout=layout, n_paths=arguments.paths,
            n_dense=N_DENSE, corr_cells=CORR_CELLS,
        )
        print(f"[ref-records] {layout}: ref_stats {ref_stats.shape}")
    (REF_DIR / "metadata.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "n_ref_records": len(order),
        "n_paths_per_record": arguments.paths,
        "seed_scheme": f"[{REF_SEED_BASE}, record, path]",
        "boundary_seed_scheme": (
            f"{REF_BOUNDARY_BASE} + record (own incident phases + own "
            f"deterministic baseline per record)"),
        "world": v3_world.describe(),
        "truth_dir": str(TRUTH_DIR),
        "layouts": list(LAYOUTS),
        "note": "UNIFY_y_G_Gamma_spec 1.3 method (a): reference-side "
                "variance = sample variance of stats over these "
                "independent records; y itself remains the single "
                "official truth record.",
    }, indent=2), encoding="utf-8")
    print("[ref-records] done")


if __name__ == "__main__":
    main()
