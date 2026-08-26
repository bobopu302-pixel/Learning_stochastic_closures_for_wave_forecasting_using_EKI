"""Phi at the TRUE S2 parameters -- the target the inversion can reach.

Origin: 3. KDV_nonlinear_case/v3_s2_truth_phi.py
Changes vs origin:
- dropped the MPLBACKEND setdefault (nothing in this release plots);
- comments/docstrings only otherwise (this script never plotted).

Builds theta with the truth's GP nodes (m(u,s) = -(3 eps/2) s0 u s), the
true depth exponent p = -1/2 and the true noise parameters, evaluates
G_hat as the run does (mean of N_G realisations with the same CRN seed
scheme) and reports Phi and its per-family split.  v3_s2_finalise.py
records the result as the measured floor the inversion is judged against.

Usage (spec chain)::

    SW_FINE=1 SW_VERSION_DIR=v3spec SW_DURATION_S=6600 \
    SW_SYNTH_PERIOD_S=6600 SW_FORWARD_PATHS=1 SW_N_G=10 \
    python v3_s2_truth_phi.py --processes 10
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
import sw_gamma_unified as gu                             # noqa: E402
import sw_eki_h as h                                      # noqa: E402

ROOT = _HERE / "results" / "stepwise" / v3_world.version_dir("v2_fine")
TRUTH_DIR = ROOT / "truth_S1a_fine"
SEED = 35000


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--processes", type=int, default=10)
    p.add_argument("--n-g", type=int,
                   default=int(os.environ.get("SW_N_G", "10")))
    a = p.parse_args()

    calib = h.load_calibration()
    t0, s0 = calib["t0"], calib["s0"]
    meta = json.loads((TRUTH_DIR / "metadata.json").read_text(
        encoding="utf-8"))["weight"]
    phi_true, q_true = float(meta["phi_true"]), float(meta["q_true"])

    theta = np.zeros(h.THETA_DIM)
    theta[: h.R_NODES] = h.true_surface_values(t0, s0)
    theta[h.I_LOG[0]] = np.log(1.0e-2 * t0)      # tiny nugget: interpolate
    theta[h.I_LOG[1]] = np.log(t0)               # amplitude
    theta[h.I_LOG[2]] = np.log(1.25)             # l_u = node spacing
    theta[h.I_LOG[3]] = np.log(1.00)             # l_s = node spacing
    theta[h.I_P] = -0.5
    theta[h.I_PHI] = np.log(phi_true)            # log coordinates (spec)
    theta[h.I_Q] = q_true
    d = h.decode_theta_h(theta)
    print(f"[truth-phi] nodes max |m| = {np.abs(theta[:h.R_NODES]).max():.4f}, "
          f"p = {d['p']}, phi = {d['phi']:.5f}, q = {d['q']}, N_G = {a.n_g}",
          flush=True)

    tasks = [(theta, "H2", (SEED, 0, k)) for k in range(a.n_g)]
    pool = Pool(processes=min(a.processes, a.n_g),
                initializer=h._init_worker_h,
                initargs=(v3_world.DURATION_S, 20260801, calib))
    try:
        raw = pool.map(h.forward_statistics_h, tasks, chunksize=1)
    finally:
        pool.close()
        pool.join()
    ok = [r for r in raw if r is not None]
    if not ok:
        raise SystemExit("every realisation failed")
    g_hat = np.mean(np.asarray(ok), axis=0)

    ref = np.asarray(gu.load_ref_stats(TRUTH_DIR, "incr", None), dtype=float)
    var_ref = ref.var(0, ddof=1)
    y, _, _ = s1.truth_statistics_and_blocks_twin(
        TRUTH_DIR / "truth_bundle.npz", dense=True, incr=True)
    z = (g_hat - y) / np.sqrt(var_ref)
    phi_val = 0.5 * float((z ** 2).sum())
    print(f"[truth-phi] {len(ok)}/{a.n_g} finite -> Phi at the TRUE "
          f"parameters = {phi_val:.1f}   (q/2 = {y.size/2:.1f})", flush=True)
    fam = gu.family_layout("incr", n_dense=s1.N_DENSE)
    for name, i0, i1 in fam:
        part = 0.5 * float((z[i0:i1] ** 2).sum())
        print(f"   {name:<8} {part:>9.1f}  ({100*part/phi_val:>5.1f} %)  "
              f"median |z| {np.median(np.abs(z[i0:i1])):.2f}", flush=True)
    out = ROOT / "S2_truth_phi.json"
    out.write_text(json.dumps({
        "phi_at_truth": phi_val, "q_over_2": y.size / 2,
        "n_g": a.n_g, "realisations_finite": len(ok),
        "theta": theta.tolist(),
        "by_family": {n: 0.5 * float((z[i0:i1] ** 2).sum())
                      for n, i0, i1 in fam},
    }, indent=2), encoding="utf-8")
    print(f"[truth-phi] written {out}", flush=True)


if __name__ == "__main__":
    main()
