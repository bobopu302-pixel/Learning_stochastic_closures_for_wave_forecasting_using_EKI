"""Persist the long-run validation ARRAYS behind thesis Figure 4.3(a,b):
invariant-measure and tail densities, and the eta autocorrelation, saved as
``longrun_arrays.npz`` next to the run's bundle.

The long-run validation originally lived in a figure script that drew 40 fitted
members against 40 reference records and threw the curves away, keeping only
scalars (see long_run_validation.py in this folder).  This script runs the same
simulation with the same seeds and the same estimators -- Gaussian KDE on the
same subsampled series, the same grids, the same autocorrelation -- and saves
the curves, so the thesis figure is reproducible without re-simulating.

    python dump_longrun_arrays.py results/gp_T1000_NG100_9nodes_resample
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("STOCHASTIC_N", "10000")

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # the code_rp root
for _p in (str(HERE), str(ROOT), str(ROOT / "linear_wave")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import gp_closure as gp                                      # noqa: E402
import stochastic_truth as st                                # noqa: E402
from scipy import stats as sps                               # noqa: E402

T_LONG = float(os.environ.get("LONG_T", "1000"))
N_LONG = int(round(T_LONG / st.DT_DATA))
N_MEMBERS = 40
N_REF = 40
N_LAG = 400                                                  # 400 * 0.05 s = 20 s


def _model_job(args):
    theta, seed = args
    eta, v, failed = gp.simulate_batch(theta[None, :], rng=np.random.default_rng(seed),
                                       t_record=T_LONG)
    return None if failed[0] else (eta[0], v[0])


def _ref_job(seed):
    return st.deterministic_fields(N_LONG, seed)


def acf(x, n_lag=N_LAG):
    x = x - x.mean()
    return np.correlate(x, x, "full")[len(x) - 1:][:n_lag] / (x @ x)


def main():
    directory = Path(sys.argv[1] if len(sys.argv) > 1
                     else "results/gp_T1000_NG100_9nodes_resample")
    bundle = np.load(directory / "bundle.npz")
    ensemble = bundle["final_ensemble_physical"][:N_MEMBERS]

    workers = int(os.environ.get("LONG_WORKERS", max(1, (os.cpu_count() or 4) - 4)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        refs = list(pool.map(_ref_job, [90000 + i for i in range(N_REF)], chunksize=1))
        models = [m for m in pool.map(
            _model_job, [(ensemble[j], 91000 + j) for j in range(len(ensemble))],
            chunksize=1) if m is not None]
    print("%d/%d members simulated" % (len(models), len(ensemble)), flush=True)

    ref_eta = np.concatenate([r[0] for r in refs], axis=0)
    mod_eta = np.concatenate([m[0] for m in models], axis=0)

    # --- invariant measure of eta, on the grid the thesis figure uses -------
    span = 4.2 * ref_eta.std()
    grid = np.linspace(-span, span, 400)
    ref_pdf = sps.gaussian_kde(ref_eta.ravel()[::37])(grid)
    mod_pdf = sps.gaussian_kde(mod_eta.ravel()[::37])(grid)
    per_member = np.array([sps.gaussian_kde(m[0].ravel()[::17])(grid) for m in models])
    lo, hi = np.percentile(per_member, [5, 95], axis=0)

    # --- the same on a wider grid, for the tails ---------------------------
    span_t = 5.5 * ref_eta.std()
    grid_t = np.linspace(-span_t, span_t, 400)
    ref_tail = sps.gaussian_kde(ref_eta.ravel()[::37])(grid_t)
    mod_tail = sps.gaussian_kde(mod_eta.ravel()[::37])(grid_t)
    gauss_tail = sps.norm.pdf(grid_t, 0.0, ref_eta.std())

    # --- autocorrelation of eta at the first gauge -------------------------
    lags = np.arange(N_LAG) * st.DT_DATA
    ref_acf = acf(refs[0][0][:, 0])
    mem_acf = np.array([acf(m[0][:, 0]) for m in models])
    acf_lo, acf_hi = np.percentile(mem_acf, [5, 95], axis=0)

    out = directory / "longrun_arrays.npz"
    np.savez(out, grid=grid, ref_pdf=ref_pdf, mod_pdf=mod_pdf, mod_lo=lo, mod_hi=hi,
             grid_tail=grid_t, ref_tail=ref_tail, mod_tail=mod_tail, gauss_tail=gauss_tail,
             lags=lags, ref_acf=ref_acf, mod_acf_mean=mem_acf.mean(axis=0),
             mod_acf_lo=acf_lo, mod_acf_hi=acf_hi,
             ref_var=ref_eta.var(), mod_var=mod_eta.var(),
             ref_kurt=sps.kurtosis(ref_eta.ravel()), mod_kurt=sps.kurtosis(mod_eta.ravel()),
             t_long=T_LONG, n_members=len(models), n_ref=N_REF)
    print("written %s" % out)
    print("  ref var %.4f  model var %.4f | ref ex.kurt %+.4f  model %+.4f"
          % (ref_eta.var(), mod_eta.var(), sps.kurtosis(ref_eta.ravel()),
             sps.kurtosis(mod_eta.ravel())))


if __name__ == "__main__":
    main()
