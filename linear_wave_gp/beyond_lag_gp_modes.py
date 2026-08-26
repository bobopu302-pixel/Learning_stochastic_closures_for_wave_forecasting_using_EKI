"""Beyond-the-fitted-lags ACF of the GP closure at different mode counts --
the GP-side data of thesis Figure 4.3(c) / discussion point C4.8: where does
the recurrence beyond the fitted lags come from, the closure or the mode count?

Chapter 4 fits the two GP functions on a grid of M = 10 modes.  Its
autocorrelation is right out to the lags the objective sees (0.8 s) and then
recurs at 6, 12, 18 s, which the 10-mode spacing df = 1.5/(M-1) = 0.167 Hz
predicts: a beat period of 1/df = 6.0 s.

This script holds the *learned closure fixed* -- the same Phi_q, Phi_p, the
same 40 ensemble members, the same reference field, the same seeds and the same
estimator -- and changes only the mode grid.  The noise block is carried onto
the finer grid by interpolating sigma(f) and rescaling by one scalar so the
long-run elevation variance is unchanged, because a mode on a finer grid stands
for a narrower band.  Nothing is re-calibrated: this is a statement about where
the recurrence comes from, not a claim that a 25-mode model fits better.

M is a module constant of modal_closure.experiment, fixed by the environment at
import, so one invocation handles one M:

    MODAL_CLOSURE_M_MODES=10 python beyond_lag_gp_modes.py acf_M10.npz
    MODAL_CLOSURE_M_MODES=25 python beyond_lag_gp_modes.py acf_M25.npz
                             python beyond_lag_gp_modes.py acf_ref.npz --reference
"""
from __future__ import annotations

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

RUN = Path(os.environ.get("BUNDLE_DIR",
                          str(HERE / "results" / "gp_T1000_NG100_9nodes_resample")))
T_LONG = 1000.0                       # same window as the fit and as Figure 4.3
N_LONG = int(round(T_LONG / st.DT_DATA))
N_MEMBERS = 40                        # the members Figure 4.3 already uses
N_REF = 40
N_LAG = 400                           # 400 * 0.05 s = 20 s, the range of Figure 4.3(c)
M_FIT = 10                            # the grid the closure was fitted on
BLOCK = gp.BLOCK                      # 12 = 9 nodes + (tau, a, ell)


def acf(x, n_lag=N_LAG):
    x = x - x.mean()
    return np.correlate(x, x, "full")[len(x) - 1:][:n_lag] / (x @ x)


def regrid(theta_fit, m_new, sigma_scale=1.0):
    """Carry a theta fitted on M_FIT modes onto m_new modes, closure untouched."""
    if m_new == M_FIT:
        return np.array(theta_fit, dtype=float)
    f_old = np.linspace(0.3, 1.8, M_FIT)
    f_new = np.linspace(0.3, 1.8, m_new)
    sig_old = np.asarray(theta_fit[2 * BLOCK:2 * BLOCK + M_FIT], dtype=float) ** 2
    sig_new = np.interp(f_new, f_old, sig_old) * (M_FIT - 1) / (m_new - 1) * sigma_scale
    return np.concatenate([np.asarray(theta_fit[:2 * BLOCK], dtype=float), np.sqrt(sig_new)])


def _model_job(args):
    theta, seed = args
    eta, _v, failed = gp.simulate_batch(theta[None, :], rng=np.random.default_rng(seed),
                                        t_record=T_LONG)
    return None if failed[0] else eta[0]


def _ref_job(seed):
    return st.deterministic_fields(N_LONG, seed)[0]


def main():
    out = sys.argv[1]
    reference = "--reference" in sys.argv
    workers = int(os.environ.get("LONG_WORKERS", max(1, (os.cpu_count() or 4) - 4)))

    if reference:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            refs = list(pool.map(_ref_job, [90000 + i for i in range(N_REF)], chunksize=1))
        # Figure 4.3(c) takes the reference curve from one record at one gauge; averaging over
        # all 40 records and all 10 gauges is the same estimator with a smaller error bar, which
        # matters here because the whole question is the size of a small long-lag wiggle.
        curves = np.array([acf(r[:, k]) for r in refs for k in range(r.shape[1])])
        per_rec = np.array([np.mean([acf(r[:, k]) for k in range(r.shape[1])], axis=0)
                            for r in refs])
        lo, hi = np.percentile(per_rec, [5, 95], axis=0)
        np.savez(out, acf=curves.mean(axis=0), lags=np.arange(N_LAG) * st.DT_DATA,
                 acf_lo=lo, acf_hi=hi,
                 n_curves=len(curves), var=np.mean([r.var() for r in refs]),
                 m_modes=0, t_long=T_LONG)
        print("reference: %d curves, var %.4f" % (len(curves), np.mean([r.var() for r in refs])))
        return

    m = gp.M_MODES
    ens = np.load(RUN / "bundle.npz")["final_ensemble_physical"][:N_MEMBERS]
    scale = float(os.environ.get("SIGMA_SCALE", "1"))
    # A bundle fitted on this very grid is used as it stands.  regrid() is only for carrying a
    # closure fitted elsewhere onto a different grid, and that transplant is not well defined for
    # a nonlinear closure -- the functions get evaluated at amplitudes they were never fitted at,
    # so the recurrence amplitude depends on how the noise is renormalised.  Only its period,
    # which is the beat of the mode grid, survives the transplant unchanged.
    if ens.shape[1] == gp.N_THETA:
        thetas = np.array(ens, dtype=float)
        print("  bundle already on M = %d; theta used unchanged" % m, flush=True)
    else:
        thetas = np.array([regrid(t, m, scale) for t in ens])
    print("M = %d, theta dim %d -> %d, %d members x %.0f s, sigma scale %.4f"
          % (m, ens.shape[1], thetas.shape[1], N_MEMBERS, T_LONG, scale), flush=True)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        etas = [e for e in pool.map(_model_job, [(thetas[j], 91000 + j)
                                                 for j in range(len(thetas))], chunksize=1)
                if e is not None]
    print("  %d/%d members simulated" % (len(etas), len(thetas)), flush=True)

    curves = np.array([acf(e[:, k]) for e in etas for k in range(e.shape[1])])
    per_mem = np.array([np.mean([acf(e[:, k]) for k in range(e.shape[1])], axis=0)
                        for e in etas])
    lo, hi = np.percentile(per_mem, [5, 95], axis=0)
    var = float(np.mean([e.var() for e in etas]))
    np.savez(out, acf=curves.mean(axis=0), lags=np.arange(N_LAG) * st.DT_DATA,
             acf_lo=lo, acf_hi=hi,
             n_curves=len(curves), var=var, m_modes=m, t_long=T_LONG,
             n_members=len(etas), sigma_scale=scale)
    lags = np.arange(N_LAG) * st.DT_DATA
    a = curves.mean(axis=0)
    beyond = lags > 3.0
    print("  var %.4f | beat period 1/df = %.1f s | max|acf| beyond 3 s = %.3f at %.1f s"
          % (var, (m - 1) / 1.5,
             np.abs(a[beyond]).max(), lags[beyond][np.abs(a[beyond]).argmax()]))
    print("  wrote %s" % out)


if __name__ == "__main__":
    main()
