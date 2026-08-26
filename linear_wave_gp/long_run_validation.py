"""Long-run validation of a GP-closure fit against the reference: the scalar
summary behind thesis Table 4.1's long-run row (Var(eta) 1.109 vs 0.995,
H_s 4.213 vs 3.989 m, excess kurtosis +0.362 vs -0.028; 40 members).

Simulates the fitted closure and the reference over a window and compares what
the model was NOT fitted to: total variance, significant wave height
H_s = 4 sd(eta), skewness and excess kurtosis of the pooled elevation field.

    python long_run_validation.py results/gp_T1000_NG100_9nodes_resample
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

T_LONG = float(os.environ.get("LONG_T", "1000"))   # default: the fitting window T_y
N_LONG = int(round(T_LONG / st.DT_DATA))
N_MEMBERS = 40
N_REF = 40


def _model_job(args):
    theta, seed = args
    eta, v, failed = gp.simulate_batch(theta[None, :],
                                       rng=np.random.default_rng(seed),
                                       t_record=T_LONG)
    if failed[0]:
        return None
    return eta[0], v[0]


def _ref_job(seed):
    return st.deterministic_fields(N_LONG, seed)


def main():
    directory = Path(sys.argv[1] if len(sys.argv) > 1
                     else "results/gp_T1000_NG100_9nodes_resample")
    bundle = np.load(directory / "bundle.npz")
    theta_mean = bundle["theta_mean"]
    ensemble = bundle["final_ensemble_physical"][:N_MEMBERS]

    workers = int(os.environ.get("LONG_WORKERS",
                                 max(1, (os.cpu_count() or 4) - 4)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        refs = list(pool.map(_ref_job, [90000 + i for i in range(N_REF)],
                             chunksize=1))
        models = list(pool.map(
            _model_job, [(ensemble[j], 91000 + j) for j in range(len(ensemble))],
            chunksize=1))
    models = [m for m in models if m is not None]
    mean_run = _model_job((theta_mean, 92000))
    print(f"{len(models)}/{len(ensemble)} ensemble members simulated; "
          f"reported mean {'ok' if mean_run else 'FAILED'}", flush=True)

    ref_eta = np.concatenate([r[0] for r in refs], axis=0)
    ref_v = np.concatenate([r[1] for r in refs], axis=0)
    mod_eta = np.concatenate([m[0] for m in models], axis=0)
    mod_v = np.concatenate([m[1] for m in models], axis=0)

    out = {
        "T_long_s": T_LONG, "members": len(models),
        "reference": {"var_eta": float(ref_eta.var()),
                      "hs": float(4 * ref_eta.std()),
                      "var_v": float(ref_v.var()),
                      "skew": float(sps.skew(ref_eta.ravel())),
                      "excess_kurtosis": float(sps.kurtosis(ref_eta.ravel()))},
        "model": {"var_eta": float(mod_eta.var()),
                  "hs": float(4 * mod_eta.std()),
                  "var_v": float(mod_v.var()),
                  "skew": float(sps.skew(mod_eta.ravel())),
                  "excess_kurtosis": float(sps.kurtosis(mod_eta.ravel()))},
    }
    target = HERE / f"longrun_{directory.name}.json"
    target.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
