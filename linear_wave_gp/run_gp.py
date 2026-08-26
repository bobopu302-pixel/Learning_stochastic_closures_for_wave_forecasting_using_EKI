"""EKI driver for the GP wave closure, against the frozen N = 10000 reference
-- the driver of the thesis Chapter 4 main run.

Runs either model on identical data so the comparison is attributable to the
closure alone:

    gp     dq = [p + Phi_q(q)] dt,  dp = [-w^2 q + Phi_p(p)] dt + sqrt(s) dW
    delta  dq = p dt,               dp = [-w^2 q - delta p] dt + sqrt(s) dW

Everything else follows the 2026-08-23 algorithm spec and the settings of the
diagonal-Gamma, N_G = 100 run: T_y = T_G = 1000 s, N_Gamma = 200,
Gamma = diag(var_ref) with no forward term and no floor, J = 100, N_G = 100
forward runs per member under common random numbers, 30 iterations, stop on the
relative change of Phi.

    python run_gp.py --model gp --workers 188 --n-g 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
# The reference is frozen at 10 000 components; stochastic_truth reads this at
# import, and a spawned worker re-imports, so it must be in the environment
# before anything imports it -- setting the module attribute would not survive.
os.environ.setdefault("STOCHASTIC_N", "10000")

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # the code_rp root
for _p in (str(HERE), str(ROOT), str(ROOT / "linear_wave")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import gp_closure as gp                                      # noqa: E402
import stochastic_truth as st                                # noqa: E402
from modal_closure import experiment as case                 # noqa: E402
from eki_resampling import run_eki                           # noqa: E402

# --- spec settings, matching the diagonal-Gamma N_G = 100 run ---------------
T_Y = 1000.0
N_DATA = int(round(T_Y / st.DT_DATA))
N_GAMMA = 200
N_G = 100
J_ENS = 100
N_ITER = int(os.environ.get("GP_N_ITER", "30"))
STOP_REL_TOL = 0.01
STOP_PATIENCE = 3
CALIB_K = 20

OBS_SEED = 22001
GAMMA_SEED = 30000
CRN_SEED = 40000
OBS_PERTURB_SEED = 50000
INIT_SEED = 60000
SINGLE_SEED = 70000
CALIB_SEED = 80000
SENTINEL = case.FORWARD_SENTINEL


# --- reference side ---------------------------------------------------------
def _reference_job(seed):
    eta, v = st.deterministic_fields(N_DATA, seed)
    return case.spatial_statistics(eta, v, case.AUTO_LAGS, case.CROSS_LAGS)


# --- forward side -----------------------------------------------------------
# Members evaluated together in one job.  Batching amortises the per-step numpy
# overhead, which profiling showed to be the whole cost -- the state is ten
# numbers, so nothing in a step is real arithmetic.  Measured 7.7x per member at
# 16.  It is only legitimate because the spec's common random numbers mean every
# member of an iteration is driven by the SAME noise, so one draw serves the
# batch; with independent streams per member this would change the result.
BATCH = 25


def _forward_job(job):
    """G(theta) for a batch of members under one shared noise realisation.

    The model name and the window travel inside the job: a spawned worker
    re-imports this module and would otherwise use whatever the committed
    defaults are.
    """

    theta_latents, seed_key, model, t_record = job
    theta_latents = np.atleast_2d(np.asarray(theta_latents, dtype=float))
    rng = np.random.default_rng(np.random.SeedSequence(list(seed_key)))
    try:
        if model == "gp":
            return gp.statistics_batch(gp.from_latent(theta_latents),
                                       rng=rng, t_record=t_record)
        rows = np.empty((theta_latents.shape[0], 38))
        for b, latent in enumerate(theta_latents):
            eta, v = case.simulate_grid(*case.unpack(np.exp(latent)), rng=rng,
                                        t_record=t_record)
            stats = case.spatial_statistics(eta, v, case.AUTO_LAGS,
                                            case.CROSS_LAGS)
            rows[b] = stats if np.all(np.isfinite(stats)) else SENTINEL
        return rows
    except (FloatingPointError, ValueError, np.linalg.LinAlgError):
        return np.full((theta_latents.shape[0], 38), SENTINEL)


def _mapper(pool):
    if pool is None:
        return lambda fn, items: [fn(i) for i in items]

    def mapped(fn, items):
        items = list(items)
        chunk = max(1, len(items) // (4 * pool._max_workers))
        return list(pool.map(fn, items, chunksize=chunk))

    return mapped


def evaluator(map_fn, model, t_record, n_g, batch=BATCH):
    """Ensemble evaluation with common random numbers, as the spec requires.

    One job carries (a slice of members, one replicate).  The seed key omits the
    member index -- that omission IS the common random numbers -- so members in
    different slices still share the replicate's noise.
    """

    def evaluate(theta_ensemble, iteration):
        members = theta_ensemble.shape[0]
        slices = [slice(s, min(s + batch, members))
                  for s in range(0, members, batch)]
        jobs = [(theta_ensemble[sl], (CRN_SEED, iteration, a), model, t_record)
                for a in range(n_g) for sl in slices]
        raw = list(map_fn(_forward_job, jobs))

        out = np.empty((members, n_g, 38))
        for index, (a, sl) in enumerate(
                ((a, sl) for a in range(n_g) for sl in slices)):
            out[sl, a] = raw[index]
        failed = np.all(out == SENTINEL, axis=2)
        averaged = out.mean(axis=1)
        averaged[failed.any(axis=1)] = SENTINEL
        return averaged

    return evaluate


def main():
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("gp", "delta"), default="gp")
    parser.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 4) - 4))
    parser.add_argument("--n-g", type=int, default=N_G)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out = args.out or (HERE / "results" / f"{args.model}_T{T_Y:.0f}_NG{args.n_g}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    pool = (ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context("spawn"))
        if args.workers > 1 else None)
    map_fn = _mapper(pool)

    print("=" * 74)
    print(f"model = {args.model}   reference = deterministic, "
          f"N = {st.N_COMPONENTS}   T_y = T_G = {T_Y:.0f} s")
    print(f"N_Gamma = {N_GAMMA}   N_G = {args.n_g}   J = {J_ENS}   "
          f"n_iter = {N_ITER}   workers = {args.workers}")
    print("=" * 74, flush=True)

    y = _reference_job(OBS_SEED)
    reference = np.asarray(map_fn(_reference_job,
                                  [GAMMA_SEED + i for i in range(N_GAMMA)]))
    var_ref = reference.var(axis=0, ddof=1)
    if np.any(var_ref <= 0.0):
        raise ValueError("var_ref vanished; Gamma would be singular")
    print(f"[Gamma] diag(var_ref) only: min {var_ref.min():.3e}  "
          f"median {np.median(var_ref):.3e}  max {var_ref.max():.3e}   "
          f"each known to {np.sqrt(2 / (N_GAMMA - 1)):.1%}   "
          f"[{time.perf_counter() - started:.0f} s]", flush=True)

    # N_G calibration, spec section 1.3: prior mean and a near-optimum probe.
    if args.model == "gp":
        probe = gp.to_latent(gp.prior_mean())
        n_theta = gp.N_THETA
        labels = gp.parameter_labels()
    else:
        # Release adaptation (see header): the origin's prior_mean_parameters()
        # / initial_ensemble(rng, log_coords=False) helpers existed only on the
        # run machine; the released modal_closure exposes the same prior in log
        # coordinates directly.
        probe = case.log_prior_mean_parameters()
        n_theta = 2 * case.M_MODES
        labels = ([f"delta_{i + 1}" for i in range(case.M_MODES)]
                  + [f"sqrt_sigma_{i + 1}" for i in range(case.M_MODES)])

    probe_stats = np.vstack(map_fn(
        _forward_job, [(probe[None, :], (CALIB_SEED, 0, i), args.model, T_Y)
                       for i in range(CALIB_K)]))
    good = ~np.all(probe_stats == SENTINEL, axis=1)
    if good.sum() >= 2:
        ratio = probe_stats[good].var(axis=0, ddof=1) / var_ref
        covered = int((ratio <= 0.2 * args.n_g).sum())
        print(f"[N_G] ratio at the prior mean: p50 {np.median(ratio):.2f}  "
              f"p75 {np.quantile(ratio, 0.75):.2f}  max {ratio.max():.1f}   "
              f"-> N_G = {args.n_g} covers {covered}/38", flush=True)
    else:
        ratio = np.full(y.size, np.nan)
        print(f"[N_G] the prior-mean probe produced only {int(good.sum())} "
              "usable runs; the closure is not dissipative there", flush=True)

    rng_init = np.random.default_rng(INIT_SEED)
    initial = (gp.initial_ensemble(rng_init, J_ENS) if args.model == "gp"
               else case.initial_ensemble(rng_init))
    print(f"[EKI] {n_theta} parameters, {J_ENS} members "
          f"(ensemble rank {J_ENS - 1} vs {n_theta} unknowns)", flush=True)

    result = run_eki(
        initial, None, y, var_ref, n_iter=N_ITER,
        rng=np.random.default_rng(INIT_SEED + 1), bounds=None,
        perturb_observations=True, verbose=True, sentinel_value=SENTINEL,
        stop_rel_tol=STOP_REL_TOL, stop_patience=STOP_PATIENCE,
        observation_rng=np.random.default_rng(OBS_PERTURB_SEED),
        # The stationarity gate fails whole members by design, so a sentinel row
        # in C^GG is expected rather than exceptional; repair the ensemble
        # instead of letting the penalty vector set the Kalman gain.
        resample_failures=True,
        ensemble_evaluator=evaluator(map_fn, args.model, T_Y, args.n_g))

    final_latent = result.final_ensemble
    final_physical = (gp.from_latent(final_latent) if args.model == "gp"
                      else np.exp(final_latent))
    theta_mean = final_physical.mean(axis=0)
    theta_sd = final_physical.std(axis=0, ddof=1)
    q_half = 0.5 * y.size

    print(f"\n[fit] stopped after {result.n_updates} updates "
          f"({result.stop_reason}); Phi = {result.final_objective:.4g}  "
          f"= {result.final_objective / q_half:.1f} x q/2")
    print(f"      sentinel evaluations: {result.total_sentinels}, "
          f"of which resampled: {int(result.resampled_counts.sum())}")

    summary = {
        "model": args.model, "reference_components": st.N_COMPONENTS,
        "t_y_s": T_Y, "n_gamma": N_GAMMA, "n_g": args.n_g, "j_ens": J_ENS,
        "n_theta": int(n_theta), "n_iter_max": N_ITER,
        "updates_applied": int(result.n_updates),
        "stop_reason": result.stop_reason,
        "phi_history": [float(v) for v in result.objective_history],
        "phi_final": float(result.final_objective),
        "phi_over_q_half": float(result.final_objective / q_half),
        "sentinel_evaluations": int(result.total_sentinels),
        "resampled_members": int(result.resampled_counts.sum()),
        "resampled_per_iteration": [int(v) for v in result.resampled_counts],
        "parameter_labels": labels,
        "theta_mean": theta_mean.tolist(), "theta_sd": theta_sd.tolist(),
        "var_ref": var_ref.tolist(), "y": y.tolist(),
        "ratio_at_prior_mean": ratio.tolist(),
        "seconds": time.perf_counter() - started,
    }
    if args.model == "gp":
        trace, det, stationary = gp.linear_stability(theta_mean)
        summary["linear_trace"] = float(trace)
        summary["linear_det_min"] = float(det.min())
        summary["linearly_stationary"] = bool(stationary)
        print(f"      linearised closure: trace = {trace:+.4f}, "
              f"det min = {det.min():.2f}, stationary = {stationary}")

    (out / "summary.json").write_text(json.dumps(summary, indent=2),
                                      encoding="utf-8")
    np.savez(out / "bundle.npz", y=y, var_ref=var_ref,
             theta_history=result.theta_history,
             final_ensemble_latent=final_latent,
             final_ensemble_physical=final_physical,
             theta_mean=theta_mean, theta_sd=theta_sd,
             objective_history=result.objective_history,
             final_outputs=result.final_outputs)

    if pool is not None:
        pool.shutdown()
    print(f"\n[done in {time.perf_counter() - started:.0f} s] -> {out}")


if __name__ == "__main__":
    main()
