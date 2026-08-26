"""Noisy Lorenz 63 under EKI_algorithm_spec_2026-08-23.md -- PARALLEL driver.

Four fits, exactly as in the paper: fixed g_L (ODE / SDE) and GP g_L
(ODE / SDE).  Spec conventions:

  y        : ONE reference record of length T_y after T_burn (seed S0+1000)
  Gamma    : the FULL sample covariance of N_Gamma independent reference records
  N_G      : fixed at N_G_FIXED for every fit; the near-optimum probe is still
             run and the realised omitted fraction max(ratio)/N_G is checked
             against delta
  G(theta) : mean of N_G forward runs of length T_G = T_y, common random numbers
  EKI      : spec section 2, stop on the relative change of Phi

The full sample-covariance Gamma is the only scheme kept;

    python run_spec.py --workers 190        (run from this folder)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

# Import bootstrap: the case folder itself (for simulator.py under Windows
# spawn workers) and the code_rp root (for the algorithms package).
CASE_DIR = Path(__file__).resolve().parent
for _path in (str(CASE_DIR), str(CASE_DIR.parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from algorithms import eki
from algorithms.gamma import build_gamma, calibrate_n_g
from algorithms.gpr import make_gp_mean_from_theta
from algorithms.parameterization import (
    decode_positive_columns,
    encode_positive_columns,
    gp_positive_indices,
)
from algorithms.statistics import centered_first_second_moments
from simulator import simulate_lorenz63

# ---------------------------------------------------------------- spec parameters
T_Y = 1500.0
T_G = T_Y
N_GAMMA = 200
T_BURN = 20.0
S0 = 7
J_ENS = 100
N_ITER = 30
PHI_TOL = 0.01
DELTA_NG = 0.10
K_PROBE = 40
# One uniform N_G for every fit (2026-08-24 decision).
N_G_FIXED = 20

DT = 0.002

TRUTH = dict(alpha=10.0, rho=28.0, beta=8.0 / 3.0, sigma=10.0)
NODES = np.linspace(-30.0, 30.0, 5)
FAMILIES = [list(range(0, 3)), list(range(3, 9))]          # means, second moments
XLIMS = [(-25.0, 25.0), (-25.0, 25.0), (-2.0, 52.0)]
EDGES = [np.linspace(lo, hi, 161) for lo, hi in XLIMS]
INVARIANT_T, INVARIANT_BURN = 5000.0, 100.0

FIT_NAMES = ("fixed_ode", "fixed_sde", "gp_ode", "gp_sde")

# Near-optimum probe parameters: the final-ensemble means of the archived
# 2026-08-23 paper run (1. Reproduce_papers/Lorenz63/result_data/
# 20260823T140932127263Z_paper_8ad6802b51/learned_parameters.npz), embedded so
# this release does not depend on that archive.  Raw (physical) coordinates;
# GP layout [node values (5) | nugget | amplitude | lengthscale (| sqrt(sigma))].
NEAR_OPTIMUM_RAW = {
    "fixed_ode": [10.405237346138554],
    "fixed_sde": [9.920225667086427, 3.2810541046120396],
    "gp_ode": [1.9519068770531893, -15.156204569085869, -0.016988449799467665,
               15.290435136308952, -5.321444081414367, 5.112715364526447,
               13.140992976001382, 8.223862844007465],
    "gp_sde": [4.297529492508175, -18.292970731880004, 0.3370906446086899,
               17.470040199869064, 2.685279249718155, 3.117518903631222,
               8.738707091857453, 8.77305048286084, 3.4544664478107343],
}


def _is_gp(fit: str) -> bool:
    return fit.startswith("gp")


def _is_sde(fit: str) -> bool:
    return fit.endswith("sde")


def positive_indices(fit: str) -> tuple[int, ...]:
    if fit == "fixed_ode":
        return (0,)
    if fit == "fixed_sde":
        return (0, 1)
    gp_pos = gp_positive_indices(NODES.size)
    return gp_pos if fit == "gp_ode" else (*gp_pos, NODES.size + 3)


def simulate(theta_raw: np.ndarray, fit: str, seed: int, t_total: float, burn: float) -> np.ndarray:
    theta_raw = np.asarray(theta_raw, dtype=float)
    if _is_gp(fit):
        g_func = make_gp_mean_from_theta(theta_raw[:-1] if _is_sde(fit) else theta_raw, NODES)
        alpha = TRUTH["alpha"]
        sigma = float(theta_raw[-1]) ** 2 if _is_sde(fit) else 0.0
    else:
        g_func = None
        alpha = float(theta_raw[0])
        sigma = float(theta_raw[1]) ** 2 if _is_sde(fit) else 0.0
    return simulate_lorenz63(
        alpha=alpha, rho=TRUTH["rho"], beta=TRUTH["beta"], sigma=sigma, dt=DT,
        t_total=t_total, burn_in=burn, rng=np.random.default_rng(int(seed)), g_func=g_func,
    )


# ---- picklable workers ----
# The window length travels INSIDE the task: with spawn (Windows) a worker
# re-imports this module and would otherwise use the module default rather than
# the value main() set, silently mixing window lengths.
def _w_truth(task) -> np.ndarray:
    seed, t_y = task
    trajectory = simulate_lorenz63(**TRUTH, dt=DT, t_total=float(t_y), burn_in=T_BURN,
                                   rng=np.random.default_rng(int(seed)))
    return centered_first_second_moments(trajectory)


def _w_forward(task) -> tuple[int, np.ndarray]:
    index, theta_raw, seed, fit, t_g = task
    return index, centered_first_second_moments(simulate(theta_raw, fit, seed, float(t_g), T_BURN))


def _w_density(task) -> np.ndarray:
    theta_raw, seed, fit, t_inv = task
    try:
        trajectory = simulate(theta_raw, fit, seed, float(t_inv), INVARIANT_BURN)
    except FloatingPointError:
        return np.zeros((3, EDGES[0].size - 1))
    return np.array([np.histogram(trajectory[:, c], bins=EDGES[c], density=True)[0] for c in range(3)])


def warm_up() -> None:
    """Compile the numba kernel in the parent before the pool starts."""

    _w_truth((S0, 1.0))
    for fit in FIT_NAMES:
        theta = prior_mean_raw(fit)
        simulate(theta, fit, S0, 1.0, 0.5)


def prior_mean_raw(fit: str) -> np.ndarray:
    # Means of the paper's uniform initial ranges: alpha ~ U(1,20),
    # sqrt(sigma) ~ U(0.1,15); GP block node values ~ U(-20,20) (mean 0),
    # nugget/amplitude ~ U(0.1,10), lengthscale ~ U(5,10).
    if fit == "fixed_ode":
        return np.array([10.5])
    if fit == "fixed_sde":
        return np.array([10.5, 7.55])
    block = np.concatenate([np.zeros(NODES.size), [5.05, 5.05, 7.5]])
    return block if fit == "gp_ode" else np.concatenate([block, [7.55]])


def near_optimum_raw(fit: str) -> np.ndarray:
    return np.asarray(NEAR_OPTIMUM_RAW[fit], dtype=float)


def draw_initial(fit: str, rng: np.random.Generator, j_ens: int) -> np.ndarray:
    if fit == "fixed_ode":
        return rng.uniform(1.0, 20.0, size=(j_ens, 1))
    if fit == "fixed_sde":
        return np.column_stack([rng.uniform(1.0, 20.0, j_ens), rng.uniform(0.1, 15.0, j_ens)])
    block = np.hstack([
        rng.uniform(-20.0, 20.0, (j_ens, NODES.size)),
        rng.uniform(0.1, 10.0, (j_ens, 1)),
        rng.uniform(0.1, 10.0, (j_ens, 1)),
        rng.uniform(5.0, 10.0, (j_ens, 1)),
    ])
    return block if fit == "gp_ode" else np.hstack([block, rng.uniform(0.1, 15.0, (j_ens, 1))])


def main(workers: int, output_dir: Path | None, quick: bool, only: str | None) -> Path:
    global T_Y, T_G, N_GAMMA, J_ENS, N_ITER, K_PROBE, INVARIANT_T, N_G_FIXED
    if quick:
        T_Y = T_G = 20.0
        N_GAMMA, J_ENS, N_ITER, K_PROBE, INVARIANT_T = 30, 8, 3, 6, 50.0   # N_Gamma > q = 9
        N_G_FIXED = None                                                   # tiny windows: calibrate instead
    started = time.perf_counter()
    out_root = Path(output_dir) if output_dir else (CASE_DIR / "spec_results")
    out_root.mkdir(parents=True, exist_ok=True)
    warm_up()
    fits = [only] if only else list(FIT_NAMES)
    print(f"[L63 spec] T_y = T_G = {T_Y:g}, burn {T_BURN:g}, N_Gamma = {N_GAMMA}, J = {J_ENS}, workers = {workers}", flush=True)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        # Observation y: one reference record, seed block disjoint from Gamma's.
        y = _w_truth((S0 + 1000, T_Y))
        t0 = time.perf_counter()
        records = np.array(list(pool.map(_w_truth, [(S0 + 2000 + r, T_Y) for r in range(N_GAMMA)], chunksize=1)))
        print(f"  {N_GAMMA} reference records in {time.perf_counter()-t0:.0f} s", flush=True)

        structures = ["full"]           # L63 keeps the full covariance only
        gammas = {}
        for structure in structures:
            try:
                gammas[structure] = build_gamma(records, structure=structure, families=FAMILIES)
            except ValueError as exc:
                print(f"  gamma_structure={structure} rejected: {exc}", flush=True)
        for structure, est in gammas.items():
            print(f"  Gamma[{structure}]: rank {est.rank}/{len(y)}, cond {est.condition_number:.3g}, "
                  f"min corr eig {est.corr_min_eigenvalue:.2e}", flush=True)
        if not gammas:
            raise RuntimeError("no usable Gamma: the full sample covariance is rank deficient, "
                               f"raise N_Gamma (now {N_GAMMA}) above the statistic count")
        var_ref = next(iter(gammas.values())).var_ref

        summary = {
            "case": "lorenz63", "spec": "EKI_algorithm_spec_2026-08-23.md", "date": "2026-08-23",
            "T_y": T_Y, "T_G": T_G, "N_Gamma": N_GAMMA, "T_burn": T_BURN, "S0": S0, "J": J_ENS,
            "n_iter": N_ITER, "phi_tol": PHI_TOL, "delta_N_G": DELTA_NG, "K_probe": K_PROBE,
            "y_convention": "single_window", "forward_convention": "mean_of_N_G",
            "gamma_terms": "var_ref_only", "parameter_choice": "final_ensemble_mean",
            "engine": "algorithms.eki.run_eki (jitter_mode=absolute, stop_patience=1)",
            "gamma_diagnostics": {s: dict(rank=e.rank, condition_number=e.condition_number,
                                          corr_min_eigenvalue=e.corr_min_eigenvalue) for s, e in gammas.items()},
            "fits": {},
        }

        for fit in fits:
            positive = positive_indices(fit)

            # N_G check: probe the forward-sampling variance at the prior mean
            # and near the optimum; the near-optimum probe is decisive.
            probe_stats = {}
            for label, theta_raw in (("prior_mean", prior_mean_raw(fit)), ("near_optimum", near_optimum_raw(fit))):
                tasks = [(k, theta_raw, S0 + 7000 + 1000 * FIT_NAMES.index(fit) + k, fit, T_G) for k in range(K_PROBE)]
                got = sorted(pool.map(_w_forward, tasks, chunksize=1), key=lambda t: t[0])
                probe_stats[label] = np.array([g for _, g in got])
            cal = calibrate_n_g(probe_stats, var_ref, delta=DELTA_NG, decisive_probe="near_optimum")
            n_g = N_G_FIXED or cal.n_g          # None (--quick) -> use the calibration
            omitted = cal.ratio_max / n_g
            if N_G_FIXED and omitted > DELTA_NG:
                raise RuntimeError(
                    f"N_G_FIXED = {n_g} leaves {omitted:.1%} of the misfit variance out of "
                    f"Gamma (> delta = {DELTA_NG:.0%}); the calibration asks for {cal.n_g}")
            print(f"  [{fit}] N_G = {n_g} (calibration asks {cal.n_g}, omitted {omitted:.1%})"
                  f"  (near-optimum ratio min/med/max = "
                  f"{cal.ratio_min:.3f}/{cal.ratio_median:.3f}/{cal.ratio_max:.3f}; "
                  f"prior-mean implies {cal.per_probe['prior_mean']['n_g_implied']})", flush=True)

            clip_count = {"n": 0}
            base_seed = S0 + 500_000 + 1000 * FIT_NAMES.index(fit)
            seeds_log: list[np.ndarray] = []

            # Spec evaluation semantics: one ensemble evaluation = N_G forward
            # runs per member, common random numbers (the same CRN seed set for
            # every member within an iteration, a fresh set each iteration),
            # averaged over the realisation axis; runs distributed over the
            # shared process pool exactly as in the source driver.
            def evaluate(theta_latent: np.ndarray, iteration: int,
                         _fit=fit, _pos=positive, _n_g=n_g, _base=base_seed,
                         _log=seeds_log) -> np.ndarray:
                seed_set = eki.crn_seeds(_base, int(iteration), _n_g)
                _log.append(seed_set.copy())
                raw = decode_positive_columns(theta_latent, _pos)
                tasks = [(j * len(seed_set) + a, raw[j], int(s), _fit, T_G)
                         for j in range(raw.shape[0]) for a, s in enumerate(seed_set)]
                got = sorted(pool.map(_w_forward, tasks, chunksize=8), key=lambda t: t[0])
                stacked = np.array([g for _, g in got]).reshape(raw.shape[0], len(seed_set), -1)
                return stacked.mean(axis=1)

            # Latent guard as a post_update hook: clip the ensemble into the
            # safe exp() range after every Kalman update, counting occurrences
            # (a healthy run clips nothing).
            def clip_hook(thetas: np.ndarray, iteration: int, _pos=positive) -> np.ndarray:
                safe, clipped = eki.clip_latent(thetas, _pos)
                clip_count["n"] += clipped
                return safe

            for structure, est in gammas.items():
                seeds_log.clear()      # one seed log per EKI run
                rng = np.random.default_rng(S0 + 100 * FIT_NAMES.index(fit))
                init_raw = draw_initial(fit, rng, J_ENS)
                t1 = time.perf_counter()
                print(f"  [{fit} | {structure}] EKI ...", flush=True)
                res = eki.run_eki(
                    encode_positive_columns(init_raw, positive), None, y, est.matrix,
                    n_iter=N_ITER, rng=rng, perturb_observations=True,
                    ensemble_evaluator=evaluate, post_update=clip_hook,
                    stop_rel_tol=PHI_TOL, stop_patience=1,
                    jitter=1e-8, jitter_mode="absolute", verbose=True,
                )
                stopped_early = res.stop_reason != "n_iter"
                stopped_on = "phi_tol" if stopped_early else "n_iter"
                # Evaluations before the update loop ended; when the run
                # exhausts n_iter, run_eki appends one extra evaluation of the
                # final ensemble, excluded here to match the legacy count.
                iterations_run = res.n_updates + 1 if stopped_early else res.n_updates
                final_raw = decode_positive_columns(eki.clip_latent(res.theta_history[-1], positive)[0], positive)
                theta_mean, theta_sd = final_raw.mean(axis=0), final_raw.std(axis=0, ddof=1)
                dens_tasks = [(final_raw[m], S0 + 20000 + m, fit, INVARIANT_T) for m in range(final_raw.shape[0])]
                member_dens = np.array(list(pool.map(_w_density, dens_tasks, chunksize=1)))
                mean_dens = _w_density((theta_mean, S0 + 30000, fit, INVARIANT_T))
                singles = np.array([g for _, g in sorted(
                    pool.map(_w_forward, [(k, theta_mean, S0 + 60000 + k, fit, T_G) for k in range(8)], chunksize=1),
                    key=lambda t: t[0])])
                key = f"{fit}_{structure}"
                np.savez(
                    out_root / f"spec_{key}.npz", y=y, records=records, gamma=est.matrix, var_ref=var_ref,
                    theta_history=res.theta_history, phi_history=res.objective_history,
                    phi_member_history=res.objective_member_history,
                    seeds_history=np.asarray(seeds_log),
                    g_history=res.output_history, final_ensemble_raw=final_raw, theta_mean=theta_mean,
                    theta_sd=theta_sd, member_dens=member_dens, mean_dens=mean_dens,
                    single_forward_stats=singles, ratio=cal.ratio, nodes=NODES,
                    inv_edges=np.array(EDGES),
                )
                entry = {
                    "gamma_structure": structure, "N_G": n_g, "N_G_calibrated": cal.n_g,
                    "omitted_variance_fraction": float(omitted),
                    "ratio": {"min": cal.ratio_min, "median": cal.ratio_median, "max": cal.ratio_max,
                              "per_probe": cal.per_probe},
                    "iterations_run": iterations_run, "stopped_on": stopped_on,
                    "stop_reason": res.stop_reason,
                    "phi_first": float(res.objective_history[0]),
                    "phi_final": float(res.objective_history[-1]),
                    "phi_over_q_half": float(res.objective_history[-1] / (0.5 * len(y))),
                    "theta_mean": theta_mean.tolist(), "theta_sd": theta_sd.tolist(),
                    "latent_clips": int(clip_count["n"]),
                    "runtime_seconds": time.perf_counter() - t1,
                }
                if not _is_gp(fit):
                    entry["alpha"] = float(theta_mean[0]); entry["alpha_sd"] = float(theta_sd[0])
                if _is_sde(fit):
                    entry["sigma"] = float(theta_mean[-1] ** 2)
                    entry["sigma_sd"] = float(np.std(final_raw[:, -1] ** 2, ddof=1))
                summary["fits"][key] = entry
                msg = (f"  [{key}] {iterations_run} iters ({stopped_on}), "
                       f"Phi {res.objective_history[0]:.1f} -> "
                       f"{res.objective_history[-1]:.3f} (q/2 = {0.5*len(y):.1f})")
                if "alpha" in entry: msg += f", alpha = {entry['alpha']:.3f}"
                if "sigma" in entry: msg += f", sigma = {entry['sigma']:.3f}"
                print(msg + f", {time.perf_counter()-t1:.0f} s", flush=True)

    summary["total_runtime_seconds"] = time.perf_counter() - started
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"total {summary['total_runtime_seconds']:.0f} s -> {out_root}", flush=True)
    return out_root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--only", choices=FIT_NAMES)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    main(args.workers, args.output_dir, args.quick, args.only)
