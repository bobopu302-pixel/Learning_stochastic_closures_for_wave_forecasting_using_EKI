"""Lorenz 96 case (a) EKI driver under EKI_algorithm_spec_2026-08-23 (parallel).

Origin: 1. Reproduce_papers/Lorenz96/code/spec_case_a.py
Changes vs origin:
- rewired to the shared algorithms package: eki_spec.run_eki_spec ->
  algorithms.eki.run_eki (ensemble_evaluator pattern, stop_rel_tol=PHI_TOL,
  stop_patience=1, perturb_observations=True, jitter_mode="absolute" with
  jitter=1e-8 -- the same gain solve C^GG + Gamma + 1e-8 I as the origin);
  eki_spec.build_gamma/calibrate_n_g/clip_latent/crn_seeds -> algorithms.gamma
  and algorithms.eki; parameterization -> algorithms.parameterization;
- the CRN seed set S(n, r) is now drawn inside the case evaluator via
  algorithms.eki.crn_seeds (identical arithmetic) and recorded case-side, so
  the saved seeds_history is unchanged in meaning;
- the Phi path is bit-identical to the origin loop: Phi is computed by
  Cholesky whitening with the exact Gamma (the eki_spec._phi construction, no
  regularisation -- the jitter conditions the Kalman-gain solve only), so
  reported Phi values reproduce the origin exactly given identical forward
  outputs;
- one engine-level difference vs the origin loop, documented: when the run
  exhausts n_iter, the shared engine evaluates the final post-update ensemble
  once more, so phi_history gains one entry and phi_final then refers to the
  REPORTED final ensemble (the origin left that ensemble unevaluated); on a
  phi_tol stop the histories match the origin one for one.  "iterations_run"
  therefore counts ensemble evaluations (equal to the origin's count on a
  phi_tol stop);
- the near-optimum N_G probe parameter is embedded as a constant (the frozen
  v12 production fit) instead of being loaded from the reproduction tree's
  result_data, so this release is self-contained;
- comments translated/tightened; spec constants, seeds, worker layout, saved
  arrays, and the summary schema unchanged.  No figures are produced anywhere
  in this release (the origin produced none either).

Spec configuration (EKI_algorithm_spec_2026-08-23.md):

  y        : ONE reference record of length T_y (seed S0+1000), after T_burn
  Gamma    : the FULL sample covariance of N_Gamma independent reference
             records -- no forward term, no floor
  N_G      : fixed at N_G_FIXED for every fit; the near-optimum probe is still
             run and the realised omitted fraction max(ratio)/N_G is checked
             against delta
  G(theta) : mean of N_G forward runs of length T_G = T_y, common random numbers
  EKI      : spec section 2, stop on the relative change of Phi
  report   : final-ensemble mean +/- sd (no selection)

The full sample-covariance Gamma is the only scheme kept for case (a)
(2026-08-23 decision); the diagonal variant and the earlier decoupled-window
schemes were removed at the source already.

Parallelism: a process pool over the J * N_G independent forward runs of each
iteration.  Every worker re-imports this module, so the numba kernels compile
once per process and hit the on-disk cache afterwards.

    python run_spec.py --workers 8            # from this folder
    python run_spec.py --quick --workers 4    # tiny end-to-end check
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

# Import bootstrap: make the code_rp root (shared `algorithms` package) and
# this case folder importable, both in the parent and in spawned workers.
_HERE = Path(__file__).resolve().parent
for _path in (str(_HERE.parents[0]), str(_HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from algorithms.eki import clip_latent, crn_seeds, run_eki  # noqa: E402
from algorithms.gamma import build_gamma, calibrate_n_g  # noqa: E402
from algorithms.parameterization import (  # noqa: E402
    decode_positive_columns,
    encode_positive_columns,
)
from observations import (  # noqa: E402
    case_a_statistics,
    initial_ensemble,
    positive_parameter_indices,
    unpack_theta_parts,
)
from simulator import (  # noqa: E402
    B_FAST,
    DT_FULL,
    F_FORCE,
    J_FAST,
    K_SLOW,
    STORE_FULL,
    simulate_closed_lorenz96_gp_fast,
    simulate_two_scale_lorenz96,
)

# ---------------------------------------------------------------- spec parameters
T_Y = 1500.0          # window of y and of every Gamma record
T_G = T_Y             # spec: T_G = T_y
N_GAMMA = 200         # reference records behind Gamma
T_BURN = 0.0          # case a discards nothing (paper's [0, T] convention)
S0 = 31               # base seed of the case
J_ENS = 100
N_ITER = 30
PHI_TOL = 0.01
DELTA_NG = 0.10       # spec: neglected var_fwd/N_G below 10 % of var_ref
K_PROBE = 40          # forward runs per probe in the N_G calibration
# One uniform N_G for every fit (2026-08-24 decision).  It is >= every value
# the per-fit calibration returns, so the omitted variance fraction
# max(rho)/N_G stays under DELTA_NG everywhere; the probe still runs and the
# realised fraction is checked below.
N_G_FIXED = 20


# Case-(a) settings, all paper-explicit (Sec. 4.2): 7 GP nodes on [-15, 15],
# length-scale prior U(5, 20), c = 10, h = 1, lambda = h^2 c / J fixed.
SETTINGS = {"node_count": 7, "length_low": 5.0, "length_high": 20.0,
            "learn_linear_coefficient": False, "c": 10.0, "h_slow": 1.0}
C_FAST, H_COUPLE = float(SETTINGS["c"]), float(SETTINGS["h_slow"])
LAMBDA = H_COUPLE * H_COUPLE * C_FAST / J_FAST          # fixed in case (a)
NODES = np.linspace(-15.0, 15.0, SETTINGS["node_count"])
FAMILIES = [list(range(0, 8)), list(range(8, 16)), list(range(16, 44))]  # means, variances, covariances
EDGES = np.linspace(-10.0, 15.0, 161)  # invariant-measure histogram bin edges

# Near-optimum probe parameters for the N_G calibration: the frozen v12
# production fit (run 20260823T162629322003Z_paper_a540c727a9,
# learned_parameters_case_a_paper.npz, keys ode/sde_selected_parameter),
# embedded so the release does not depend on the reproduction tree.
# Layout: [7 node values | nugget | amplitude | lengthscale (| sqrt(sigma))].
NEAR_OPTIMUM_RAW = {
    "ode": np.array([
        2.0022417249541293, 3.7055121950299226, -2.0767385618325562,
        0.5588313663584521, 2.46710117620933, 6.220674159284549,
        2.490887032198542, 0.0781742108820016, 1.3678526273315657,
        8.14118555431968,
    ]),
    "sde": np.array([
        1.3110982182081896, -0.5898280620068218, 0.8571446730230546,
        -0.3714291319893986, 2.888705424631116, 6.056562926605591,
        0.05352931037257124, 0.15599769681339337, 3.051436692332498,
        6.158302588719248, 1.5188884567902499,
    ]),
}


# ---------------------------------------------------------------- forward model
def truth_record(t_total: float, seed: int) -> np.ndarray:
    slow, _ = simulate_two_scale_lorenz96(
        K=K_SLOW, J=J_FAST, F=F_FORCE, h_slow=H_COUPLE, h_fast=H_COUPLE, c=C_FAST, b=B_FAST,
        dt=DT_FULL, t_total=t_total, burn_in=T_BURN, rng=np.random.default_rng(int(seed)),
        store_every=STORE_FULL, return_closure_residual=False,
    )
    return slow


def forward_stats(theta_raw: np.ndarray, seed: int, stochastic: bool, t_total: float = T_G) -> np.ndarray:
    gp, linear, sigma = unpack_theta_parts(
        np.asarray(theta_raw, dtype=float), NODES,
        learn_linear_coefficient=False, stochastic=stochastic, default_linear=LAMBDA,
    )
    trajectory = simulate_closed_lorenz96_gp_fast(
        theta_gp=gp, sigma=sigma if stochastic else 0.0, linear_coefficient=linear, nodes=NODES,
        K=K_SLOW, F=F_FORCE, dt=DT_FULL, t_total=t_total, burn_in=T_BURN,
        rng=np.random.default_rng(int(seed)), store_every=STORE_FULL,
    )
    return case_a_statistics(trajectory)


# ---- picklable worker entry points (ProcessPoolExecutor) ----
# The window length travels INSIDE the task: with spawn (Windows) a worker
# re-imports this module and would otherwise use the module default rather
# than the value main() set under --quick, silently mixing window lengths.
def _w_truth(task) -> np.ndarray:
    seed, t_y = task
    return case_a_statistics(truth_record(float(t_y), int(seed)))


def _w_forward(task) -> tuple[int, np.ndarray]:
    index, theta_raw, seed, stochastic, t_g = task
    return index, forward_stats(np.asarray(theta_raw, dtype=float), int(seed), bool(stochastic), t_total=float(t_g))


def _w_density(task) -> np.ndarray:
    # Long-run invariant-measure histogram; a blown-up member is recorded as a
    # zero density row (counted implicitly by its all-zero signature).
    theta_raw, seed, stochastic = task[0], task[1], task[2]
    gp, linear, sigma = unpack_theta_parts(
        np.asarray(theta_raw, dtype=float), NODES,
        learn_linear_coefficient=False, stochastic=stochastic, default_linear=LAMBDA,
    )
    try:
        trajectory = simulate_closed_lorenz96_gp_fast(
            theta_gp=gp, sigma=sigma if stochastic else 0.0, linear_coefficient=linear, nodes=NODES,
            K=K_SLOW, F=F_FORCE, dt=DT_FULL, t_total=1000.0, burn_in=100.0,
            rng=np.random.default_rng(int(seed)), store_every=STORE_FULL,
        )
    except FloatingPointError:
        return np.zeros(EDGES.size - 1)
    counts, _ = np.histogram(trajectory.reshape(-1), bins=EDGES, density=True)
    return counts


def warm_up() -> None:
    """Compile / load the numba kernels once in the parent before spawning."""
    truth_record(1.0, S0)
    forward_stats(np.concatenate([np.zeros(NODES.size), [0.5, 0.5, 10.0, 1.0]]), S0, True, t_total=1.0)


# ---------------------------------------------------------------- driver
def prior_mean_raw(stochastic: bool) -> np.ndarray:
    # Midpoints of the paper's Section 4.2 prior ranges (probe parameter only).
    parts = [np.zeros(NODES.size), [0.55, 0.55, 0.5 * (SETTINGS["length_low"] + SETTINGS["length_high"])]]
    if stochastic:
        parts.append([0.5 * (0.01 + 10.0)])
    return np.concatenate([np.asarray(p, dtype=float) for p in parts])


def near_optimum_raw(stochastic: bool) -> np.ndarray:
    """A point near the expected optimum: the frozen production fit (v12)."""
    return NEAR_OPTIMUM_RAW["sde" if stochastic else "ode"].copy()


def main(workers: int, output_dir: Path | None, quick: bool) -> Path:
    global T_Y, T_G, N_GAMMA, J_ENS, N_ITER, K_PROBE, N_G_FIXED
    if quick:                                   # tiny end-to-end check
        T_Y = T_G = 20.0
        N_GAMMA, J_ENS, N_ITER, K_PROBE = 60, 8, 3, 6   # N_Gamma > q = 44, else full Gamma is singular
        N_G_FIXED = None                                # tiny windows: calibrate instead
    started = time.perf_counter()
    out_root = Path(output_dir) if output_dir else (_HERE / "spec_results" / "case_a")
    out_root.mkdir(parents=True, exist_ok=True)
    warm_up()
    print(f"[case a spec] T_y = T_G = {T_Y:g}, N_Gamma = {N_GAMMA}, J = {J_ENS}, workers = {workers}", flush=True)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        # ---- 1.1 observation (one record, seed disjoint from the Gamma seeds) ----
        y = _w_truth((S0 + 1000, T_Y))

        # ---- 1.2 reference records ----
        t0 = time.perf_counter()
        seeds = [(S0 + 2000 + r, T_Y) for r in range(N_GAMMA)]
        records = np.array(list(pool.map(_w_truth, seeds, chunksize=1)))
        print(f"  {N_GAMMA} reference records in {time.perf_counter()-t0:.0f} s", flush=True)

        structures = ["full"]           # case (a) keeps the full covariance only
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
            "case": "lorenz96_case_a", "spec": "EKI_algorithm_spec_2026-08-23.md", "date": "2026-08-23",
            "T_y": T_Y, "T_G": T_G, "N_Gamma": N_GAMMA, "T_burn": T_BURN, "S0": S0, "J": J_ENS,
            "n_iter": N_ITER, "phi_tol": PHI_TOL, "delta_N_G": DELTA_NG, "K_probe": K_PROBE,
            "y_convention": "single_window", "forward_convention": "mean_of_N_G",
            "gamma_terms": "var_ref_only", "parameter_choice": "final_ensemble_mean",
            "gamma_diagnostics": {s: dict(rank=e.rank, condition_number=e.condition_number,
                                          corr_min_eigenvalue=e.corr_min_eigenvalue) for s, e in gammas.items()},
            "fits": {},
        }

        for stochastic, tag in ((False, "ode"), (True, "sde")):
            positive = positive_parameter_indices(NODES, learn_linear_coefficient=False, stochastic=stochastic)

            # ---- 1.3 calibrate N_G (two probes; the near-optimum one decides) ----
            probe_stats = {}
            for label, theta_raw in (("prior_mean", prior_mean_raw(stochastic)),
                                     ("near_optimum", near_optimum_raw(stochastic))):
                tasks = [(k, theta_raw, S0 + 7000 + 100 * int(stochastic) + k, stochastic, T_G) for k in range(K_PROBE)]
                got = sorted(pool.map(_w_forward, tasks, chunksize=1), key=lambda t: t[0])
                probe_stats[label] = np.array([g for _, g in got])
            cal = calibrate_n_g(probe_stats, var_ref, delta=DELTA_NG, decisive_probe="near_optimum")
            n_g = N_G_FIXED or cal.n_g          # None (--quick) -> use the calibration
            omitted = cal.ratio_max / n_g
            if N_G_FIXED and omitted > DELTA_NG:
                raise RuntimeError(
                    f"N_G_FIXED = {n_g} leaves {omitted:.1%} of the misfit variance out of "
                    f"Gamma (> delta = {DELTA_NG:.0%}); the calibration asks for {cal.n_g}")
            print(f"  [{tag}] N_G = {n_g} (calibration asks {cal.n_g}, omitted {omitted:.1%})"
                  f"  (ratio near-optimum min/med/max = "
                  f"{cal.ratio_min:.3f}/{cal.ratio_median:.3f}/{cal.ratio_max:.3f}; "
                  f"prior-mean implies {cal.per_probe['prior_mean']['n_g_implied']})", flush=True)

            # Per-fit evaluator state: the CRN seed sets actually used (one row
            # per ensemble evaluation) and the latent-guard clip counter.
            clip_count = {"n": 0}
            seeds_log: list[np.ndarray] = []
            base_seed = S0 + 500_000

            def evaluate(theta_latent: np.ndarray, iteration: int, _stoch=stochastic,
                         _clip=clip_count, _seeds_log=seeds_log, _n_g=n_g) -> np.ndarray:
                """N_G-averaged forward statistics for the whole ensemble (CRN).

                Every member evaluates the same seed set S(iteration, 0..N_G-1)
                (common random numbers); the set changes each iteration.  The
                declared latent guard clips the EVALUATED parameter only -- the
                ensemble keeps the coordinates EKI produced.
                """
                seed_set = crn_seeds(base_seed, iteration, _n_g)
                _seeds_log.append(np.asarray(seed_set).copy())
                safe, clipped = clip_latent(theta_latent, positive)
                _clip["n"] += clipped
                raw = decode_positive_columns(safe, positive)
                tasks = [(j * len(seed_set) + a, raw[j], int(s), _stoch, T_G)
                         for j in range(raw.shape[0]) for a, s in enumerate(seed_set)]
                got = sorted(pool.map(_w_forward, tasks, chunksize=4), key=lambda t: t[0])
                stacked = np.array([g for _, g in got]).reshape(raw.shape[0], len(seed_set), -1)
                averaged = stacked.mean(axis=1)
                if not np.all(np.isfinite(averaged)):
                    raise RuntimeError(f"iteration {iteration}: non-finite forward output")
                return averaged

            for structure, est in gammas.items():
                rng = np.random.default_rng(S0 + (202 if stochastic else 101))
                init_raw = initial_ensemble(
                    rng=rng, ensemble_size=J_ENS, nodes=NODES, length_low=SETTINGS["length_low"],
                    length_high=SETTINGS["length_high"], learn_linear_coefficient=False, stochastic=stochastic,
                )
                t1 = time.perf_counter()
                print(f"  [{tag} | {structure}] EKI ...", flush=True)
                # Spec section 2 via the shared engine: perturbed observations
                # drawn from the same rng that drew the initial ensemble (the
                # origin convention), absolute 1e-8 jitter on the gain solve,
                # stop when |dPhi|/Phi < PHI_TOL once (patience 1).
                res = run_eki(
                    encode_positive_columns(init_raw, positive), None, y, est.matrix,
                    n_iter=N_ITER, rng=rng, perturb_observations=True,
                    jitter=1e-8, jitter_mode="absolute", verbose=True,
                    ensemble_evaluator=evaluate,
                    stop_rel_tol=PHI_TOL, stop_patience=1,
                )
                stopped_on = "n_iter" if res.stop_reason == "n_iter" else "phi_tol"
                iterations_run = len(res.objective_history)  # ensemble evaluations
                final_raw = decode_positive_columns(clip_latent(res.theta_history[-1], positive)[0], positive)
                theta_mean, theta_sd = final_raw.mean(axis=0), final_raw.std(axis=0, ddof=1)
                _, _, sigma = unpack_theta_parts(theta_mean, NODES, learn_linear_coefficient=False,
                                                 stochastic=stochastic, default_linear=LAMBDA)
                sigma_members = np.array([unpack_theta_parts(t, NODES, learn_linear_coefficient=False,
                                                            stochastic=stochastic, default_linear=LAMBDA)[2]
                                          for t in final_raw]) if stochastic else np.zeros(J_ENS)

                # long-run invariant measure: ensemble-mean parameter + every member
                dens_tasks = [(final_raw[m], S0 + 20000 + m, stochastic) for m in range(final_raw.shape[0])]
                member_dens = np.array(list(pool.map(_w_density, dens_tasks, chunksize=1)))
                mean_dens = _w_density((theta_mean, S0 + (12000 if stochastic else 11000), stochastic))
                # single forward realisations for like-for-like figure comparison (spec section 3)
                single_tasks = [(k, theta_mean, S0 + 60000 + k, stochastic, T_G) for k in range(8)]
                singles = np.array([g for _, g in sorted(pool.map(_w_forward, single_tasks, chunksize=1), key=lambda t: t[0])])

                key = f"{tag}_{structure}"
                np.savez(
                    out_root / f"spec_{key}.npz",
                    y=y, records=records, gamma=est.matrix, var_ref=var_ref,
                    theta_history=res.theta_history, phi_history=res.objective_history,
                    phi_member_history=res.objective_member_history,
                    seeds_history=np.asarray(seeds_log),
                    g_history=res.output_history, final_ensemble_raw=final_raw, theta_mean=theta_mean,
                    theta_sd=theta_sd, member_dens=member_dens, mean_dens=mean_dens,
                    single_forward_stats=singles, ratio=cal.ratio, nodes=NODES, inv_edges=EDGES,
                )
                summary["fits"][key] = {
                    "gamma_structure": structure, "N_G": n_g, "N_G_calibrated": cal.n_g,
                    "omitted_variance_fraction": float(omitted),
                    "ratio": {"min": cal.ratio_min, "median": cal.ratio_median, "max": cal.ratio_max,
                              "per_probe": cal.per_probe},
                    "iterations_run": iterations_run, "stopped_on": stopped_on,
                    "stop_reason": res.stop_reason,
                    "phi_first": float(res.objective_history[0]), "phi_final": float(res.objective_history[-1]),
                    "phi_over_q_half": float(res.objective_history[-1] / (0.5 * len(y))),
                    "sigma": float(sigma), "sigma_sd": float(sigma_members.std(ddof=1)) if stochastic else 0.0,
                    "theta_mean": theta_mean.tolist(), "theta_sd": theta_sd.tolist(),
                    "latent_clips": int(clip_count["n"]),
                    "runtime_seconds": time.perf_counter() - t1,
                }
                print(f"  [{key}] {iterations_run} evaluations ({stopped_on}), "
                      f"Phi {res.objective_history[0]:.1f} -> "
                      f"{res.objective_history[-1]:.2f} (q/2 = {0.5*len(y):.0f}), sigma = {sigma:.3f}, "
                      f"{time.perf_counter()-t1:.0f} s", flush=True)

    summary["total_runtime_seconds"] = time.perf_counter() - started
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["fits"], indent=2)[:2000], flush=True)
    print(f"total {summary['total_runtime_seconds']:.0f} s -> {out_root}", flush=True)
    return out_root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--quick", action="store_true", help="tiny end-to-end check")
    args = parser.parse_args()
    main(args.workers, args.output_dir, args.quick)
