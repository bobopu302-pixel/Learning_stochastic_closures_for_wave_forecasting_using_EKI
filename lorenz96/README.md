# `lorenz96` — SSW21 reproduction, two-scale Lorenz-96 case (a)

EKI stochastic-closure reproduction on the two-scale Lorenz-96 system
(thesis Chapter 3, case (a): c = 10, h = 1).  A Gaussian-process closure on 7
nodes replaces the fast-variable coupling; an ODE fit (sigma = 0) and an SDE
fit (sigma learned) are estimated by ensemble Kalman inversion under the
2026-08-23 algorithm spec: log-space positive parameters, common random
numbers, N_G forward averaging, full-covariance Gamma from independent
reference records, final-ensemble-mean reporting, and the relative-Phi
stopping rule.  The linear damping coefficient lambda = h^2 c / J is fixed
(not learned) in case (a).

No figures are produced; every result is persisted as data (see below).

## File map

| File | Role |
|---|---|
| `simulator.py` | Two-scale truth system + closed GP model (numba), truth-trajectory disk cache, truth-system constants (`K_SLOW`, `J_FAST`, `F_FORCE`, `B_FAST`, `DT_FULL`, `STORE_FULL`). |
| `observations.py` | Case definitions: the 44-dim statistics vector (`case_a_statistics`), theta packing (`unpack_theta_parts`), prior ensemble (`initial_ensemble`), log-space column indices (`positive_parameter_indices`). |
| `run_spec.py` | The spec driver (entry point): observation, Gamma, N_G calibration, both EKI fits, invariant-measure densities, all data saving. |

Shared machinery comes from `../algorithms` (EKI engine, Gamma/N_G
estimators, GP conditional mean, log parameterization, moment statistics);
see `../algorithms/README.md`.

## Run

Run **from this folder** (the driver bootstraps `sys.path` itself):

```
cd "4. Dissertation/code_rp/lorenz96"
python run_spec.py --workers 8
```

Flags:

- `--workers N` — process-pool size (default: CPU count − 2).  The J × N_G
  forward runs of each EKI iteration are distributed over the pool; every
  worker re-imports this module, so the numba kernels compile once per
  process and then hit the on-disk cache.
- `--output-dir PATH` — where to write results (default:
  `spec_results/case_a/` inside this folder).
- `--quick` — tiny end-to-end check (T = 20, N_Gamma = 60, J = 8, 3
  iterations, calibrated N_G); finishes in a few minutes and exercises every
  code path including the saving.

Environment variables: none are required.  When using many workers, pinning
the per-process BLAS threads avoids oversubscription:

```
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
python run_spec.py --workers 32
```

The production configuration (T_y = T_G = 1500, N_Gamma = 200, J = 100,
N_G = 20, up to 30 iterations) is CPU-heavy — it was run on a rented
many-core server; on a laptop use `--quick` to validate the pipeline.

## What gets saved where

Everything lands in the output directory (default `spec_results/case_a/`):

- `spec_ode_full.npz`, `spec_sde_full.npz` — one archive per fit
  (`full` = full-covariance Gamma, the only scheme kept for case (a)):
  - `y` (44,) observation; `records` (N_Gamma, 44) reference records;
    `gamma` (44, 44); `var_ref` (44,) reference variances;
  - `theta_history` (n+1, J, p) latent ensemble history; `g_history`
    (n_eval, J, 44) N_G-averaged forward outputs; `phi_history` (n_eval,)
    objective of the ensemble-mean output; `phi_member_history` (n_eval, J);
    `seeds_history` (n_eval, N_G) CRN seed sets actually used;
  - `final_ensemble_raw` (J, p) decoded final ensemble; `theta_mean`,
    `theta_sd` (p,) the reported estimate (final-ensemble mean ± sd);
  - `member_dens` (J, 160) long-run invariant-measure densities (t = 1000
    after 100 burn-in) for every final member; `mean_dens` (160,) same for
    the ensemble-mean parameter; `inv_edges` (161,) histogram edges;
  - `single_forward_stats` (8, 44) single-realisation statistics at the mean
    parameter (like-for-like comparison with the single-window y);
  - `ratio` (44,) forward/reference variance ratios from the N_G
    calibration; `nodes` (7,) GP nodes.
- `summary.json` — run configuration, Gamma diagnostics (rank, condition
  number, minimum correlation eigenvalue), per-fit N_G calibration results,
  iterations run and stop reason, Phi first/final (and Phi / (q/2)), the
  learned sigma, `theta_mean`/`theta_sd`, latent clip counts, runtimes.

Parameter vector layout (p = 10 ODE / 11 SDE):
`[7 GP node values | nugget | amplitude | lengthscale (| sqrt(sigma))]`;
the last 3 (4) columns are evolved in log space, and the model noise variance
is `sigma = sqrt(sigma)^2`.

## Seed conventions

Base seed `S0 = 31`.  All streams are disjoint `default_rng` seed blocks:

| Stream | Seed |
|---|---|
| Observation record y | `S0 + 1000` |
| Gamma reference record r | `S0 + 2000 + r` |
| N_G probe run k (ODE / SDE) | `S0 + 7000 + 100*stochastic + k` |
| EKI init ensemble + observation perturbations (ODE / SDE) | `default_rng(S0 + 101)` / `default_rng(S0 + 202)` |
| CRN forward seeds | `S(n, r) = (S0 + 500000) + 1000003 (n+1) + r`, r = 0..N_G−1 |
| Member invariant densities | `S0 + 20000 + member` |
| Mean-parameter density (ODE / SDE) | `S0 + 11000` / `S0 + 12000` |
| Single forward realisations | `S0 + 60000 + k` |

Common random numbers: within one EKI iteration every ensemble member uses
the SAME seed set `S(n, ·)`, so member differences reflect parameters rather
than noise draws; the set changes every iteration.
