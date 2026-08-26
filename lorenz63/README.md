# `lorenz63` — SSW21 reproduction on noisy Lorenz 63 (thesis Chapter 3)

EKI stochastic-closure calibration on the additive-noise Lorenz 63 system,
under the 2026-08-23 algorithm specification (`EKI_algorithm_spec_2026-08-23.md`)
— the configuration actually reported in the thesis. Four fits, exactly as in
the source paper:

| Fit | Closure `g_L(x2)` | Learned parameters |
|---|---|---|
| `fixed_ode` | fixed `x2` | `alpha` |
| `fixed_sde` | fixed `x2` | `alpha`, `sqrt(sigma)` |
| `gp_ode` | GP conditional mean | 5 node values, nugget, amplitude, lengthscale |
| `gp_sde` | GP conditional mean | GP block + `sqrt(sigma)` |

Spec conventions (all implemented here):

- **y**: statistics (3 means + 6 centered second moments, q = 9) of ONE
  reference record of length `T_y = 1500` after a burn-in of 20.
- **Gamma**: the FULL sample covariance of `N_Gamma = 200` independent
  reference records (reference side only — no forward term, no floor).
- **G(theta)**: mean of `N_G = 20` forward runs of length `T_G = T_y` with
  common random numbers (CRN). The near-optimum probe is still run and the
  realised omitted variance fraction is checked against `delta = 0.10`.
- **EKI**: perturbed observations, log-space positive parameters, stop when
  the relative change of `Phi` falls below `phi_tol = 0.01`.
- **Reporting**: final-ensemble mean +/- sd, no selection.

## Files

| File | Role |
|---|---|
| `simulator.py` | Euler-Maruyama Lorenz 63 simulator (`simulate_lorenz63` + numba `_sim_core`), extracted from the legacy runner |
| `run_spec.py` | The spec driver: builds y and Gamma, checks N_G, runs the four EKI fits on a process pool, saves per-fit npz + `summary.json` |
| `spec_results/` | Default output directory (created on first run) |

Algorithm code (EKI engine, Gamma/N_G estimators, GP mean, log
parameterization, statistics) is imported from `../algorithms` — see
`../algorithms/README.md`.

## Run

From THIS folder (`code_rp/lorenz63/`):

```
python run_spec.py --workers 190
```

Options:

- `--workers N` — process-pool size (default: CPU count minus 2). The
  production run was sized for a rented many-core machine; a laptop run at
  full `T_y = 1500` takes many hours.
- `--output-dir DIR` — write results elsewhere (default `./spec_results`).
- `--only {fixed_ode,fixed_sde,gp_ode,gp_sde}` — run a single fit.
- `--quick` — smoke-test configuration (`T_y = T_G = 20`, `N_Gamma = 30`,
  `J = 8`, 3 iterations, N_G from the calibration instead of the fixed 20).
  Minutes, not hours; results are NOT scientifically meaningful.

No environment variables are required. Requires `numpy`, `scipy`, and
`numba` (the simulator kernel is numba-compiled; the first call compiles and
caches it).

## Outputs

Everything goes to the output directory:

- `spec_<fit>_full.npz` for each fit (`fixed_ode`, `fixed_sde`, `gp_ode`,
  `gp_sde`), with keys:
  - `y` (q,) observation; `records` (N_Gamma, q) reference statistics;
    `gamma` (q, q) full covariance; `var_ref` (q,) its diagonal
  - `theta_history` (n+1, J, p) latent ensemble history;
    `g_history` (n_eval, J, q) N_G-averaged forward outputs;
    `phi_history` (n_eval,) objective of the ensemble-mean output;
    `phi_member_history` (n_eval, J); `seeds_history` (n_eval, N_G) CRN seed
    sets actually used
  - `final_ensemble_raw` (J, p) decoded final ensemble; `theta_mean`,
    `theta_sd` (p,) the reported estimate
  - `member_dens` (J, 3, 160), `mean_dens` (3, 160) invariant-measure
    histograms over `T = 5000` on `inv_edges`; `single_forward_stats` (8, q)
    single-run statistics at the mean; `ratio` (q,) N_G calibration ratios;
    `nodes` (5,) GP design nodes
- `summary.json` — full configuration, Gamma diagnostics (rank, condition
  number, minimum correlation eigenvalue), and per-fit results: N_G check,
  iterations, stop reason, `Phi` first/final, `theta_mean +/- theta_sd`,
  `alpha`/`sigma` where applicable, latent clip count, runtimes.

## Seed conventions

One master seed `S0 = 7`; every random object gets a disjoint block:

| Purpose | Seed |
|---|---|
| Observation record y | `S0 + 1000` |
| Gamma reference records | `S0 + 2000 + r`, `r = 0..N_Gamma-1` |
| N_G probe forwards | `S0 + 7000 + 1000*fit_index + k` |
| Initial ensemble + observation perturbations (one `Generator`) | `S0 + 100*fit_index` |
| CRN forward seeds within EKI | `S(n, r) = base + 1_000_003*(n+1) + r`, `base = S0 + 500_000 + 1000*fit_index` |
| Invariant densities (members / mean) | `S0 + 20000 + m` / `S0 + 30000` |
| Single forward runs at the mean | `S0 + 60000 + k` |

`fit_index` is the position in `("fixed_ode", "fixed_sde", "gp_ode",
"gp_sde")`. CRN: within one EKI iteration every ensemble member uses the SAME
seed set, so member differences reflect parameters, not noise draws; the set
changes every iteration.
