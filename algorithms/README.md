# `algorithms` — the shared algorithm layer

One algorithm layer, imported by every case driver (Lorenz 63/96 reproduction,
linear water waves, coastal vKdV). Every module here depends on numpy — plus
one lazy scipy import: `eki.py`'s objective evaluation whitens with the exact
Gamma via `scipy.linalg.solve_triangular` (scipy is in `requirements.txt`). No
numba, no matplotlib.

**Design rule: physics and observation-vector assembly live in the case
folders; only case-agnostic algorithm code lives here.** Dispersion relations,
modal propagators, PDE solvers, statistic block layouts, lags/bands choices,
parallel worker pools, and BLAS thread pinning are all case-side. This package
contains the EKI engine, the Gamma/N_G spec estimators, the GP conditional
mean, the log parameterization, and the raw estimator primitives — nothing
that knows what a wave or an attractor is.
---

## `eki.py` — the engine

```python
result = run_eki(
    initial_ensemble,          # (J, p) — in the coordinates EKI evolves
    forward_map,               # theta, rng -> (q,)  (or None, see ensemble_evaluator)
    observation,               # y, (q,)
    gamma,                     # Gamma, (q, q) or diagonal (q,)
    *,
    n_iter,                    # maximum number of Kalman updates
    rng,                       # np.random.Generator
    perturb_observations=False,
    bounds=None,               # [(lo, hi), ...] box clip after each update
    jitter=1e-8,               # conditions the Kalman-gain solve ONLY (Phi is always exact)
    verbose=True,
    sentinel_value=None,       # constant failure vector to detect/count
    ensemble_evaluator=None,   # (theta_ensemble, iteration) -> (J, q); overrides forward_map
    stop_rel_tol=None,         # 2026-08-23 spec stopping rule (None = fixed n_iter)
    stop_patience=3,
    observation_rng=None,      # own stream for eta ~ N(0, Gamma)
    # -- extensions (defaults reproduce the linear-wave engine exactly) --
    sentinel_row_fn=None,      # (y, diag(Gamma)) -> (q,) replacement row for failed members
    failed_mask_fn=None,       # (outputs (J, q)) -> (J,) bool mask of failed members
    post_update=None,          # (thetas, iteration) -> thetas, after each update
    iteration_callback=None,   # (state dict) -> None, per applied update (checkpointing)
    jitter_mode="relative",    # 'relative' (wave) | 'absolute' (legacy Lorenz/vKdV)
) -> EKIResult
```


## `gamma.py` — Gamma and N_G (spec-2026-08-23 reference implementation)

- `build_gamma(records, *, structure, families=None, neff_correction=False)
  -> GammaEstimate`: Gamma from `N_Gamma` independent reference windows
  (`records` is `(N_Gamma, q)`), `"diagonal"` (ddof=1 per-component variances)
  or `"full"` (sample covariance with rank/condition checks); optional
  effective-sample-size inflation for correlated statistic families. **No
  forward term, no floor** — the model's own fluctuation is handled by N_G
  forward averaging, not by inflating Gamma.
- `calibrate_n_g(probe_stats, var_ref, *, delta=0.10, decisive_probe,
  retain_fraction=0.01) -> CalibrationResult`: choose N_G so the omitted
  forward-variance term `var_fwd / N_G` is at most `delta` of the reference
  variance, judged at the decisive probe; near-degenerate components dropped.

## `gpr.py` — GP conditional mean (the learned closure function)

`make_gp_mean(nodes, values, amplitude, lengthscales, nugget)` unifies the 1-D
closure curve (Lorenz) and the 2-D closure surface (vKdV): `nodes` is `(R,)`
or `(R, d)`, `lengthscales` scalar or `(d,)` (anisotropic product RBF),
representer solve `(K_nn + (nugget^2 + 1e-8) I) w = v`, callable
`m(x) = k(x, nodes) @ w`. The node `values` are **free parameters estimated by
EKI** ("virtual observations"); `m` interpolates them only as `nugget -> 0`;
amplitude and nugget enter the mean only through `nugget^2 / amplitude^2`. For
1-D nodes the callable carries the `_gp_params = (nodes, weights, amplitude,
lengthscale)` contract used by the numba fast path in the Lorenz runs.
`make_gp_mean_from_theta` unpacks the packed `(v, tau, a, ell)` vector.

## `statistics.py` — estimator primitives

Moved, not rewritten; every docstring names its consuming case(s).

| Function | Case |
|---|---|
| `normalized_autocorr`, `gauge_acf`, `xcorr_pair`, `cross_corr` | linear wave |
| `band_energy_spectrum` | linear wave |
| `demeaned_acf` | vKdV |
| `centered_first_second_moments`, `raw_first_second_moments` (+ compat alias `first_second_moments`) | Lorenz 63/96 |
| `cov_from_samples` | Lorenz (legacy full-Gamma path; spec path is `gamma.build_gamma`) |
| `histogram_density` | Lorenz (bundle.npz densities) |

## `parameterization.py` — log encoding

`log_encode` / `log_decode` (safe range `[log(1e-12), 80]`),
`encode_positive_columns` / `decode_positive_columns` for vectors, ensembles,
and histories, and `gp_positive_indices` for the GP block layout. EKI evolves
latent coordinates; decode inside the forward map, never inside the engine.
