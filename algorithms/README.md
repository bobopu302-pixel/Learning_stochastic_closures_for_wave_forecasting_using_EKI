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

Provenance: each module's docstring records its exact source file under
`research_project/` and the changes made (mostly comments/docstrings only; the
EKI engine additionally gained optional hooks, all defaulting to the original
behaviour, and its objective was de-regularised — see below).

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

Semantics (see the module docstring for the full contract):

- **Perturbed-observations EKI** with **unbiased `J - 1` empirical
  covariances**; gain solve `(C^GG + Gamma) K^T = (C^thetaG)^T`, conditioned
  by `jitter`/`jitter_mode`.
- **Objective** `Phi = 0.5 (G - y)^T Gamma^{-1} (G - y)`, tracked both for the
  ensemble-mean output (drives "best iteration" and the stopping rule) and per
  member ("best member"). **Uniform convention: Phi is always exact** —
  computed by Cholesky whitening of the un-regularised Gamma (the
  `eki_spec._phi` construction, via a lazy `scipy.linalg.solve_triangular`
  import); `jitter`/`jitter_mode` condition the **Kalman-gain solve only** and
  never enter Phi.
- **Stopping rule** (2026-08-23 spec): relative change of the ensemble-mean
  objective below `stop_rel_tol` for `stop_patience` consecutive iterations,
  checked **before** the update, so the reported final ensemble is the one that
  satisfied the rule.
- **Final extra evaluation**: after the last update the new ensemble is
  evaluated once (rng state restored afterwards) so it competes for "best".
- **Gamma validation**: finite, symmetric, Cholesky-positive-definite — a wrong
  Gamma fails loudly before any randomness is drawn.
- **Log-space parameters are the caller's responsibility** (encode/decode via
  `parameterization`); the **reported estimate is the caller's convention** —
  `EKIResult` exposes both `final_mean/final_sd` (spec reporting, no selection)
  and `best_*` (legacy paper-figure selection).
- `EKIResult` also records `sentinel_counts` (failed members per evaluation),
  `n_updates`, and `stop_reason`.

CRN helpers: `crn_seed(base_seed, iteration, realisation)` and its vector form
`crn_seeds(base_seed, iteration, n_g)` reproduce the spec's seed set
`S(n, r) = base_seed + 1_000_003 (n + 1) + r`: shared streams across members
within an iteration, fresh set each iteration. `clip_latent` is the declared
latent-range guard for log-parameterised cases.

### Injection points and who uses them

| Injection point | What it does | Used by |
|---|---|---|
| `ensemble_evaluator` | whole-ensemble evaluation with caller-keyed streams (parallelisable) | linear wave (process pool), vKdV, Lorenz spec runs |
| `forward_map` | legacy one-member-at-a-time path, shared rng | early Lorenz reproduction runs |
| `sentinel_value` | count members returning the constant failure vector | linear wave |
| `failed_mask_fn` | case-defined failure detection on the (J, q) outputs | vKdV (solver blow-up flags) |
| `sentinel_row_fn` | replace failed rows, convention `y + 10*sqrt(diag(Gamma))` | vKdV |
| `post_update` | clip hook after each Kalman update | Lorenz (`clip_latent`-based guard), vKdV spec clip |
| `iteration_callback` | per-iteration state for checkpoint/`--resume` | vKdV long (> 1 h) runs |
| `stop_rel_tol` / `stop_patience` | spec-2026-08-23 stopping rule | all spec-convention runs |
| `observation_rng` | disjoint seed block for the observation noise | all spec-convention runs |
| `jitter_mode` | `'relative'` scale-free conditioning vs `'absolute'` `+ jitter*I` — gain solve only, never the objective | wave = relative; legacy Lorenz/vKdV = absolute |
| `bounds` | box clip in the evolved coordinates | cases with hard parameter boxes |

Every hook defaults to "off": calling `run_eki` with only the original
arguments reproduces the linear-wave engine's ensemble trajectory bit for bit.
The one deliberate departure is the objective (release decision 2026-08-25):
the origin engine regularised the objective solve too, so its reported Phi
carried a relative-1e-8 diagonal inflation; this engine's Phi is exact, a
`<= 1e-8` relative difference.

---

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
