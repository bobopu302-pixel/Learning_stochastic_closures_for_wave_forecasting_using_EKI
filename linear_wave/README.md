# Linear wave case: dispersion-locked stochastic modal closure

Reproduction code for **thesis Chapter 4**: a broadband linear wave field
(100 random-frequency components, 10 gauges) closed by a ten-mode stochastic
modal model whose frequencies are fixed by the finite-depth dispersion relation
and whose per-mode damping `delta_j` and noise amplitude `sqrt(sigma_j)` are
estimated by ensemble Kalman inversion (EKI):

    dq_j = p_j dt,
    dp_j = (-omega_j^2 q_j - delta_j p_j) dt + sqrt(sigma_j) dW_j,
    omega_j^2 = g k_j tanh(k_j h).

This release ships the **2026-08-23 spec configuration actually used in the
thesis** and nothing else: log-space positive parameters, common random
numbers, `G_hat` = mean of `N_G` forward runs, `Gamma = diag(var_ref)` from
independent reference records (no forward term, no floor), final-ensemble-mean
reporting, and the 1%-for-3-iterations stopping rule.  Legacy code paths were
deleted, not disabled; each module's docstring records what was removed.

The EKI engine, the diagonal-Gamma estimator and the statistic estimator
primitives come from the shared package `code_rp/algorithms/`
(`algorithms.eki.run_eki`, `algorithms.gamma.build_gamma`,
`algorithms.statistics`).  The engine was originally built *from* this case's
`modal_closure/eki.py`, so the call is drop-in; that local module is removed.

## File map

| file | role |
|---|---|
| `run_closure.py` | the only command-line entry point (validate / replay / audit / recompute / `--calibrate-only`) |
| `modal_closure/__init__.py` | pins BLAS/OpenMP to one thread **before numpy loads** (critical for the spawn pool), re-exports the case constants |
| `modal_closure/truth.py` | 100-component random-phase truth generator (analytic, seeded) |
| `modal_closure/experiment.py` | case definition: truth spectrum, 38-dim statistics vector, y/Gamma build, N_G calibration, exact-OU forward model, parallel evaluation, the EKI run, `bundle.npz` + `summary.json` writers |
| `modal_closure/numerics.py` | wave physics only: dispersion relation and the exact Ornstein-Uhlenbeck modal propagator |
| `modal_closure/bundle_io.py` | `save_bundle` (renamed from the misnamed `plotting.py`) |
| `modal_closure/metrics.py` | bundle validation + `derived_metrics.json` (renamed from `diagnostics.py`; figures removed) |
| `modal_closure/audit.py` | post-hoc audit: out-of-sample objective, irreducible truth level, like-for-like spectrum, velocity-readout Ito check; writes `audit_metrics.json` + `audit_arrays.npz` |

## How to run

Run **from this case folder** (`code_rp/linear_wave/`), with a Python that has
numpy and scipy:

```bash
cd linear_wave

# Full EKI calibration (the frozen thesis run's configuration).
# Refuses to overwrite an existing bundle unless --force is given.
python run_closure.py --mode recompute --output results/modal_closure

# Validate a stored bundle (shape/energy/identity checks only).
python run_closure.py --mode validate --results results/modal_closure

# Validate + rewrite derived_metrics.json (no figures in this release).
python run_closure.py --mode replay --results results/modal_closure

# Post-hoc audit (out-of-sample Phi, irreducible truth level, spectrum,
# velocity readout).  Never modifies bundle.npz / summary.json /
# derived_metrics.json.
python run_closure.py --mode audit --results results/modal_closure

# Error model + N_G calibration printout only, then stop.
python run_closure.py --calibrate-only
```

Useful flags: `--workers N` (1 = serial), `--n-g`, `--n-gamma`, `--t-record`,
`--audit-repeats`, `--calib-probe-from <results dir>` (feeds a previous run's
identified theta to the N_G calibration as the spec's near-optimum probe).

Environment variables (all optional):

| variable | effect |
|---|---|
| `MODAL_CLOSURE_M_MODES` | model mode count M (default 10); env rather than a flag because spawned workers re-import the module and must land on the same grid |
| `MODAL_CLOSURE_ENSEMBLE` | EKI ensemble size J (default 100) |
| `MODAL_CLOSURE_WORKERS` | worker processes (default: CPU-affinity-sized) |
| `MODAL_CLOSURE_TRUTH_WORKERS` | separate cap for the memory-heavy truth-generator stage (~0.35 GB/worker) |
| `MODAL_CLOSURE_CHECKPOINT` | path of an `.npz` salvage checkpoint written after every ensemble evaluation (not a bit-exact resume) |

## What gets saved where

`--mode recompute` writes into `--output` (default `results/modal_closure/`):

- `bundle.npz` -- every array needed to validate and re-analyse the run: long
  truth/model records and ensembles, spectra, `y` (`target_statistics`),
  `gamma_diag`/`gamma_var_ref`, the full theta/objective histories, the final
  ensemble (log and physical summaries), and 50 unaveraged
  `sde_single_realisations` for error-bar comparisons.
- `summary.json` -- configuration, conventions, seeds (with the block table),
  Gamma diagnostics, the N_G calibration + coverage report, and the EKI
  stopping record.
- `MODAL_CLOSURE_CHECKPOINT` (if set) -- per-evaluation ensemble checkpoint.

`--mode replay` writes `derived_metrics.json` next to the bundle.
`--mode audit` writes `audit_metrics.json` and `audit_arrays.npz` (every raw
audit sample) next to the bundle; it never touches the other three files.

No figures are produced anywhere: this release ships computation and data
saving only.  Everything a figure was drawn from is in the `.npz`/`.json`
files above.

## Seed conventions

Seven **disjoint seed blocks**, separated by 10 000 and asserted disjoint at
run time by `experiment.check_seed_blocks()` (so `y` is provably independent
of its own error bar, and the forward randomness of both):

| block | seed | used for |
|---|---|---|
| `observation_y` | 22001 | the single reference record behind `y` |
| `gamma_reference` | 30000+i | the `N_Gamma = 200` records behind `var_ref` |
| `forward_crn` | 40000 | common random numbers of `G_hat` |
| `observation_perturbation` | 50000 | `eta ~ N(0, Gamma)` in the EKI update |
| `initial_ensemble` | 60000 | log-uniform initial ensemble (60001 drives the engine's internal generator) |
| `single_realisations` | 70000 | 50 unaveraged `G(theta_hat)` draws |
| `calibration` | 80000 | the K probe runs of the N_G calibration |

Audit-only seeds are further out still: 90 000 (out-of-sample), 300 000
(truth reference), 777 001 (velocity check).  The truth frequency draw is
seed 7; long validation records use 1021 (truth) and 49/1042+ (model).

Common random numbers: within one EKI iteration, every member's `N_G`
replicates use the same seed keys `(FORWARD_CRN_SEED, iteration, replicate)`
-- the member index is deliberately omitted, which is what makes the residual
forward noise a common shift that cancels in the ensemble anomalies.  Fresh
keys every iteration.

## Reproducibility design

- **Spawn-pinned pool.** The process pool always uses the `spawn` start
  method, and `modal_closure/__init__.py` pins every BLAS/OpenMP thread count
  to 1 *before numpy is imported*.  Forking would inherit an already-threaded
  OpenMP runtime (silent oversubscription) and stale module constants.
- **Per-job SeedSequence.** Every simulation draws from
  `np.random.default_rng(np.random.SeedSequence(seed_key))` where the key
  travels inside the job tuple, together with the window length and the
  coordinate flag -- a worker never reads run-dependent state from the module.
- **Bitwise-identical across worker counts.** Because of the two points above,
  1, 8, 96 and 188 workers return bitwise-identical arrays (verified on a
  192-core server).  ACROSS machines the streams are identical but the arrays
  differ in the last one or two bits (different libm/BLAS builds; Windows vs
  Ubuntu agree to ~1e-15 relative) -- which is why the audit's error-model
  rebuild asserts `rtol = 1e-11` rather than bit equality.

## Frozen-run anchor numbers (for cross-checking a recompute)

From the thesis run (`T = 1000 s`, `N_Gamma = 200`, `N_G = 100`, `J = 100`,
`q = 38` statistics: 2 variances, 8 ACF lags, 18 signed cross-correlations,
10 band energies):

| quantity | value |
|---|---|
| final objective `Phi` | **31.39 = 1.65 x q/2** (q = 38) |
| sentinel evaluations | **0** |
| spectrum L1, like for like (audit) | 17.5% |
| total band-energy error | +5.7% |
| `eta` / `v` KS distance | 0.011 / 0.019 |
| ACF RMSE | 0.048 |
| out-of-sample `Phi` (median, single runs) | 204.3 |
| irreducible truth level (median, single runs) | 24.6 |

The last two use single unaveraged forward runs and are **not** comparable
with the reported `Phi`, which averages `N_G = 100` runs (smoother by
`sqrt(N_G) = 10`); `audit_metrics.json` labels which is which.  The analysis
window is set by **stability, not identifiability** (their admissible ranges
overlap only on 879-1375 s; 1000 s is inside), `delta_3` and `delta_6` are
not individually identified at this window, and the like-for-like periodogram
number -- not the modal-energy-vs-binned-truth number, which is inflated by
Lorentzian band leakage -- is the spectral accuracy to quote.

## Naming traps

- **`bundle.npz -> sde_best` is the final-ensemble MEAN** (physical units),
  not a selected best member.  The key predates the 2026-08-23
  no-selection reporting convention and is kept so archived bundles still
  validate; `sde_log_mean`/`sde_log_sd`/`sde_theta_sd`/`sde_final_ensemble`
  carry the honest names.  `audit_arrays.npz -> selected_theta` stores the
  same vector under the same historical misnomer.
- **`derived_metrics.json -> objective.reported_final_ensemble_mean`** was
  called `selected_member` in the source tree; it is the objective of the
  final-ensemble-mean output and equals `objective.final_ensemble_mean`.  The
  key was renamed here because no shipped code consumed the old name.

## Changes vs the source tree (`2.Linear_wave_case/`)

- `modal_closure/eki.py` **removed**; the case now calls
  `algorithms.eki.run_eki` (the shared engine was built from that file -- the
  call is drop-in, and the default `jitter_mode='relative'` is this case's
  convention).  `sys.path` is bootstrapped to the `code_rp` root in
  `run_closure.py` and again at the top of `experiment.py` so spawned workers
  resolve `algorithms` too.  One declared engine-behaviour change
  (release decision 2026-08-25): the shared engine computes `Phi` with the
  EXACT Gamma (Cholesky whitening; `jitter` conditions the Kalman-gain solve
  only), whereas the archived frozen run's engine regularised the objective
  solve by a relative 1e-8 diagonal inflation.  A fresh recompute's
  `phi_history` may therefore differ from the archived bundle at `<= 1e-8`
  relative -- far below every quoted precision and audit tolerance; the
  ensemble trajectory is unaffected (the gain solve is unchanged), and the
  archived bundle values themselves are untouched (`metrics.py` and the
  validators read the stored `Phi`, they never recompute it).
- `modal_closure/numerics.py` keeps only the wave physics; the estimators
  (`normalized_autocorr`, `gauge_acf`, `xcorr_pair`, `cross_corr`,
  `band_energy_spectrum`) are imported from `algorithms.statistics`.
- The diagonal-Gamma variance estimate delegates to
  `algorithms.gamma.build_gamma(structure='diagonal')`; the **N_G calibration
  stays local** because this case applies the spec rule at the 75th percentile
  of `var_fwd/var_ref` (the literal max rule would demand `N_G ~ 1e5` due to
  structurally tiny `var_ref` on low-energy bands) and filters sentinel-failed
  probe runs -- not compatible with `algorithms.gamma.calibrate_n_g`.
- `plotting.py` renamed to `bundle_io.py` (only ever wrote the bundle);
  `replot()` deleted.  `diagnostics.py` renamed to `metrics.py`, keeping only
  the computation half (no matplotlib, no figure builders).  `audit.py` lost
  its three figure functions; all JSON/NPZ outputs unchanged.
- `run_closure.py --mode replot` removed; `replay` = validate + metrics.
- Deleted dead/legacy paths (recorded per file in the module docstrings): the
  per-member seeding branch of `_ensemble_evaluator`, the unused `seed`
  parameter of `fit()` and the `EKI_FIT_SEED` constant.
- **Not shipped**: `run_server.sh` (it passes a `--gamma-full` flag that
  `run_closure.py` does not implement -- running it would crash at the first
  stage; use the commands above directly), the `results/` trees, the
  `beyond_lag` scripts, `tools/tectonic.exe`, `report/`, and
  `test_modal_closure.py` (regresses against the archived results tree, which
  is not part of this release).
- All comments and docstrings are in English; numerics of every retained code
  path are untouched.
