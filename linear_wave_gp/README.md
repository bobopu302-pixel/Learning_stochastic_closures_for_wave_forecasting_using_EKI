# Linear wave case, GP closure: the thesis Chapter 4 MAIN results

Reproduction code for the run behind **thesis Chapter 4's main results**:
Figures 4.1–4.2, Table 4.1, the long-run validation of Figure 4.3(a,b), and
the GP-closure ACF curves discussed alongside Figure 4.3(c).  The
**modal-closure companion** (per-mode *linear* damping `-delta_j p_j`,
100-component reference) that supplies Figure 4.3(c)'s M=10/M=25 truth-side
curves lives in [`../linear_wave/`](../linear_wave/README.md); this folder is
the model that replaces that fixed damping by **two learned Gaussian-process
functions**, one pair shared by all ten modes:

    dq_j = [ p_j + Phi_q(q_j) ] dt
    dp_j = [ -omega_j^2 q_j + Phi_p(p_j) ] dt + sqrt(sigma_j) dW_j,  j = 1..10

    theta = [ Phi_q: 9 node values | obs_noise, amplitude, lengthscale ]
            [ Phi_p: 9 node values | obs_noise, amplitude, lengthscale ]
            [ sqrt(sigma_1) .. sqrt(sigma_10) ]          = 34 parameters

against a **deterministic N = 10 000-component** random-phase reference,
under the 2026-08-23 algorithm spec: `T_y = T_G = 1000 s`,
`Gamma = diag(var_ref)` from `N_Gamma = 200` independent reference records
(no forward term, no floor), `N_G = 100` common-random-number forward runs
per member, `J = 100`, 30 iterations (cap; the 1%-for-3 stopping rule never
fired), stationarity gate + ensemble member resampling.

The authoritative source is the frozen run's **code snapshot**
`5.GP_wave_closure/results/gp_T1000_NG100_9nodes_resample/code/` (saved
2026-08-24; driver invocation on the run machine:
`run_gp.py --model gp --workers 188 --n-g 100`).  The archived data of that
run also lives in `4. Dissertation/result_rp/chapter4_linear_wave/`.

## File map

| file | role |
|---|---|
| `run_gp.py` | the EKI driver (thesis run: `--model gp`); builds y and Gamma, calibrates N_G at the prior mean, runs the resampling EKI, writes `bundle.npz` + `summary.json` |
| `gp_closure.py` | case/closure definition: GP-function parameterisation, van Loan split-step integrator, stationarity gate, batched forward map, 38-dim statistics assembly, priors, latent coordinates |
| `eki_resampling.py` | the frozen run's EKI loop **with ensemble member resampling** — case-local on purpose, see "Why the engine is local" below |
| `stochastic_truth.py` | the reference wave fields: the deterministic fixed-amplitude generator the run observes (N from env `STOCHASTIC_N`, frozen at 10 000) and the stochastically forced variant kept for like-for-like comparisons |
| `long_run_validation.py` | post-run: 40 fitted members vs 40 reference records over 1000 s; writes the `longrun_*.json` scalars quoted in Table 4.1 (Var, H_s, skew, excess kurtosis) |
| `dump_longrun_arrays.py` | post-run: same simulation/seeds/estimators; saves the Figure 4.3(a,b) curves (KDE densities, tails, ACF envelope) to `longrun_arrays.npz` |
| `beyond_lag_gp_modes.py` | post-run: GP-closure ACF beyond the fitted lags at M = 10/25 and the matched reference curve (`beyond_lag_M/acf_*.npz`), the GP side of the Figure 4.3(c) discussion |

The truth physics shared with the companion case (`modal_closure.numerics`,
`modal_closure.truth`, `modal_closure.experiment.spatial_statistics`) is
imported from `../linear_wave/`, and the GP mean + log parameterization from
`../algorithms/` — not copied.

## Why the EKI engine is case-local (`eki_resampling.py`)

The shared engine `algorithms/eki.py` covers every other case through its
injection hooks, but **not** this run, for two reasons stated in that file's
header and repeated here:

1. **Member resampling** (`resample_failures=True`): a member failed by the
   stationarity gate is replaced by a copy of a *surviving* member —
   parameters and output together, from a dedicated generator (seed
   987654321) — after the evaluation and *before* the objective / covariances
   / Kalman update, with the recorded pre-update ensemble rewritten.  The
   shared hooks cannot express this: `sentinel_row_fn` swaps failed *output*
   rows for one fixed penalty row and never touches theta; `post_update` runs
   after the Kalman update.
2. **The frozen objective solve**: the archived `phi_history`
   (1.41e10 → 80.5698) was produced by solving against Gamma with a relative
   1e-8 diagonal inflation; the shared engine deliberately computes Phi with
   the exact Gamma (release decision 2026-08-25).  The difference is
   ≤ 1e-8 relative — but the frozen numbers are reproduced with the
   construction that made them, not an approximation of it.

The file is the archived engine of the frozen run (byte-identical in numerics
to the run's `code/eki.py`), not a fork of the shared engine.

## How to run

Run **from this folder** (`code_rp/linear_wave_gp/`), with a Python that has
numpy and scipy:

```bash
cd linear_wave_gp

# The thesis run (CPU-heavy: 393 s on 188 server cores; hours on a laptop).
python run_gp.py --model gp --workers 8 --n-g 100

# Post-run tools (point them at the results directory the driver wrote):
python long_run_validation.py results/gp_T1000_NG100
python dump_longrun_arrays.py results/gp_T1000_NG100
BUNDLE_DIR=results/gp_T1000_NG100 \
  MODAL_CLOSURE_M_MODES=10 python beyond_lag_gp_modes.py acf_M10.npz
BUNDLE_DIR=results/gp_T1000_NG100 \
  MODAL_CLOSURE_M_MODES=25 python beyond_lag_gp_modes.py acf_M25.npz
python beyond_lag_gp_modes.py acf_ref.npz --reference
```

Flags of `run_gp.py`: `--model {gp,delta}` (delta = the per-mode linear
baseline, not a thesis result — see the driver header), `--workers N`
(1 = serial), `--n-g N`, `--out DIR`.

Environment knobs (all read at import, so set them before launching —
spawned workers re-import and must see the same values):

| variable | effect |
|---|---|
| `STOCHASTIC_N` | reference component count (frozen thesis value 10000) |
| `GP_N_ITER` | EKI iteration cap (default 30) |
| `LONG_T` | window of the two long-run tools (`beyond_lag_gp_modes.py` fixes its window at 1000 s) |
| `LONG_WORKERS` | workers of all three post-run tools |
| `MODAL_CLOSURE_M_MODES` | mode grid of `beyond_lag_gp_modes.py` |
| `BUNDLE_DIR`, `SIGMA_SCALE` | bundle location / noise rescale of `beyond_lag_gp_modes.py` |

**Smoke test (minutes, laptop)** — same pipeline, reduced sizes; the numbers
are NOT comparable with the thesis run:

```bash
STOCHASTIC_N=1000 GP_N_ITER=2 python run_gp.py --model gp --workers 4 --n-g 2 --out results/smoke
```

## What gets saved where

`run_gp.py` writes into `--out` (default `results/gp_T1000_NG<n_g>/`):

- `bundle.npz` — `y`, `var_ref`, the full `theta_history` (latent), the final
  ensemble (latent and physical), `theta_mean`/`theta_sd`,
  `objective_history`, `final_outputs`.
- `summary.json` — settings, `phi_history`/`phi_final`/`phi_over_q_half`,
  sentinel and resampling counts (total and per iteration), parameter labels,
  `theta_mean`/`theta_sd`, `y`/`var_ref`, the N_G calibration ratios, the
  linearised-closure stability report, wall time.

`long_run_validation.py` writes `longrun_<results-dirname>.json` (into this
folder, matching the archived layout).  `dump_longrun_arrays.py` writes
`longrun_arrays.npz` next to the bundle.  `beyond_lag_gp_modes.py` writes the
`.npz` named on its command line.

No figures are produced anywhere in this release.

## Seed conventions

Identical to the companion case's disjoint blocks: observation y 22001; the
`N_Gamma` records 30000+i; forward common random numbers keyed
`(40000, iteration, replicate)` — the member index is deliberately omitted,
which is what makes the noise common across members; observation
perturbations 50000; initial ensemble 60000 (60001 drives the engine's
generator); N_G calibration probes 80000; resampling donors 987654321.
Long-run/beyond-lag tools: references 90000+i, members 91000+j, mean run
92000.  The truth frequency draw is seed 7.

## Frozen-run anchor numbers (for cross-checking a recompute)

From `results/gp_T1000_NG100_9nodes_resample/` (`summary.json`,
`longrun_*.json`, `beyond_lag_M/`), also archived under
`result_rp/chapter4_linear_wave/`:

| quantity | value |
|---|---|
| objective `Phi` | **1.41e10 → 80.5698 = 4.24 × q/2** (q = 38; 30 updates, stop reason `n_iter`) |
| sentinel evaluations / resampled | 45 / 45 (per iteration: 10, 24, 10, 1, then 0) |
| `sqrt(sigma_m)` profile | 0.167 … **1.047 at mode 6** … 0.373 |
| linearised closure | trace −0.338, det min 3.57, stationary True |
| long-run (40 members, 1000 s) | Var(eta) **1.109** vs ref **0.995**; H_s **4.213** vs **3.989 m**; excess kurtosis **+0.362** vs **−0.028** |
| GP M10 ACF beyond 3 s | peak **0.817 at 6.20 s** (`beyond_lag_M/acf_M10.npz`) |

A recompute on different hardware reproduces the random streams exactly but
the floating-point arrays to ~1e-15 relative (libm/BLAS builds), so compare
at quoted precision, not bit for bit.

## Changes vs the sources

Sources: the frozen snapshot
`5.GP_wave_closure/results/gp_T1000_NG100_9nodes_resample/code/` (authoritative;
`eki.py`, `gp_closure.py`, `gpr.py`, `parameterization.py`, `run_gp.py`,
`stochastic_truth.py`) and the live `5.GP_wave_closure/` extras
(`beyond_lag_gp_modes.py`, `dump_longrun_arrays.py`, `plot_long_run.py`).
The snapshot's `eki.py`, `gp_closure.py`, `run_gp.py` and `stochastic_truth.py`
are byte-identical to the live copies (`eki.py` = `eki_with_resampling.py`).

- Snapshot `gpr.py` and `parameterization.py` **dropped**: the case imports
  `algorithms.gpr.make_gp_mean_from_theta` (drop-in for the origin's
  `make_gp_mean_function` — same packed layout, same RBF kernel, same
  `(obs_noise^2 + 1e-8)` solver jitter, same `_gp_params` contract; verified
  numerically identical at import-smoke time) and
  `algorithms.parameterization` (numerics identical to the origin file).
- `eki.py` shipped as `eki_resampling.py`, case-local, numerics untouched
  (see "Why the EKI engine is local").
- `run_gp.py`: engine import now `from eki_resampling import run_eki`;
  sys.path bootstrap re-targeted to the release layout (code_rp root +
  `../linear_wave`), the origin's source-tree and `/root` server paths
  removed; the **`--model delta` baseline branch re-wired** to the released
  modal_closure API (`log_prior_mean_parameters()`, `initial_ensemble(rng)`)
  because the origin's two helpers existed only on the run machine — the
  thesis `--model gp` path is numerically untouched; docstring's stale
  "20 iterations" corrected to 30 (the code always defaulted to 30).
- `gp_closure.py` / `stochastic_truth.py`: import bootstraps re-targeted to
  the release layout; `gp_closure.py`'s stale five-node docstring arithmetic
  updated to R = 9; all numerics untouched.
- `plot_long_run.py` shipped as `long_run_validation.py` with **all figure
  code removed** (mixed file: it computed and wrote the thesis-quoted
  `longrun_*.json`); the figure-only curves it discarded are saved by
  `dump_longrun_arrays.py` instead.
- **Not shipped**: `plot_gp_result.py` (pure figure script, reads
  bundle/summary and draws — no data writer); `run_gp_modes.py` (a shadow
  driver for *re-fitting* the GP closure on other mode grids; its M = 25
  re-fit attempt left only empty result directories, no thesis number depends
  on it, and the Figure-4.3(c)-side GP curves come from
  `beyond_lag_gp_modes.py`, which transplants the frozen M = 10 fit without
  re-fitting); the `results/` trees and figure PNGs.
- All comments and docstrings are in English (the sources already were);
  numerics of every retained code path are untouched.
