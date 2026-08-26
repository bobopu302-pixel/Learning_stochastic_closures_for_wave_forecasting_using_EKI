# Learning stochastic closures for coastal wave forecasting using ensemble Kalman inversion — code release

Code release accompanying the MSc dissertation. One statistics-based
calibration framework — observation vector `y` built from long-run statistics,
noise covariance `Gamma` estimated from independent reference records, forward
map `G(theta)` averaged over `N_G` common-random-number runs, parameters
estimated by ensemble Kalman inversion (EKI) — is instantiated on four systems
of increasing physical realism: the additive-noise **Lorenz-63** system and the
two-scale **Lorenz-96** system (reproductions of Schneider, Stuart & Wu 2021;
thesis Chapter 3), a **broadband linear wave field** closed by a
dispersion-locked stochastic modal model (Chapter 4), and a **stochastically
forced variable-depth KdV** coastal twin whose terrain-law noise parameters and
nonlinear tendency are learned in a stepwise ladder (Chapter 5). All four
cases run the same 2026-08-23 algorithm specification (log-space positive
parameters, CRN, `N_G` forward averaging, reference-only `Gamma`, relative-`Phi`
stopping, final-ensemble-mean reporting) and share a single algorithm layer,
`algorithms/`, so a convention is defined once and used everywhere.

## Repository map

```
code_rp/
  algorithms/        shared algorithm layer: EKI engine, Gamma / N_G estimators,
                     GP conditional mean, log parameterization, statistics
  lorenz63/          noisy Lorenz-63 reproduction, four closure fits   (Ch. 3)
  lorenz96/          two-scale Lorenz-96 case (a), GP closure          (Ch. 3)
  linear_wave/       dispersion-locked stochastic modal closure        (Ch. 4)
  nonlinear_wave/    coastal vKdV stochastic-closure ladder S1 -> S2   (Ch. 5)
  requirements.txt   runtime dependencies (numpy, scipy, numba)
```

## Install

Python >= 3.11 (the release was verified on CPython 3.12):

```
pip install -r requirements.txt
```

`numpy` is needed everywhere; `scipy` by the wave cases and (lazily) by the
shared EKI engine's objective evaluation, so every case needs it at run time;
`numba` only by the two Lorenz simulators (kernels compile on first call and
are cached).

## Running the cases

Each case is run **from its own folder** and documented in its own README:

| Case | Entry point | Details |
|---|---|---|
| Lorenz-63 | `python run_spec.py --workers N` (add `--quick` for a smoke test) | [`lorenz63/README.md`](lorenz63/README.md) |
| Lorenz-96 | `python run_spec.py --workers N` (add `--quick` for a smoke test) | [`lorenz96/README.md`](lorenz96/README.md) |
| Linear wave | `python run_closure.py --mode recompute --output results/modal_closure` | [`linear_wave/README.md`](linear_wave/README.md) |
| Coastal vKdV | `bash v3_spec_chain.sh P` (reads `SW_*` env at import — see the README) | [`nonlinear_wave/README.md`](nonlinear_wave/README.md) |

The production configurations are CPU-heavy (they were run on rented many-core
servers); every case ships a minutes-scale smoke path (`--quick` / `--smoke`)
that exercises the full pipeline including data saving.

## The shared-algorithms contract

The `algorithms` package contains only case-agnostic code: the EKI engine, the
`Gamma`/`N_G` spec estimators, the GP conditional mean, the log
parameterization, and the raw statistic estimators — nothing that knows what a
wave or an attractor is. Physics, observation-vector assembly, seed tables,
and parallel worker pools live in the case folders and reach the engine only
through the documented injection points of `run_eki` (ensemble evaluator,
failure hooks, stopping rule, clip hook, checkpoint callback). Every hook
defaults to off, so calling `run_eki` with only the original arguments
reproduces the linear-wave engine's ensemble trajectory bit for bit. One
uniform convention holds everywhere: the objective `Phi` is always computed
with the exact `Gamma` (Cholesky whitening — no regularisation ever enters
`Phi`), and the `jitter` setting conditions the Kalman-gain solve only; see
[`algorithms/README.md`](algorithms/README.md) for the full contract.

## Data outputs

Every run saves its full data: observation, reference records, `Gamma`,
ensemble and objective histories, CRN seed sets, final ensembles, validation
arrays — as `.npz` archives plus a `summary.json` per run. The release
produces **no figures** (all plotting was stripped); wherever a figure was the
only holder of derived data, that data is now saved instead, so everything a
thesis figure was drawn from is in the `.npz`/`.json` files. Each case README
has a "what gets saved where" section.

## Provenance

This code accompanies the MSc dissertation; the original development trees
(`1. Reproduce_papers/`, `2.Linear_wave_case/`, `3. KDV_nonlinear_case/`) are
not included. Every Python file starts with a provenance docstring recording
its origin path under `research_project/` and the changes made for this
release (for most library files: comments/docstrings only; the case READMEs
summarise the larger changes, such as rewiring the drivers onto the shared
EKI engine and deleting legacy/plotting code paths).
