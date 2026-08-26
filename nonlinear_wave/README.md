# `nonlinear_wave` — coastal vKdV stochastic-closure case (thesis Chapter 5)

Reproduction package for the Chapter-5 experiments: the stepwise ladder
S1 → S2 on the variable-depth KdV (vKdV) coastal twin.  S1 recovers the
terrain-law noise parameters (φ, q) by EKI; S2 additionally learns the
nonlinear tendency as a free 2-D GP surface (H2 tier) and measures how
identifiable it is.  This release ships **only the 2026-08-23 spec
configuration** — the production run actually reported in the thesis
(2026-08-25): log-space positive parameters, common random numbers, N_G
forward averaging, Γ from independent reference records (var_ref only),
the spec stopping rule, and final-ensemble-mean reporting.

Shared algorithms live in `../algorithms/` (EKI engine, Γ/N_G spec
estimators, GP mean, log parameterization).  The two EKI drivers here
call `algorithms.eki.run_eki`; each entry script puts the `code_rp` root
on `sys.path` itself, so **run everything from this folder**.

---

## IMPORTANT: environment variables are read at import time

`v3_world.py` and `sde_closure_config.py` read `SW_*` / `SDE_*` from the
environment **once, at import**, and `v3_world.ensure_patched()` rebinds
frozen names (`_tma_inputs`, `ModelAConfig`) before the first
`sde_closure_*` import.  A driver imported under the wrong environment
silently runs a *different world* (wrong duration, wrong incident
record, wrong grid).

**Always launch through `v3_spec_chain.sh`, or export exactly the
variables it exports before calling any driver by hand.**  The minimal
spec preamble is:

```bash
export SW_FINE=1 SW_VERSION_DIR=v3spec SW_DURATION_S=6600 \
       SW_SYNTH_PERIOD_S=6600 SW_FORWARD_PATHS=1 SW_REF_NEW_SEA_STATE=1 \
       SW_S1_VARIANT=S1a SW_S1_SUFFIX=_fine OMP_NUM_THREADS=1
```

---

## File map

### Solver stack (deterministic PDE; comments/docstrings-only changes vs origin)

| File | Role |
|---|---|
| `pde_core.py` | Core deterministic operators for the physical-time coastal vKdV model |
| `transparent_boundary_vkdv.py` | Open-boundary solvers and verification tests |
| `high_order_matched_dabc.py` | 4th-order centered KdV operator + matched CN discrete boundary |
| `high_order_variable_depth_dabc.py` | High-order variable-depth vKdV solvers with matched linear DABC |
| `high_order_variable_depth_dabc_study.py` | Isolated high-order linear variable-depth solver validation (provides `_tma_inputs`, three-trace lifting) |
| `high_order_incident_lifting.py` | Three-trace modal incident lifting for the C6/C4–CN discretisation |
| `high_order_implicit_midpoint_candidate.py` | Entropy-C6 nonlinear implicit-midpoint candidate |
| `high_order_nonlinear_candidates.py` | Isolated nonlinear discretisation candidates |
| `high_order_nonlinear_candidate_screening.py` | Nonlinear candidate screening driver |
| `coastal_entropy_midpoint_production.py` | Final physical-time production driver (provides `_analysis_mask`, `_one_sided_psd`) |
| `sea_state_boundary.py` | Exact prescribed finite-depth TMA boundary (incident sea state) |

### Frozen closure layer (numerics untouched; see each file's provenance header)

| File | Role |
|---|---|
| `sde_closure_config.py` | Closure-chain configuration switch (reads `SDE_*` env at import) |
| `sde_closure_context.py` | Shared worker context (`ModelAConfig`, `CoarseModelContext`) |
| `sde_closure_core.py` | Stochastic solver + `GridWhiteNoise` + `terrain_weight` |
| `sde_closure_eki.py` | Frozen EKI conventions: θ layout/decoding, statistics (q=44), worker init |
| `sde_closure_eki_dense.py` | Dense-station statistics (q=111) + dense worker init |
| `sw_linear_core.py` | Linear vKdV core (zero-λ flux solver; GPR reference propagator) |
| `sw_gamma_unified.py` | Unified y/G/Γ convention; spec branch `SW_GAMMA_TERMS=var_ref_only` builds Γ = diag(var_ref) or shrunk full Cov_ref |

### Drivers (the spec chain; rewired onto `algorithms.eki`)

| File | Role |
|---|---|
| `v3_world.py` | Env-driven world patcher (duration, incident period, ref-record mode).  Load-bearing; see the env warning above |
| `sw_truth.py` | Stage 1a: twin truth bundle + deterministic baseline (S1a terrain law) |
| `sw_gpr_reference.py` | Stage 1b: GPR regression reference — residual surface, `t0`/`s0` calibration |
| `sw_ref_records.py` | Stage 1c: N_Γ independent truth reference records → `ref_stats_*.npz` (resumable) |
| `v3_calibrate_ng.py` | Stage 2: N_G calibration (K repeats at two probes) |
| `sw_eki_s1.py` | Stage 3: S1 inversion of (φ, q), q=111, via `algorithms.eki.run_eki` |
| `sw_eki_h.py` | Stage 4: S2 = H2/H3 GP-surface inversion, q=151, θ ∈ R²⁷, checkpoint/`--resume` |
| `v3_s1_finalise.py` / `v3_s2_finalise.py` | Rebuild `summary.json` from the per-iteration audit files when a run is truncated at a fixed iteration count |
| `v3_s1_validation.py` | S1 physical validation: fresh-seed paths at the reported (φ, q) → field + profile data |
| `v3_s2_validation.py` | S2 validation: learned-surface slices, convergence data, fresh-seed paths |
| `v3_s2_val_truth.py` | Same validation paths at the TRUE parameters (`--mode truth`) or with m ≡ 0 (`--mode zero`) |
| `v3_s2_truth_phi.py` | Φ at the true S2 parameters (the measured floor) → `S2_truth_phi.json` |
| `v3_s2_threeway.py` | Observed / inversion / truth / m≡0 comparison + nonlinearity-signal test (data + printed report) |
| `v3_spec_chain.sh` | The production chain: stages [1]–[4] with skip-if-present logic |

---

## The spec chain, stage by stage

All commands are run **from this folder**.  `bash v3_spec_chain.sh P`
(default P = 100) runs stages [1]–[4] end to end and logs to
`results/stepwise/v3spec/v3_spec_chain.log`.  Note that P sets **both**
the EKI ensemble size J (`--members`) and the per-stage process counts;
to run the production J = 100 on fewer cores, keep P = 100 and set
`PROCS=<cores>` in the environment (it overrides the EKI stages' pool
size only).  Every stage skips
itself if its output already exists, so re-launching the script resumes
the chain.  Wall-clock scale: the 2026-08-25 production run used a
rented 48–64-core box; one 6600 s fine-grid realisation ≈ 4 min on one
core, and one EKI iteration evaluates J × N_G = 1000 realisations.

Individual stages (each needs the spec preamble above):

```bash
# [1] truth + GPR reference + 50 reference records (run in parallel)
python sw_truth.py --variant S1a --suffix _fine --paths 1 --overwrite
python sw_gpr_reference.py --paths 1
python sw_ref_records.py --n-ref 50 --paths 1 --processes 50 --resume

# [2] N_G calibration (writes NG_calibration.json; the chain exports
#     SW_N_G from its "N_G_chosen")
python v3_calibrate_ng.py --k 20 --processes 100
export SW_N_G=10        # = N_G_chosen of the production run

# [3]/[4] per branch (diag = the spec; full = shrunk full Cov_ref)
SW_GAMMA_TYPE=diag SW_S1_OUTTAG=_diag \
  python sw_eki_s1.py --config S1a --members 100 --iterations 20 \
      --processes 100 --overwrite
SW_GAMMA_TYPE=diag SW_H_PHIQ_PRIOR=s1 SW_H_OUTTAG=_diag \
  SW_H_S1_SUMMARY=results/stepwise/v3spec/S1a_eki_dense_fine_diag/summary.json \
  python sw_eki_h.py --variant H2 --members 100 --iterations 20 \
      --processes 100 --resume

# Post-processing (not part of the chain script)
python v3_s1_finalise.py --branch diag --iterations 10   # if truncated
SW_N_G=10 python v3_s2_truth_phi.py --processes 10
python v3_s2_finalise.py --branch diag --iterations 10
python v3_s1_validation.py --branch diag --paths 8 --processes 8
SW_N_G=10 python v3_s2_validation.py --branch diag --paths 8 --processes 8
SW_N_G=10 python v3_s2_val_truth.py --branch diag --paths 32 --processes 32
SW_N_G=10 python v3_s2_val_truth.py --branch diag --paths 32 --processes 32 --mode zero
python v3_s2_threeway.py --branch diag
```

Smoke tests: `--smoke` on either EKI driver runs J=8, 2 iterations, a
240 s window.  `sw_eki_h.py --check` runs the degeneration gate (the GP
surface loaded with the true bilinear values must reproduce the twin).

---

## Environment-variable table

| Variable | Default | Meaning |
|---|---|---|
| `SW_FINE` | unset | `1` = production fine grid (3073 cells, dt 0.002, stride 70).  The spec chain requires it; `sw_eki_s1` refuses to run without it |
| `SW_GRID` | unset | `1537` = certification tier (`sw_truth.py` only) |
| `SW_VERSION_DIR` | `v2_fine` | results subfolder under `results/stepwise/` (spec: `v3spec`) |
| `SW_DURATION_S` | 1800 | total simulated seconds per run (spec: 6600 = 600 burn-in + 6000 analysis) |
| `SW_SYNTH_PERIOD_S` | 1800 | incident TMA synthesis period; must be ≥ duration (spec: 6600) |
| `SW_FORWARD_PATHS` | 16 | closure-noise paths per realisation (spec: 1) |
| `SW_REF_NEW_SEA_STATE` | 0 | `1` = each Γ reference record gets its own incident phases + baseline (spec: 1; `sw_ref_records` refuses to run without it) |
| `SW_CRN` | 1 | legacy per-member seed switch in `v3_world.member_seed_root`; the spec drivers use their own CRN scheme and ignore it |
| `SW_N_G` | 1 | N_G forward realisations averaged per EKI evaluation (production: 10, from `NG_calibration.json`) |
| `SW_GAMMA_TYPE` | `diag` | `diag` = Γ = diag(var_ref) (the spec); `full` = shrunk full Cov_ref |
| `SW_GAMMA_SHRINK` | 0.1 | shrinkage intensity of the full-Γ branch (read by `sw_gamma_unified`) |
| `SW_GAMMA_NF` | 12 | N_F diagnostic forward repeats at the prior mean (feeds the var_fwd/var_ref ratio check, not Γ itself) |
| `SW_S1_VARIANT` | — | must be `S1a`; wires the twin baseline before import |
| `SW_S1_SUFFIX` | `` | truth-dir suffix (spec: `_fine`) |
| `SW_S1_OUTTAG` / `SW_H_OUTTAG` | `` | output-dir tag per branch (`_diag`, `_full`) |
| `SW_H_PHIQ_PRIOR` | `uniform` | `s1` = S2 inherits the S1 posterior of (φ, q) as its prior |
| `SW_H_S1_SUMMARY` | spec S1 diag dir | path of the S1 summary the prior is read from |
| `SW_H_PATH_PARALLEL` | 0 | `1` = each closure path is its own pool task (bit-identical; for boxes with ≫ J cores) |
| `SW_INCR_SUB` | 1 | cadence subsampling of the core blocks in the q=151 layout |
| `SDE_H_NODE_BOUND_T0` | 5.0 | GP-node box half-width in units of t0 (pre-registered remedy: 3) |
| `SW_GAMMA_TERMS` | — | **hard-set to `var_ref_only` by the drivers** (spec Γ; no forward term, no floor, no n_eff) |
| `SDE_COARSE_N4`, `SDE_COARSE_DT`, `SDE_OUTPUT_STRIDE`, `SDE_CLOSURE_V3`, `SDE_BASELINE_NPZ`, `SDE_SPLINE_KNOTS` | — | set by the driver preambles from `SW_FINE` / `SW_S1_VARIANT`; do not set by hand |
| `OMP_NUM_THREADS` | — | pin to 1: parallelism is process-level |

---

## Outputs (everything lands under `results/stepwise/<SW_VERSION_DIR>/`)

Stage [1]:

- `truth_S1a_fine/truth_bundle.npz` — times, grids, η paths, gauge series, Hs profile, spread target, true σ(y)
- `truth_S1a_fine/baseline_data.npz` — deterministic baseline (key layout consumed via `SDE_BASELINE_NPZ`)
- `truth_S1a_fine/metadata.json` — seeds, grid, truth parameters
- `GPR_reference/{reference_bundle.npz, reference_baseline.npz}` — full-grid fields for residual extraction
- `GPR_reference/calibration.json` — `t0_tendency_scale`, `s0_slope_scale` (consumed by `sw_eki_h`)
- `GPR_reference/surface.npz` — bin grid, bin means/counts, σ of the mean
- `truth_S1a_fine/ref_records/ref_stats_{standard,dense,incr}.npz` — (N_Γ, q) reference statistics + seeds
- `truth_S1a_fine/ref_records/{records_cache.npz, metadata.json}` — resumable per-record cache + provenance

Stage [2]: `NG_calibration.json`, `NG_calibration_stats.npz`.

Stages [3]/[4], per branch `<br>` ∈ {diag, full}:

- `S1a_eki_dense_fine_<br>/iter_NNN.npz` — per-iteration audit: pre-update `thetas`, `g_matrix` (penalty rows substituted), per-member `phis`, `phi_ensemble`
- `S1a_eki_dense_fine_<br>/summary.json` — spec report (final ensemble mean ± sd in model coordinates), recovery vs truth, validation, Γ diagnostics
- `gamma_fwd_cache/fwd_stats_*.npz` — cached N_F prior-mean repeats (shared across re-runs)
- `H2_eki_<br>/iter_NNN.npz`, `H2_eki_<br>/checkpoint.npz` (see resume below)
- `H2_eki_<br>/gamma_unified_diag.json`, `H2_eki_<br>/gamma_cache_unified_p1_nf12.npz`
- `H2_eki_<br>/summary.json`

Post-processing:

- `S2_truth_phi.json` — Φ at the true parameters + per-family split
- `S1a_eki_dense_fine_<br>/analysis/{validation_fields.npz, validation_profiles.npz}`
- `H2_eki_<br>/analysis/{validation_fields.npz, validation_profiles.npz, s2_surface_slices.npz, s2_convergence.npz}`
- `H2_eki_<br>/analysis/validation_fields_{truth,zero}.npz`
- `H2_eki_<br>/analysis/{threeway_data.npz, threeway_pdf_spectra.npz}`

Note on iteration counts: when a run hits the iteration cap (the spec
stop rule cannot trigger under the Monte-Carlo fluctuation of Φ at
N_G = 10), the engine evaluates the final post-update ensemble once
more, so one extra `iter_NNN.npz` appears beyond the cap and the
summary reports its Φ under `phi_ensemble_final_evaluation`.  The
production summaries were assembled by the `v3_*_finalise.py` scripts
at a declared truncation point.

---

## Seed table (all disjoint by construction)

| Stream | Seed scheme |
|---|---|
| Official incident phases (truth + EKI forward baseline) | boundary seed `20260801` |
| Truth closure-noise path m | `[20260801, m]` |
| Γ reference record r: incident phases | boundary seed `202609010 + r` |
| Γ reference record r, path p: closure noise | `[20260901, r, p]` |
| N_G calibration realisation k at probe j | `[27000, j, k]` |
| S1 EKI initial ensemble + observation perturbations | `default_rng([42, 72, ord('B'), 7])` (base.EKI_SEED = 42) |
| S1/S2 Γ diagnostic repeats k | `(EKI seed, 2999, k)` |
| CRN forward seeds, iteration n, realisation a, path p | `(EKI seed, 3000 + n, a)` + `[p]` — shared by every member within an iteration, fresh set each iteration |
| S2 EKI initial ensemble + perturbations | `default_rng([20260814, ord('2')])` |
| S1 twin validation paths (inside `sw_eki_s1`) | `[20260810, m]` |
| S1 posterior validation (`v3_s1_validation`) | `[33000, k]` |
| S2 posterior validation (`v3_s2_validation`, `v3_s2_val_truth`) | `[34000, k]` |
| Φ at truth (`v3_s2_truth_phi`) | `(35000, 0, k)` |
| GPR reference bundle paths | `[20260801, m]` (identical realisation to the twin, by design) |

Failed forward runs: a member fails only when **all** of its N_G
realisations return non-finite fields; its statistics row is replaced by
the declared penalty row `y + 10·sqrt(diag(Γ))` before entering the
objective and the Kalman update (implemented via the engine's
`failed_mask_fn` / `sentinel_row_fn` hooks).

---

## Checkpoint / resume (`sw_eki_h.py`)

Long S2 runs write `checkpoint.npz` twice per iteration (pre-update from
the evaluator, post-update from the engine's `iteration_callback`) with:
ensemble, per-member Φ history, means, best member, failure counters,
`next_iteration`, and the rng bit-generator state.  `--resume` restores
all of it and calls the engine for the remaining iterations, with the
evaluator offsetting the iteration index so CRN seeds and `iter_NNN.npz`
names continue exactly where the run stopped.  The stop-rule streak
restarts on resume (as in the origin).  `sw_ref_records.py --resume`
resumes from its per-record cache, or rebuilds its state from the
`ref_stats_*.npz` files when only those were shipped.
