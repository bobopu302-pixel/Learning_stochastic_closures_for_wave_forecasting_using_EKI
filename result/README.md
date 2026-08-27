# Thesis results data archive (result_rp)

Data files backing every quantitative result cited in the MSc thesis
(EKI stochastic closures: Lorenz systems -> linear wave field -> variable-depth KdV).

`thesis_results_inventory.md` (the structured item list extracted from the thesis).

Data only: `.npz` / `.json` / `.csv` / `.md` / `.log` run logs. No images, no PDFs, no code
(code paths are named in the chapter READMEs where a thesis statement is code-defined).
Total size: 220.2 MB (58 files).

## Top-level map

| Folder | Backs | Contents | Size |
|---|---|---|---|
| `chapter3_verification/` | Ch 3 (Lorenz 63/96), fig 2.2 of Ch 2, Ch-3 halves of Appendix A | `lorenz63/` four spec fits (ODE/SDE x Case A/B) + summary + sensitivity table + truth density cache + protocol log; `lorenz96/` two spec fits + summary + the truth-run bundle (101 MB); `FREEZE_20260824.json` sha256 freeze record | 105.0 MB |
| `chapter4_linear_wave/` | Ch 4 (random wave field), Ch-4 rows of Appendix A | `gp_T1000_NG100_9nodes_resample/` the frozen thesis-cited GP-closure fit (bundle, summary, long-run validation, beyond-lag ACFs, run log, run README); `modal_closure/` and `modal_closure_M25/` the companion per-mode linear-damping runs behind fig 4.3(c) | 70.9 MB |
| `chapter5_nonlinear_wave/` | Ch 5 (vKdV), vKdV rows + D5 + A.7 of Appendix A | `shared/` twin truth, reference records, N_G calibration, forward-repeat caches; `s1/` step-one summary; `s2/` step-two summary + curated figure/analysis npz + unified Gamma; `numerics_convergence/` solver convergence study (Appendix A.7) | 44.3 MB |
| `thesis_results_inventory.md` | - | the item-by-item inventory this archive is checked against | 24 KB |

Chapters 2 and 6 and the Abstract have no folders of their own: Chapter 2's only data-backed
item (fig 2.2, EKI convergence illustration) reads
`chapter3_verification/lorenz63/spec_fixed_sde_full.npz`, and every Chapter 6 / Abstract number
is a recap of a chapter item and traces to the same files (mapping in the chapter READMEs).

Each chapter folder has its own `README.md` with the full thesis-item -> file -> quantity
mapping, run provenance, deliberately-not-copied lists, and issues.

## Coverage status (verified 2026-08-26)

Method: every inventory item was traced to its mapped file(s); all mapped files exist and are
non-empty; no image files and no Chinese characters anywhere in the archive; 4-6 items per
chapter were spot-checked by re-reading the copied npz/json with the project venv Python and
comparing against the thesis-quoted values (all matched at the printed rounding; the full
spot-check lists are in the chapter READMEs and in the verification notes below).

| Chapter | Items in inventory | Mapped to archived data | Notes |
|---|---|---|---|
| Ch 2 Methodology | 4 | 1 of 1 data-backed | fig 2.1 (TikZ schematic), fig 2.3 (synthetic GP illustration) and tab:statistics (conceptual) need no data by design |
| Ch 3 Verification | 12 (3 figs, 1 table, C3.1-C3.8) | 12 of 12 | one sub-claim PARTIAL (T_y=80 scatter, see Issues #1); priors/dt are code constants (Issues #2) |
| Ch 4 Linear wave | 14 (3 figs, 1 table, C4.1-C4.10) | 14 of 14 | two sub-elements code-defined only (Issues #3, #4); C4.3 slopes recomputed not stored (Issues #5) |
| Ch 5 vKdV | 18 (6 figs, C5.1-C5.12) | 18 of 18 | physical constants code-defined (Issues #6) |
| Ch 6 + Abstract | 5 recap items | 5 of 5 | all trace to Ch 3-5 files; each recap number re-verified in the underlying file |
| Appendix A | 11 (4 tables, 4 figs, CA.1-CA.3) | 11 of 11 | CA.3 convergence data located and added 2026-08-26 (`chapter5_nonlinear_wave/numerics_convergence/`); D5's q=111 requirement reproduced from archived caches (see chapter5 README) |
| Appendix B | 2 | 2 of 2 | the archive itself mirrors the named paths; environment claim see Issues #7 |

Spot-checks performed at verification (examples; all against thesis rounding):
- Ch 3: Case A SDE alpha 9.9443+/-0.0214, Phi 1.7325; Case A ODE 72.78 vs 65.10; Case B Phi 0.2332 / 46.71;
  L96 sqrt(sigma) 0.5620+/-0.0239, Phi 35.78 vs ODE 33.46, <X_k> 2.58, slow var 12.59;
  sensitivity rows (-7.15/+9.35, -4.22/-4.35/+2.47/-3.13, -7.88/-7.53/+5.00/-5.72, v3 block, v5/tau ~0)
- Ch 4: Phi 1.41e10 -> 80.5698 (last ten 80.6-115.4); band L1 6.9%, total -2.04%; worst xcorr -6.75 sd;
  Var(eta) 1.1095 vs 0.9947, H_s 4.2132 vs 3.9894, ex. kurt +0.3617 vs -0.0279; sqrt(sigma_m) 0.167->1.047->0.373;
  var_ref span 7.88 orders; resampling gate [10,24,10,1,0,...]; beyond-lag peaks 0.821@6.2s (M10), 0.418@16.2s (M25)
- Ch 5: s1 recovery phi 0.011987+/-0.000081 (-3.3%), xi 2.654+/-0.162 (+12.9%), history ending 11.49/11.87/11.81;
  s2 phi 0.012046 (-2.9%), xi 2.093 (-10.9%), Phi 4.07e5 -> 568.8 = 1.084x truth level 524.775 (q/2 = 75.5);
  surface corr -0.70, amplitude 0.31, p_d +0.719+/-0.038; per-family per-stat split (0.62/2.94/16.22/0.71/2.49/32.57/3.00
  vs 2.91/0.66/0.99/1.01/3.44/9.12/4.90); signal fractions skew 17.4%, kurt 62.5%, sigma 3.45x; residuals 54>2sd, 29>3sd;
  Hs 0.302->0.590, max|eta| 0.737, gauges 0/500/1000/2000/3000/4000 m; A.7 orders 3.550/3.898/3.973 and 3.650/3.924/3.981

## ISSUES (aggregated, honest list)

1. PARTIAL (Ch 3, C3.8): "T_y=80: learned sigma scattered 1.96-13.23 over 20 windows" has no raw
   per-window data anywhere in `1. Reproduce_papers/` (searched result_data, spec_runs, code,
   docs). The numbers survive only as text in
   `chapter3_verification/lorenz63/first_paper_protocol.json` (evidence_note) and in
   `1. Reproduce_papers/README.md`. To regenerate: rerun the L63 fixed-SDE fit at T_y=80 over 20
   independent windows under the pre-spec single-window convention.
2. Ch 3 (C3.1/C3.5): priors (alpha~U(1,20), sqrt(sigma)~U(0.1,15), L96 node/hyper priors) and
   dt = 0.002 / 0.005 are constants in the driver code
   (`1. Reproduce_papers/Lorenz63/code/spec_l63.py`, L96 spec driver), not stored in any data
   file. All other settings are in the archived summary.json files.
3. Ch 4 (C4.9): the "~38% dissipative prior mass" scalar exists only as a comment in the source
   code snapshot `5.GP_wave_closure/results/gp_T1000_NG100_9nodes_resample/code/gp_closure.py`
   (lines 150-158); evidently a one-off Monte-Carlo estimate, no data artifact.
4. Ch 4 (C4.10): the reference amplitude construction (Gaussian envelope f_p = 1.0 Hz,
   Delta_f = 0.35 Hz, unit total variance) is defined by `code/stochastic_truth.py` in the same
   snapshot; its observable consequence (reference Var(eta) = 0.995 ~ 1 m^2) IS archived in
   `longrun_gp_T1000_NG100_9nodes_resample.json`.
5. Ch 4 (C4.3): the closure slopes at the origin (-0.049, -0.286) are not stored as scalars; they
   are recomputed from `bundle.npz` by the fig script (needs `common/code/gpr.py`); their sum is
   cross-checked by the archived `summary.json:linear_trace` = -0.338.
6. Ch 5 (C5.1): h_ref = 15 m, a_ref = 0.15 m, lambda_ref = 150 m, c_ref = 12.13 m/s,
   eps = mu = 0.01 and the beta_C7 depth profile are code constants (`pde_core.py`,
   `high_order_variable_depth_dabc.py`); `shared/truth_metadata.json` records grid/seed/noise.
   Figures 5.1 and 5.3 recompute the depth profile from code at plot time.
7. Appendix B environment claim: the thesis states NumPy 2.4.4; the local project venv used for
   this verification carries NumPy 2.4.6 (Python 3.12.10, SciPy 1.17.1, Matplotlib 3.10.9,
   Numba 0.65.1 all match). The 2.4.4 figure may refer to the rented-server environment of the
   vKdV runs; the user should confirm which environment the statement describes.
8. Appendix A D5 provenance note (resolved but worth knowing): the "N_G >= 7 for q=111" figure is
   not stored in any source json; it lives in the source chain log `v3spec/v3_spec_chain.log` and
   is exactly reproducible from the archived
   `chapter5_nonlinear_wave/shared/gamma_fwd_cache/` + `shared/ref_records/ref_stats_dense.npz`
   (ceil(5 x max variance ratio) = 7; verified 2026-08-26). "N_G >= 20 for q=151" is stored in
   `s2/gamma_unified_diag.json`.

No file over 300 MB was needed; the largest archived file is the L96 truth bundle (101 MB).
Larger raw artifacts deliberately left at source (all under their original paths, with reasons)
are listed in each chapter README's "not copied" section.
