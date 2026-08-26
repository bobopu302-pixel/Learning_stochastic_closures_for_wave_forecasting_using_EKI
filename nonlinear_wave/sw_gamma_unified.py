"""Unified y / G / Gamma convention for the vKdV (Chapter 5) EKI drivers.

Origin: 3. KDV_nonlinear_case/sw_gamma_unified.py
Changes vs origin: comments/docstrings only (this provenance header added).

Implements ``4. Dissertation/Dissertation_writting/UNIFY_y_G_Gamma_spec.md``
(author decisions of 2026-08-16) WITHOUT touching the frozen
sde_closure_* files:

  y      = stats(official truth record, full analysis window)   -- once
  G(θ)   = stats(one simulation of θ, same window, same #paths as y)
  Γ      = diag( var_ref + var_fwd + floor² )
             var_ref : sample variance (ddof=1) of stats over N_R
                       INDEPENDENT truth records (each record = its own
                       set of N_PATHS_REF paths, fresh seeds)  [method a]
             var_fwd : sample variance (ddof=1) of stats over N_F forward
                       evaluations at a representative θ
             floor_i = max(f_rel·|y_i|, f_abs_family_i),  f_rel = 0.05

Design decisions recorded here (see UNIFY spec §2, Ch5 row and the
project memory):
  * forward path count is raised to the truth's 16 paths
    (SW_FORWARD_PATHS, default 16) so G and y use the same #paths;
  * NO block normalisation (the dense-block ×N_DENSE/8, ×N_DENSE/5
    inflation of build_gamma_dense / build_gamma_incr) unless the
    author re-enables it via SW_GAMMA_BLOCK_NORM=1 -- pending decision;
  * the block-split reference term (var/N_BLOCKS) is retired.

Reference records are produced by ``sw_ref_records.py`` and stored as
``<truth_dir>/ref_records/ref_stats_<layout>.npz`` (stats only, small).

Public API
----------
family_layout(layout) -> list[(name, start, stop)]
floor_vector(observation, layout, f_rel=0.05, f_abs=None) -> floors
build_gamma_unified(observation, ref_stats, fwd_stats, layout,
                    f_rel=0.05, f_abs=None, block_norm=False)
    -> (gamma[q,q], diagnostics dict)
acceptance_line(diag) -> str      # the one-line printout the spec asks for
summary_fields(diag)  -> dict     # keys to merge into summary.json
"""

from __future__ import annotations

import os

import numpy as np

# ---------------------------------------------------------------------
# statistic-vector layouts (must mirror the frozen compute_* functions)
# ---------------------------------------------------------------------
# frozen sde_closure_eki.compute_statistics (q = 44):
#   Hs 8 | band log-power 15 (5 gauges x 3 bands) | skew 5 | kurt 5 |
#   deviation rms 5 | demeaned ACF 6 (3 gauges x 2 lags)
# sde_closure_eki_dense.compute_statistics_dense (q = 2*N_DENSE + 31):
#   Hs N_DENSE | bands 15 | skew 5 | kurt 5 | dev-rms N_DENSE | ACF 6
# sw_eki_s1.compute_statistics_incr (q = 3*N_DENSE + 31):
#   ... + increment std N_DENSE

F_REL_DEFAULT = 0.05
# family absolutes (spec 1.3): near-zero statistics need a non-zero
# floor.  Values are in the units of each block.
F_ABS_DEFAULT = {
    "hs": 0.004,          # m   (5% of a typical 0.08 m nearshore Hs err)
    "band": 0.05,         # log10 power
    "skew": 0.05,
    "kurt": 0.05,
    "devrms": 0.003,      # m
    "acf": 0.03,          # normalised correlation
    "incr": 0.0003,       # m   (one-save-step increment std)
}


def family_layout(layout: str, n_dense: int | None = None):
    """Return [(family, start, stop), ...] for a statistic layout."""
    if layout == "standard":
        n = 8
        parts = [("hs", n), ("band", 15), ("skew", 5), ("kurt", 5),
                 ("devrms", 5), ("acf", 6)]
    elif layout in ("dense", "incr"):
        if n_dense is None:
            raise ValueError("n_dense required for dense/incr layouts")
        parts = [("hs", n_dense), ("band", 15), ("skew", 5), ("kurt", 5),
                 ("devrms", n_dense), ("acf", 6)]
        if layout == "incr":
            parts.append(("incr", n_dense))
    else:
        raise ValueError(f"unknown layout {layout!r}")
    out, start = [], 0
    for name, size in parts:
        out.append((name, start, start + size))
        start += size
    return out


def infer_layout(q: int, n_dense: int) -> str:
    if q == 44:
        return "standard"
    if q == 2 * n_dense + 31:
        return "dense"
    if q == 3 * n_dense + 31:
        return "incr"
    raise ValueError(f"cannot infer layout from q={q}, n_dense={n_dense}")


def floor_vector(observation: np.ndarray, layout: str,
                 n_dense: int | None = None,
                 f_rel: float = F_REL_DEFAULT,
                 f_abs: dict | None = None) -> np.ndarray:
    """floor_i = max(f_rel*|y_i|, f_abs_family)   (spec 1.3, uniform)."""
    f_abs = dict(F_ABS_DEFAULT, **(f_abs or {}))
    y = np.asarray(observation, dtype=float)
    floors = np.empty_like(y)
    for name, a, b in family_layout(layout, n_dense):
        floors[a:b] = np.maximum(f_rel * np.abs(y[a:b]), f_abs[name])
    return floors


def effective_dof_factors(ref_stats: np.ndarray, layout: str,
                          n_dense: int | None = None) -> dict:
    """Per-family over-counting factor n/n_eff, MEASURED from the
    reference records (spec addendum 2026-08-20).

    A diagonal Gamma treats a family's n components as independent
    evidence; for spatially dense blocks (stations 100 m apart) they are
    strongly correlated, which INFLATES the block's combined precision
    by n/n_eff with n_eff = n / (1 + (n-1)*rho_bar) (equicorrelation
    model), rho_bar = mean off-diagonal correlation of the family's
    components ACROSS the independent reference records.  Measured on
    the vKdV twin: hs/devrms/incr ~ 10-12x; band/ACF ~ 1.0 (the
    internal control that N_R = 20 does not bias rho_bar high).
    Factors are clamped to >= 1.
    """
    ref = np.asarray(ref_stats, dtype=float)
    factors = {}
    for name, a, b in family_layout(layout, n_dense):
        X = ref[:, a:b] - ref[:, a:b].mean(axis=0)
        keep = X.std(axis=0, ddof=1) > 1e-14
        n = int(b - a)
        if keep.sum() < 3 or n < 2:
            factors[name] = 1.0
            continue
        C = np.corrcoef(X[:, keep], rowvar=False)
        m = C.shape[0]
        rho = float((C.sum() - np.trace(C)) / (m * (m - 1)))
        n_eff = n / (1.0 + (n - 1) * max(rho, 0.0))
        factors[name] = float(max(n / n_eff, 1.0))
    return factors


def build_gamma_unified(
    observation: np.ndarray,
    ref_stats: np.ndarray,
    fwd_stats: np.ndarray,
    layout: str,
    n_dense: int | None = None,
    f_rel: float = F_REL_DEFAULT,
    f_abs: dict | None = None,
    block_norm: bool | None = None,
    neff_correction: bool | None = None,
    gamma_type: str | None = None,
):
    """Gamma per the unified spec + diagnostics.

    gamma_type (env SW_GAMMA_TYPE, default "diag"):
      "diag"  diagonal  var_ref + var_fwd + floor^2  (x n_eff factor)
      "full"  full      Cov_ref + Cov_fwd + diag(floor^2); the floor term
              makes it positive definite for any N (sample covariances of
              N records have rank <= N-1 << q); n_eff is NOT applied
              (it is the diagonal surrogate of the correlations that the
              full matrix encodes directly); block_norm ignored.
    N_Gamma: the thesis uses one repeat count N_Gamma = N_R = N_F.

    ref_stats : (N_R, q) stats of independent truth records
    fwd_stats : (N_F, q) stats of forward repeats at a representative θ
    block_norm: apply the legacy dense-block inflation (default: env
                SW_GAMMA_BLOCK_NORM, off unless "1")
    """
    y = np.asarray(observation, dtype=float)
    ref = np.asarray(ref_stats, dtype=float)
    fwd = np.asarray(fwd_stats, dtype=float)
    if ref.ndim != 2 or fwd.ndim != 2 or ref.shape[1] != y.size \
            or fwd.shape[1] != y.size:
        raise ValueError("ref_stats/fwd_stats must be (N, q) with q=len(y)")
    if ref.shape[0] < 3 or fwd.shape[0] < 3:
        raise ValueError("need >=3 reference and >=3 forward repeats")

    # --- EKI_algorithm_spec_2026-08-23: Gamma = diag(var_ref) only ------
    # gamma_terms = "var_ref_only" drops the forward term, the floor and
    # the n_eff correction; the model's own fluctuation is removed by
    # averaging N_G forward realisations inside G instead.
    terms = os.environ.get("SW_GAMMA_TERMS", "ref_fwd_floor").lower()
    if terms == "var_ref_only":
        want_full = os.environ.get("SW_GAMMA_TYPE", "diag").lower() == "full"
        var_ref = np.var(np.asarray(ref_stats, dtype=float), axis=0, ddof=1)
        var_fwd = np.var(np.asarray(fwd_stats, dtype=float), axis=0, ddof=1)
        ratio = var_fwd / np.where(var_ref > 0, var_ref, np.inf)
        fam = family_layout(layout, n_dense)
        d = {
            "y_convention": "single_window",
            "forward_convention": "mean_of_N_G",
            "gamma_terms": "var_ref_only",
            "gamma_type": "diagonal",
            "gamma_ref_method": "independent_repeats",
            "n_gamma": int(np.asarray(ref_stats).shape[0]),
            "n_ref_repeats": int(np.asarray(ref_stats).shape[0]),
            "n_fwd_repeats": int(np.asarray(fwd_stats).shape[0]),
            "q": int(np.asarray(observation).size),
            "gamma_diag_min": float(var_ref.min()),
            "gamma_diag_median": float(np.median(var_ref)),
            "gamma_diag_max": float(var_ref.max()),
            "var_ref_rel_error": float(np.sqrt(
                2.0 / (np.asarray(ref_stats).shape[0] - 1))),
            "ratio_min": float(np.nanmin(ratio)),
            "ratio_median": float(np.nanmedian(ratio)),
            "ratio_max": float(np.nanmax(ratio)),
            "ratio_by_family": {n: float(np.nanmax(ratio[a:b]))
                                for n, a, b in fam},
            "N_G_required": int(max(1, np.ceil(5.0 * np.nanmax(ratio)))),
            "floor_dominated_fraction": 0.0,
            "floor_dominated_by_family": {n: 0.0 for n, _, _ in fam},
            "neff_correction": False,
            "neff_factors_by_family": {},
            "block_norm": False,
            "floor_rel": 0.0,
            "floor_abs_by_family": {},
        }
        if want_full:
            # Full reference covariance.  Without the floor the sample
            # covariance of N_Gamma records has rank <= N_Gamma-1 < q, so
            # it is singular: shrink towards its own diagonal (Ledoit-Wolf
            # style, intensity SW_GAMMA_SHRINK, default 0.1).
            cov = np.cov(np.asarray(ref_stats, dtype=float), rowvar=False,
                         ddof=1)
            alpha = float(os.environ.get("SW_GAMMA_SHRINK", "0.1"))
            gamma_full = (1.0 - alpha) * cov + alpha * np.diag(var_ref)
            gamma_full = 0.5 * (gamma_full + gamma_full.T)
            w = np.linalg.eigvalsh(gamma_full)
            if w.min() <= 0:
                raise ValueError(
                    f"full Gamma not positive definite (min eig {w.min():.3g});"
                    " raise SW_GAMMA_SHRINK")
            d.update({
                "gamma_type": "full",
                "shrinkage_alpha": alpha,
                "sampled_rank_bound": int(np.asarray(ref_stats).shape[0] - 1),
                "gamma_eig_min": float(w.min()),
                "gamma_eig_max": float(w.max()),
                "gamma_cond": float(w.max() / w.min()),
            })
            return gamma_full, d
        return np.diag(var_ref), d

    if gamma_type is None:
        gamma_type = os.environ.get("SW_GAMMA_TYPE", "diag").lower()
    if gamma_type not in ("diag", "full"):
        raise ValueError(f"unknown gamma_type {gamma_type!r}")
    var_ref = np.var(ref, axis=0, ddof=1)     # NOT divided by anything
    var_fwd = np.var(fwd, axis=0, ddof=1)
    floors = floor_vector(y, layout, n_dense, f_rel, f_abs)
    diag = var_ref + var_fwd + floors**2
    full_info = {}
    if gamma_type == "full":
        cov_ref = np.cov(ref, rowvar=False, ddof=1)
        cov_fwd = np.cov(fwd, rowvar=False, ddof=1)
        sampled = cov_ref + cov_fwd
        gamma_full = sampled + np.diag(floors**2)
        gamma_full = 0.5 * (gamma_full + gamma_full.T)
        w_s = np.linalg.eigvalsh(sampled)
        w_g = np.linalg.eigvalsh(gamma_full)
        tol = max(w_s.max(), 1e-300) * 1e-10
        full_info = {
            "sampled_rank": int((w_s > tol).sum()),
            "sampled_rank_bound": int(ref.shape[0] + fwd.shape[0] - 2),
            "eig_above_floor": int((w_s > float(np.median(floors**2))).sum()),
            "gamma_eig_min": float(w_g.min()),
            "gamma_eig_max": float(w_g.max()),
            "gamma_cond": float(w_g.max() / w_g.min()),
            "floor2_min": float((floors**2).min()),
        }
        if w_g.min() <= 0:
            raise ValueError("full Gamma not positive definite")
        block_norm = False
        if neff_correction is None and os.environ.get("SW_GAMMA_NEFF", "1") == "1":
            print("[gamma-unified] note: n_eff correction is not applied "
                  "to the full-covariance Gamma (correlations encoded directly)")
        neff_correction = False

    if block_norm is None:
        block_norm = os.environ.get("SW_GAMMA_BLOCK_NORM", "0") == "1"
    if block_norm and layout in ("dense", "incr"):
        fam = dict((n, (a, b)) for n, a, b in family_layout(layout, n_dense))
        a, b = fam["hs"]; diag[a:b] *= n_dense / 8.0
        a, b = fam["devrms"]; diag[a:b] *= n_dense / 5.0
        if "incr" in fam:
            a, b = fam["incr"]; diag[a:b] *= n_dense / 5.0

    # Effective-DOF correction (default ON since 2026-08-20): without
    # it the strict-spec diagonal Gamma over-counts the three dense
    # 40-station blocks ~10x and EKI's overconfident first update
    # collapses the ensemble (observed: S2_eki_0, best Phi 2564, p at
    # bound, phi at 1/4 truth).  SW_GAMMA_NEFF=0 reproduces strict-spec.
    if neff_correction is None:
        neff_correction = os.environ.get("SW_GAMMA_NEFF", "1") == "1"
    neff_factors = {}
    if neff_correction:
        neff_factors = effective_dof_factors(ref, layout, n_dense)
        for name, a, b in family_layout(layout, n_dense):
            diag[a:b] *= neff_factors[name]

    floor_dom = floors**2 > (var_ref + var_fwd)
    f_abs_used = dict(F_ABS_DEFAULT, **(f_abs or {}))
    diagnostics = {
        "y_convention": "single_window",
        "forward_convention": "single_run",
        "gamma_type": "full" if gamma_type == "full" else "diagonal",
        "n_gamma": int(ref.shape[0]) if ref.shape[0] == fwd.shape[0] else None,
        "gamma_ref_method": "independent_repeats",
        "n_ref_repeats": int(ref.shape[0]),
        "n_fwd_repeats": int(fwd.shape[0]),
        "floor_rel": float(f_rel),
        "floor_abs_by_family": {k: float(v) for k, v in f_abs_used.items()
                                if k in dict((n, 1) for n, _, _ in
                                             family_layout(layout, n_dense))},
        "block_norm": bool(block_norm),
        "neff_correction": bool(neff_correction),
        "neff_factors_by_family": {k: round(v, 2)
                                   for k, v in neff_factors.items()},
        "q": int(y.size),
        "gamma_diag_min": float(diag.min()),
        "gamma_diag_median": float(np.median(diag)),
        "gamma_diag_max": float(diag.max()),
        "floor_dominated_fraction": float(floor_dom.mean()),
        "floor_dominated_by_family": {
            n: float(floor_dom[a:b].mean())
            for n, a, b in family_layout(layout, n_dense)
        },
        "var_ref_share_median": float(np.median(var_ref / diag)),
        "var_fwd_share_median": float(np.median(var_fwd / diag)),
    }
    diagnostics.update(full_info)
    if gamma_type == "full":
        return gamma_full, diagnostics
    return np.diag(diag), diagnostics


def acceptance_line(d: dict) -> str:
    flag = "  <-- FLAG: floor dominates >50%" \
        if d["floor_dominated_fraction"] > 0.5 else ""
    neff = ""
    if d.get("neff_correction"):
        big = {k: v for k, v in d["neff_factors_by_family"].items()
               if v > 1.05}
        neff = " neff_corr=" + ",".join(
            f"{k}x{v:.1f}" for k, v in big.items())
    n = (f"N_Gamma={d['n_gamma']}" if d.get("n_gamma")
         else f"N_R={d['n_ref_repeats']} N_F={d['n_fwd_repeats']}")
    if d.get("gamma_terms") == "var_ref_only":
        if d.get("gamma_type") == "full":
            return (f"[gamma-spec] Gamma=FULL Cov_ref (shrinkage "
                    f"{d['shrinkage_alpha']:.2f} towards diag) q={d['q']} "
                    f"N_Gamma={d['n_gamma']} cond={d['gamma_cond']:.3g} "
                    f"eig min/max={d['gamma_eig_min']:.3g}/"
                    f"{d['gamma_eig_max']:.3g} | ratio var_fwd/var_ref "
                    f"min/med/max={d['ratio_min']:.3g}/"
                    f"{d['ratio_median']:.3g}/{d['ratio_max']:.3g} -> "
                    f"N_G >= {d['N_G_required']}")
        return (f"[gamma-spec] Gamma=diag(var_ref) q={d['q']} "
                f"N_Gamma={d['n_gamma']} (rel. error of each gamma_i "
                f"{d['var_ref_rel_error']:.1%}) gamma_diag min/med/max="
                f"{d['gamma_diag_min']:.3g}/{d['gamma_diag_median']:.3g}/"
                f"{d['gamma_diag_max']:.3g} | measured ratio var_fwd/var_ref "
                f"min/med/max={d['ratio_min']:.3g}/{d['ratio_median']:.3g}/"
                f"{d['ratio_max']:.3g} -> N_G required >= {d['N_G_required']}")
    full = ""
    if d.get("gamma_type") == "full":
        full = (f" FULL: sampled rank {d['sampled_rank']}/{d['q']}, "
                f"{d['eig_above_floor']} eig > median floor^2, "
                f"cond={d['gamma_cond']:.2g}")
    return (f"[gamma-unified] type={d.get('gamma_type', 'diagonal')} "
            f"q={d['q']} {n} gamma_diag min/med/max="
            f"{d['gamma_diag_min']:.3g}/{d['gamma_diag_median']:.3g}/"
            f"{d['gamma_diag_max']:.3g} floor-dominated="
            f"{d['floor_dominated_fraction']:.2f}{neff}{full}{flag}")


def summary_fields(d: dict) -> dict:
    keys = ("y_convention", "forward_convention", "gamma_type", "n_gamma",
            "gamma_terms", "var_ref_rel_error", "ratio_min", "ratio_median",
            "shrinkage_alpha",
            "ratio_max", "ratio_by_family", "N_G_required",
            "gamma_ref_method", "n_ref_repeats", "n_fwd_repeats",
            "floor_rel", "floor_abs_by_family", "block_norm",
            "neff_correction", "neff_factors_by_family",
            "gamma_diag_min", "gamma_diag_median", "gamma_diag_max",
            "floor_dominated_fraction", "floor_dominated_by_family",
            "sampled_rank", "sampled_rank_bound", "eig_above_floor",
            "gamma_eig_min", "gamma_eig_max", "gamma_cond")
    return {k: d[k] for k in keys if k in d}


# ---------------------------------------------------------------------
# reference-record store
# ---------------------------------------------------------------------

def ref_stats_path(truth_dir, layout: str):
    from pathlib import Path
    return Path(truth_dir) / "ref_records" / f"ref_stats_{layout}.npz"


def load_ref_stats(truth_dir, layout: str, q_expected: int) -> np.ndarray:
    p = ref_stats_path(truth_dir, layout)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing -- run: SW_FINE=1 python sw_ref_records.py "
            f"--layout {layout}  (generates N_R independent truth "
            f"records and stores their statistics)")
    z = np.load(p)
    stats = np.asarray(z["ref_stats"], dtype=float)
    if q_expected is not None and stats.shape[1] != q_expected:
        raise ValueError(
            f"{p}: q={stats.shape[1]} != expected {q_expected} "
            f"(layout/observation mismatch)")
    return stats


# ----------------------------------------------------------------------
# Forward-repeat cache (2026-08-24): the N_F forward repeats at the
# prior-mean theta are the only expensive ingredient of Gamma; floors and
# the n_eff correction are applied afterwards in build_gamma_unified.
# Caching the RAW repeat statistics (not the finished Gamma) makes any
# change of floor / n_eff settings a zero-cost rebuild.
# ----------------------------------------------------------------------
def fwd_cache_path(root, tag, theta, seed_base, n_paths, n_fwd, layout):
    import hashlib
    from pathlib import Path
    key = hashlib.sha1(
        np.ascontiguousarray(np.asarray(theta, dtype=float)).tobytes()
        + f"|{seed_base}|{n_paths}|{n_fwd}|{layout}".encode()
    ).hexdigest()[:12]
    d = Path(root) / "gamma_fwd_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"fwd_stats_{tag}_p{n_paths}_nf{n_fwd}_{key}.npz"


def load_fwd_cache(path, q):
    p = __import__("pathlib").Path(path)
    if p.exists():
        z = np.load(p)
        s = np.asarray(z["fwd_stats"])
        if s.ndim == 2 and s.shape[1] == q:
            return s
    return None


def save_fwd_cache(path, fwd_stats, theta, note=""):
    np.savez(path, fwd_stats=np.asarray(fwd_stats), theta=np.asarray(theta),
             note=note)
