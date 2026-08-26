"""Observation-error covariance (Gamma) and N_G calibration.

Spec summary (EKI_algorithm_spec_2026-08-23.md):

- Gamma carries the REFERENCE side only: it is estimated from ``N_Gamma``
  independent reference windows of the truth, with no forward term and no
  floor.  The model's own sampling fluctuation is removed separately, by
  averaging ``N_G`` forward realisations inside a single evaluation of G.
- ``build_gamma`` returns either a diagonal Gamma (per-component ddof=1
  sample variances) or the full sample covariance, with the rank / condition
  checks the spec requires, and an optional effective-sample-size inflation
  for correlated statistic families.
- ``calibrate_n_g`` chooses N_G so that the omitted forward-variance term
  ``var_fwd / N_G`` is at most a fraction ``delta`` of the reference variance,
  component-wise, judged at a decisive probe parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

Array = np.ndarray


# ---------------------------------------------------------------- Gamma


@dataclass(frozen=True)
class GammaEstimate:
    matrix: Array  # the Gamma actually used by EKI
    var_ref: Array  # diagonal of the reference covariance (the truth error bars)
    structure: str  # "diagonal" | "full"
    n_gamma: int
    rank: int
    condition_number: float
    corr_min_eigenvalue: float
    neff_inflation: Array | None  # per-component factor applied, or None


def _neff_inflation(records: Array, families: Sequence[Sequence[int]]) -> Array:
    """Per-family effective-sample-size inflation gamma_i *= n/n_eff, with
    n_eff = n / (1 + (n-1) * rho_bar) and rho_bar the mean off-diagonal
    correlation inside the family (spec section 0.1, ``neff_correction``)."""

    q = records.shape[1]
    factor = np.ones(q, dtype=float)
    corr = np.corrcoef(records, rowvar=False)
    for family in families:
        idx = np.asarray(list(family), dtype=int)
        n = idx.size
        if n < 2:
            continue
        block = corr[np.ix_(idx, idx)]
        off = block[np.triu_indices(n, 1)]
        rho_bar = float(np.mean(off))
        factor[idx] = max(1.0, 1.0 + (n - 1) * rho_bar)
    return factor


def build_gamma(
    records: Array,
    *,
    structure: str,
    families: Sequence[Sequence[int]] | None = None,
    neff_correction: bool = False,
) -> GammaEstimate:
    """Gamma from the N_Gamma reference records (spec section 1.2).

    ``records`` has shape (N_Gamma, q).  ``var_ref`` is the ddof=1 sample
    variance per component -- NOT divided by N_Gamma, no floor, no forward
    term.  With ``structure="full"`` the full sample covariance is used and its
    rank / condition number are recorded (the spec requires the check)."""

    records = np.asarray(records, dtype=float)
    if records.ndim != 2 or records.shape[0] < 2:
        raise ValueError("need at least two reference records")
    n_gamma, q = records.shape
    cov = np.cov(records, rowvar=False, ddof=1)
    cov = np.atleast_2d(cov)
    var_ref = np.diag(cov).copy()
    corr = cov / np.sqrt(np.outer(var_ref, var_ref))
    corr_min = float(np.linalg.eigvalsh(0.5 * (corr + corr.T)).min())

    inflation = None
    if neff_correction:
        if families is None:
            raise ValueError("neff_correction requires the statistic families")
        inflation = _neff_inflation(records, families)

    if structure == "diagonal":
        diag = var_ref * (inflation if inflation is not None else 1.0)
        matrix = np.diag(diag)
    elif structure == "full":
        matrix = 0.5 * (cov + cov.T)
        if inflation is not None:
            scale = np.sqrt(inflation)
            matrix = matrix * np.outer(scale, scale)
    else:
        raise ValueError(f"unknown gamma structure: {structure}")

    rank = int(np.linalg.matrix_rank(matrix))
    cond = float(np.linalg.cond(matrix))
    eig_min = float(np.linalg.eigvalsh(matrix).min())
    if structure == "full" and rank < q:
        raise ValueError(
            f"full Gamma is rank deficient ({rank} < {q}): drop the redundant statistic "
            "or use gamma_structure='diagonal' (spec section 0.1)"
        )
    if eig_min <= 0.0:
        raise ValueError("Gamma is not positive definite")
    return GammaEstimate(
        matrix=matrix,
        var_ref=var_ref,
        structure=structure,
        n_gamma=n_gamma,
        rank=rank,
        condition_number=cond,
        corr_min_eigenvalue=corr_min,
        neff_inflation=inflation,
    )


# ---------------------------------------------------------------- N_G


@dataclass(frozen=True)
class CalibrationResult:
    n_g: int
    ratio_min: float
    ratio_median: float
    ratio_max: float
    ratio: Array
    retained: Array
    per_probe: dict


def calibrate_n_g(
    probe_stats: dict[str, Array],
    var_ref: Array,
    *,
    delta: float = 0.10,
    decisive_probe: str,
    retain_fraction: float = 0.01,
) -> CalibrationResult:
    """N_G from the omitted term var_fwd / N_G (spec section 1.3).

    ``probe_stats`` maps a probe label to a (K, q) array of forward statistics.
    ``decisive_probe`` names the probe that sets N_G (the spec: the
    near-optimum one); the others are recorded as diagnostics.  Components
    whose reference variance is below ``retain_fraction`` of the median are
    dropped as near-degenerate before taking the maximum ratio."""

    var_ref = np.asarray(var_ref, dtype=float)
    retained = var_ref >= retain_fraction * float(np.median(var_ref))
    per_probe = {}
    for label, stats in probe_stats.items():
        stats = np.asarray(stats, dtype=float)
        if stats.shape[0] < 2:
            raise ValueError(f"probe {label}: need at least two forward runs")
        ratio = stats.var(axis=0, ddof=1) / var_ref
        per_probe[label] = dict(
            k=int(stats.shape[0]),
            ratio_min=float(ratio[retained].min()),
            ratio_median=float(np.median(ratio[retained])),
            ratio_max=float(ratio[retained].max()),
            n_g_implied=int(max(1, np.ceil(ratio[retained].max() / delta))),
        )
    if decisive_probe not in probe_stats:
        raise KeyError(f"decisive probe {decisive_probe!r} not among {list(probe_stats)}")
    ratio = np.asarray(probe_stats[decisive_probe], dtype=float).var(axis=0, ddof=1) / var_ref
    n_g = int(max(1, np.ceil(float(ratio[retained].max()) / delta)))
    return CalibrationResult(
        n_g=n_g,
        ratio_min=float(ratio[retained].min()),
        ratio_median=float(np.median(ratio[retained])),
        ratio_max=float(ratio[retained].max()),
        ratio=ratio,
        retained=retained,
        per_probe=per_probe,
    )
