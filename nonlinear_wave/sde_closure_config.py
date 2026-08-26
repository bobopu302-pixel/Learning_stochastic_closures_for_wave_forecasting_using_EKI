"""Closure-chain configuration switch (v2 frozen baseline vs v3 upgrade).

Origin: 3. KDV_nonlinear_case/sde_closure_config.py
Changes vs origin: comments/docstrings only (this provenance header added).

The upgraded deterministic core (decided 2026-07-25) is the DEFAULT for
every closure script; set ``SDE_CLOSURE_V3=0`` only to reproduce archived
legacy-core runs.  The core:

* coarse time step halved (dt = 0.00875, save stride 16 -- saved instants
  stay aligned with the production bundle);
* DRP band-optimised D1/D3 template (fitted here, self-contained), applied
  ONLY around coarse-solver/lifting construction through the
  :func:`template` context manager, so fine-truth runs in the same process
  keep the frozen standard template;
* baseline bundle read from ``results/sde_closure/final/V3_baseline/``.

Without the variable everything behaves exactly as the frozen v2 chain
(no-op context manager, original constants).
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS = PROJECT_DIR / "results" / "sde_closure"

V3 = os.environ.get("SDE_CLOSURE_V3", "1") == "1"

# Optional environment overrides support coarse-limit exploration
# (e.g. 15x/30x grids) without touching the official contract.
COARSE_N4 = int(os.environ.get("SDE_COARSE_N4", "385"))
_DEFAULT_DT = 0.00875 if V3 else 0.0175
COARSE_DT = float(os.environ.get("SDE_COARSE_DT", str(_DEFAULT_DT)))
_DEFAULT_STRIDE = 16 if V3 else 8
OUTPUT_STRIDE = int(
    os.environ.get("SDE_OUTPUT_STRIDE", str(_DEFAULT_STRIDE))
)
BASELINE_SUBDIR = (
    "final/V3_baseline" if V3 else "verification/N1_space8_time8"
)
BASELINE_NPZ = Path(
    os.environ.get(
        "SDE_BASELINE_NPZ",
        str(RESULTS / BASELINE_SUBDIR / "baseline_data.npz"),
    )
)
RUN_TAG = "v3" if V3 else "v2"

_THETA_BAND = (
    0.02,
    float(os.environ.get("SDE_DRP_BAND_MAX", "1.05")),
)


def _fit_drp_weights() -> tuple[np.ndarray, np.ndarray]:
    """Constrained band least-squares fit (same maths as N12)."""

    theta = np.linspace(_THETA_BAND[0], _THETA_BAND[1], 2001)
    sines = np.sin(np.outer(theta, np.arange(1.0, 4.0)))

    def solve(design, target, constraints, values):
        normal = design.T @ design
        size = normal.shape[0]
        m = constraints.shape[0]
        kkt = np.zeros((size + m, size + m))
        kkt[:size, :size] = normal
        kkt[:size, size:] = constraints.T
        kkt[size:, :size] = constraints
        rhs = np.concatenate([design.T @ target, values])
        return np.linalg.solve(kkt, rhs)[:size]

    d1 = solve(
        2.0 * sines,
        theta,
        np.asarray([[2.0, 4.0, 6.0]]),
        np.asarray([1.0]),
    )
    d3 = solve(
        -2.0 * sines,
        theta**3,
        np.asarray([[1.0, 2.0, 3.0], [1.0, 8.0, 27.0]]),
        np.asarray([0.0, 3.0]),
    )
    return d1, d3


@contextmanager
def template(force: bool = False):
    """DRP template patch around coarse construction; no-op unless v3.

    Fine-truth solver constructions must stay OUTSIDE this context so the
    truth always uses the frozen standard template.
    """

    if not (V3 or force):
        yield
        return
    import high_order_matched_dabc as matched

    d1, d3 = _fit_drp_weights()
    saved = (matched.D1_POSITIVE_C6, matched.D3_POSITIVE)
    matched.D1_POSITIVE_C6 = d1
    matched.D3_POSITIVE = d3
    try:
        yield
    finally:
        matched.D1_POSITIVE_C6, matched.D3_POSITIVE = saved


def describe() -> dict[str, object]:
    return {
        "v3_active": V3,
        "coarse_dt": COARSE_DT,
        "output_stride": OUTPUT_STRIDE,
        "baseline_bundle": str(BASELINE_NPZ),
        "template": "DRP band-optimised" if V3 else "frozen C6/C4",
    }
