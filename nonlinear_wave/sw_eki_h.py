"""Step 2 (S2 = H2/H3) driver: prior-free GP-surface nonlinear-term inversion.
Model tiers (pre-registered):

  H2  N = m(u, u_y/s0; theta_GP) * d(y)^p          (free GP surface)
  H3  N = -d/dy[ d(y)^p * F(u, u_y/s0; theta_GP) ] (conservative flux)

The unknown surface is the SSW21-style GP conditional mean: virtual noisy
observations m' at R=20 fixed design nodes, learnable nugget and
product-RBF kernel hyper-parameters (log space), solved through the
representer system.  GPSurface below is the 2-D instance of the shared
algorithms.gpr.make_gp_mean contract (same kernel, same representer solve);
it is kept local because the hot loop needs the tabulated fast path in
GPDriftSolver.  theta in R^27:

  [ m'_1..20 (linear) | log s_obs, log a, log l_u, log l_s | p
    | log phi | q ]

Environment is read AT IMPORT TIME (see v3_world): always launch through
v3_spec_chain.sh or replicate its exports.

Usage (spec chain)::

    SW_FINE=1 SW_VERSION_DIR=v3spec SW_DURATION_S=6600 \
    SW_SYNTH_PERIOD_S=6600 SW_FORWARD_PATHS=1 SW_N_G=10 \
    SW_GAMMA_TYPE=diag SW_H_PHIQ_PRIOR=s1 \
    SW_H_S1_SUMMARY=results/stepwise/v3spec/S1a_eki_dense_fine_diag/summary.json \
    SW_H_OUTTAG=_diag \
    python sw_eki_h.py --variant H2 --members 100 --iterations 20 \
        --processes 100 --resume
"""

from __future__ import annotations

import os
from pathlib import Path as _Path

_HERE = _Path(__file__).resolve().parent
SW_FINE = os.environ.get("SW_FINE", "") == "1"
if SW_FINE:
    os.environ["SDE_COARSE_N4"] = "3073"
    os.environ["SDE_COARSE_DT"] = "0.002"
    os.environ["SDE_OUTPUT_STRIDE"] = "70"
    os.environ["SDE_CLOSURE_V3"] = "0"
os.environ["SW_GAMMA_TERMS"] = "var_ref_only"   # spec Gamma, hard-set
import v3_world  # noqa: E402  (v3 twin world patches; no-op for v2)
_VERSION_DIR = v3_world.version_dir("v2_fine" if SW_FINE else "v1_coarse")
_TRUTH_NAME = "truth_S1a_fine" if SW_FINE else "truth_S1a"
os.environ["SDE_BASELINE_NPZ"] = str(
    _HERE / "results" / "stepwise" / _VERSION_DIR / _TRUTH_NAME
    / "baseline_data.npz"
)
# sw_eki_s1's preamble must see the same world when imported below.
os.environ.setdefault("SW_S1_VARIANT", "S1a")
os.environ.setdefault("SW_S1_SUFFIX", "_fine" if SW_FINE else "")
v3_world.ensure_patched()

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Pool

import numpy as np

# code_rp root on sys.path so the shared algorithm layer imports.
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from algorithms.eki import run_eki  # noqa: E402

import sde_closure_config as closure_config  # noqa: E402
import sde_closure_eki as base  # noqa: E402
from sde_closure_core import (  # noqa: E402
    GridWhiteNoise,
    GridWhiteNoiseParameters,
    StochasticImplicitMidpointDABCSolver,
    terrain_weight,
)
from sde_closure_eki_dense import _init_worker_dense  # noqa: E402
from sw_eki_s1 import (  # noqa: E402
    CORR_CELLS,
    FORWARD_PATHS,
    N_DENSE,
    compute_statistics_incr,
    truth_statistics_and_blocks_twin,
    N_FWD_REPEATS,
)

OUT_ROOT = _HERE / "results" / "stepwise" / _VERSION_DIR
TRUTH_DIR = OUT_ROOT / _TRUTH_NAME
GPR_REF_JSON = OUT_ROOT / "GPR_reference" / "calibration.json"
EKI_SEED_H = 20260814
VALIDATION_SEED = 20260815
EPSILON = 0.01
GAMMA_MODE = "unified"      # spec, hard-set (kept for cache file naming)

# ----------------------------------------------------------------------
# Design nodes and scales
# ----------------------------------------------------------------------

U_NODES = np.asarray([-2.5, -1.25, 0.0, 1.25, 2.5])
S_NODES = np.asarray([-1.5, -0.5, 0.5, 1.5])
NODES = np.array([[u, s] for u in U_NODES for s in S_NODES])  # (20, 2)
R_NODES = NODES.shape[0]
THETA_DIM = R_NODES + 4 + 1 + 2  # nodes | log(4) | p | log phi, q


def load_calibration() -> dict:
    """T0 (tendency scale) and s0 (slope scale).

    Preferred source: the GPR reference track's calibration.json
    (data-driven).  Fallback: a conservative constant so that --check
    and smoke runs work before the reference track has been run.
    """

    if GPR_REF_JSON.exists():
        record = json.loads(GPR_REF_JSON.read_text("utf-8"))
        return {"t0": float(record["t0_tendency_scale"]),
                "s0": float(record["s0_slope_scale"]),
                "source": str(GPR_REF_JSON)}
    return {"t0": 0.05, "s0": 1.0, "source": "fallback-constants"}


# ----------------------------------------------------------------------
# 2-D GP conditional mean (representer solve; see module docstring for
# the relation to algorithms.gpr.make_gp_mean)
# ----------------------------------------------------------------------


def rbf2(x: np.ndarray, z: np.ndarray, amplitude: float,
         l_u: float, l_s: float) -> np.ndarray:
    """Product RBF kernel between point sets x (n,2) and z (m,2)."""

    du = (x[:, 0:1] - z[None, :, 0]) / l_u
    ds = (x[:, 1:2] - z[None, :, 1]) / l_s
    return amplitude**2 * np.exp(-0.5 * (du**2 + ds**2))


class GPSurface:
    """m(u, s) = k((u,s), NODES) @ [K + s_obs^2 I]^{-1} m'."""

    def __init__(self, node_values: np.ndarray, obs_noise: float,
                 amplitude: float, l_u: float, l_s: float) -> None:
        if not (np.isfinite(obs_noise) and obs_noise > 0
                and np.isfinite(amplitude) and amplitude > 0
                and np.isfinite(l_u) and l_u > 0
                and np.isfinite(l_s) and l_s > 0):
            raise FloatingPointError("invalid GP hyper-parameters")
        k_nn = rbf2(NODES, NODES, amplitude, l_u, l_s)
        k_nn = k_nn + (obs_noise**2 + 1e-8) * np.eye(R_NODES)
        self._weights = np.linalg.solve(
            k_nn, np.asarray(node_values, dtype=float)
        )
        self._amp, self._lu, self._ls = amplitude, l_u, l_s

    def __call__(self, u: np.ndarray, s: np.ndarray) -> np.ndarray:
        pts = np.column_stack([np.ravel(u), np.ravel(s)])
        vals = rbf2(pts, NODES, self._amp, self._lu, self._ls) \
            @ self._weights
        return vals.reshape(np.shape(u))


# ----------------------------------------------------------------------
# Solver with the learned drift
# ----------------------------------------------------------------------


TABLE_U = np.linspace(-4.0, 4.0, 321)     # spacing 0.025
TABLE_S = np.linspace(-3.0, 3.0, 241)     # spacing 0.025


class GPDriftSolver(StochasticImplicitMidpointDABCSolver):
    """Frozen linear operator + GP-surface nonlinearity (H2 or H3).

    The GP mean is evaluated ONCE on a dense uniform table at
    set_surface time; the hot loop uses bilinear interpolation
    (table spacing 0.025 << kernel lengthscale bounds, error
    negligible; removes ~300k exp calls per fixed-point iteration --
    measured 23 min -> ~4 min per path).  Inputs are clamped to the
    table range (occupancy beyond |u|=4, |s|=3 is nil).
    """

    def set_surface(self, surface: GPSurface, p_power: float,
                    variant: str, s0: float) -> None:
        self._p = float(p_power)
        self._variant = variant
        self._s0 = float(s0)
        self._depth_p = self.depth_ratio ** self._p
        gu, gs = np.meshgrid(TABLE_U, TABLE_S, indexing="ij")
        self._table = surface(gu, gs)
        self._u0 = TABLE_U[0]
        self._s0g = TABLE_S[0]
        self._inv_du = 1.0 / (TABLE_U[1] - TABLE_U[0])
        self._inv_ds = 1.0 / (TABLE_S[1] - TABLE_S[0])
        self._nu = TABLE_U.size
        self._ns = TABLE_S.size

    def _interp(self, u: np.ndarray, s: np.ndarray) -> np.ndarray:
        fu = np.clip((u - self._u0) * self._inv_du, 0.0,
                     self._nu - 1.001)
        fs = np.clip((s - self._s0g) * self._inv_ds, 0.0,
                     self._ns - 1.001)
        iu = fu.astype(np.intp)
        js = fs.astype(np.intp)
        au = fu - iu
        bs = fs - js
        t = self._table
        return ((1 - au) * (1 - bs) * t[iu, js]
                + au * (1 - bs) * t[iu + 1, js]
                + (1 - au) * bs * t[iu, js + 1]
                + au * bs * t[iu + 1, js + 1])

    def nonlinear(self, normalized: np.ndarray) -> np.ndarray:
        u = self.to_surface(np.asarray(normalized, dtype=float))
        slope = np.asarray(self.d1 @ u).ravel() / self._s0
        if self._variant == "H2":
            drift_u = self._interp(u, slope) * self._depth_p
        else:  # H3: conservative flux, envelope inside the divergence
            flux = self._depth_p * self._interp(u, slope)
            drift_u = -np.asarray(self.d1 @ flux).ravel()
        drift = self.surface_to_green * drift_u
        drift[:3] = 0.0
        drift[-3:] = 0.0
        return drift


# ----------------------------------------------------------------------
# theta encode/decode
# ----------------------------------------------------------------------
# layout: [ m'_0..19 | log s_obs | log a | log l_u | log l_s | p
#           | log phi | q ]

I_LOG = (R_NODES, R_NODES + 1, R_NODES + 2, R_NODES + 3)
I_P, I_PHI, I_Q = R_NODES + 4, R_NODES + 5, R_NODES + 6
SPEC = True   # kept as a module constant: v3_* scripts test h.SPEC
N_G = int(os.environ.get("SW_N_G", "1"))


def spec_seed_root(eki_seed, iteration, realisation):
    """CRN: same seeds for every member, new ones at each iteration."""
    return (eki_seed, 3000 + iteration, realisation)


def make_bounds(t0: float) -> np.ndarray:
    # SDE_H_NODE_BOUND_T0: node-prior half-width in units of t0
    # (default 5; the pre-registered high-failure remedy sets 3).
    mult = float(os.environ.get("SDE_H_NODE_BOUND_T0", "5.0"))
    bounds = np.empty((THETA_DIM, 2))
    bounds[:R_NODES] = (-mult * t0, mult * t0)
    bounds[I_LOG[0]] = (np.log(1e-2 * t0), np.log(1e2 * t0))
    bounds[I_LOG[1]] = (np.log(1e-2 * t0), np.log(1e2 * t0))
    bounds[I_LOG[2]] = (np.log(0.5), np.log(5.0))
    bounds[I_LOG[3]] = (np.log(0.3), np.log(3.0))
    bounds[I_P] = (-2.0, 2.0)
    bounds[I_PHI] = (np.log(1e-4), np.log(0.5))   # log coordinates (spec)
    bounds[I_Q] = (0.0, 6.0)
    return bounds


def clip_theta_h(thetas: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    return np.clip(thetas, bounds[:, 0], bounds[:, 1])


def decode_theta_h(theta: np.ndarray) -> dict:
    theta = np.asarray(theta, dtype=float)
    return {
        "node_values": theta[:R_NODES],
        "obs_noise": float(np.exp(theta[I_LOG[0]])),
        "amplitude": float(np.exp(theta[I_LOG[1]])),
        "l_u": float(np.exp(theta[I_LOG[2]])),
        "l_s": float(np.exp(theta[I_LOG[3]])),
        "p": float(theta[I_P]),
        # spec: phi is a positive parameter held in log coordinates
        # inside EKI and exponentiated for the forward run
        "phi": float(np.exp(theta[I_PHI])),
        "q": float(theta[I_Q]),
    }


def initial_ensemble_h(members: int, rng: np.random.Generator,
                       t0: float) -> np.ndarray:
    thetas = np.empty((members, THETA_DIM))
    thetas[:, :R_NODES] = rng.normal(
        0.0, 2.0 * t0, size=(members, R_NODES)
    )
    thetas[:, I_LOG[0]] = np.log(t0) + rng.uniform(
        np.log(0.1), np.log(10.0), size=members
    )
    thetas[:, I_LOG[1]] = np.log(t0) + rng.uniform(
        np.log(0.1), np.log(10.0), size=members
    )
    thetas[:, I_LOG[2]] = rng.uniform(
        np.log(0.8), np.log(3.0), size=members
    )
    thetas[:, I_LOG[3]] = rng.uniform(
        np.log(0.5), np.log(2.0), size=members
    )
    thetas[:, I_P] = rng.uniform(-1.5, 1.5, size=members)
    # Spec: draw uniformly from the prior interval, then hold phi in
    # log coordinates inside EKI.
    phi_draw = rng.uniform(0.003, 0.03, size=members)
    thetas[:, I_PHI] = np.log(phi_draw)
    thetas[:, I_Q] = rng.uniform(0.0, 5.0, size=members)
    # Ladder prior (SW_H_PHIQ_PRIOR=s1): Step 2 inherits the Step-1
    # posterior of the noise parameters as an informative Gaussian prior
    # instead of the flat U(0.003,0.03) x U(0,5).  The uniform draws
    # above are still consumed so every other dimension of the initial
    # ensemble is bit-identical to the flat-prior runs (same rng stream).
    s1 = s1_phiq_prior()
    if s1 is not None:
        # Log-normal in model coordinates with the S1 posterior's
        # mean and sd -> normal in the log coordinates used by EKI.
        rel = max(s1["phi_std"] / s1["phi_mean"], 1e-6)
        sig = np.sqrt(np.log1p(rel ** 2))
        mu = np.log(s1["phi_mean"]) - 0.5 * sig ** 2
        thetas[:, I_PHI] = rng.normal(mu, sig, size=members)
        thetas[:, I_Q] = rng.normal(s1["q_mean"], s1["q_std"], size=members)
    return thetas


def s1_phiq_prior() -> dict | None:
    """Return the S1 posterior (phi, q) moments if SW_H_PHIQ_PRIOR=s1."""
    if os.environ.get("SW_H_PHIQ_PRIOR", "uniform").lower() != "s1":
        return None
    default = OUT_ROOT / "S1a_eki_dense_fine_diag" / "summary.json"
    path = _Path(os.environ.get("SW_H_S1_SUMMARY", str(default)))
    summary = json.loads(path.read_text(encoding="utf-8"))
    spread = summary["final_parameter_spread"]
    names = summary.get("parameter_names", ["phi", "q"])
    i_phi, i_q = names.index("phi"), names.index("q")
    out = {
        "source": str(path),
        "phi_mean": float(summary["decoded_final_mean"]["phi"]),
        "phi_std": float(spread[i_phi]),
        "q_mean": float(summary["decoded_final_mean"]["q"]),
        "q_std": float(spread[i_q]),
    }
    print(f"[SW_H] (phi, q) prior = S1 posterior N({out['phi_mean']:.5f}, "
          f"{out['phi_std']:.5f}) x N({out['q_mean']:.3f}, {out['q_std']:.3f}) "
          f"from {path}")
    return out


# ----------------------------------------------------------------------
# Forward map: FORWARD_PATHS paths -> dense+increment statistics (q = 151)
# ----------------------------------------------------------------------

_CAL: dict = {}


def _init_worker_h(duration_s: float, boundary_seed: int,
                   calibration: dict) -> None:
    _init_worker_dense(duration_s, boundary_seed, False)
    _CAL.update(calibration)


def forward_statistics_h(
    task: tuple[np.ndarray, str, tuple[int, ...]],
) -> np.ndarray | None:
    theta, variant, seed_root = task
    parameters = base._CTX["parameters"]
    decoded = decode_theta_h(theta)
    try:
        surface = GPSurface(
            decoded["node_values"], decoded["obs_noise"],
            decoded["amplitude"], decoded["l_u"], decoded["l_s"],
        )
    except FloatingPointError:
        return None
    weight = decoded["phi"] * terrain_weight(
        base._CTX["depth_ratio"], decoded["q"]
    )
    dense_series, gauge_series = [], []
    times_s = None
    for path in range(FORWARD_PATHS):
        rng = np.random.default_rng(list(seed_root) + [path])
        noise = GridWhiteNoise(
            GridWhiteNoiseParameters(
                phi_amplitude=1.0,
                correlation_length_cells=CORR_CELLS,
            ),
            base._CTX["y"],
            parameters.lambda_ref_m,
            base._CTX["surface_to_green"],
            base.COARSE_DT,
            rng,
            spatial_weight=weight,
        )
        with closure_config.template():
            solver = GPDriftSolver(
                base._CTX["y"],
                base._CTX["depth_ratio"],
                parameters.epsilon,
                parameters.mu,
                base.COARSE_DT,
                base._CTX["n_steps"],
            )
        solver.set_surface(surface, decoded["p"], variant, _CAL["s0"])
        try:
            times, surface_out, _, _ = solver.run_stochastic(
                np.zeros_like(base._CTX["y"]),
                base.OUTPUT_STRIDE,
                base._CTX["traces"],
                noise_increment=noise,
            )
        except FloatingPointError:
            return None
        times_s = np.asarray(times) * parameters.time_ref_s
        eta = (
            np.asarray(surface_out[:, : base.COARSE_N4], dtype=float)
            * parameters.a_ref_m
        )
        if not np.all(np.isfinite(eta)):
            return None
        dense_series.append(eta[:, base._CTX["dense_columns"]])
        gauge_series.append(eta[:, base._CTX["gauge_columns"]])
    count = times_s.size
    analysis, _ = base._analysis_mask(times_s, base.ANALYSIS_START_S)
    return compute_statistics_incr(
        times_s,
        dense_series,
        gauge_series,
        np.asarray(base._CTX["dense_baseline"])[:count],
        analysis,
    )


# ----------------------------------------------------------------------
# Path-parallel evaluation (SW_H_PATH_PARALLEL=1): the closure-noise
# paths of every member become independent pool tasks (J x paths tasks
# per iteration instead of J), so boxes with >> J cores are used fully.
# Each path is seeded exactly as in forward_statistics_h
# ([*seed_root, path]) and the pooled statistics are computed from the
# same per-path series with the same function, so the result is
# bit-identical to the member-serial evaluation.
# ----------------------------------------------------------------------
PATH_PARALLEL = os.environ.get("SW_H_PATH_PARALLEL", "0") == "1"


def forward_path_h(task):
    """One (theta, path) task -> (times_s, dense_series, gauge_series)."""
    theta, variant, seed_root, path = task
    parameters = base._CTX["parameters"]
    decoded = decode_theta_h(theta)
    try:
        surface = GPSurface(
            decoded["node_values"], decoded["obs_noise"],
            decoded["amplitude"], decoded["l_u"], decoded["l_s"],
        )
    except FloatingPointError:
        return None
    weight = decoded["phi"] * terrain_weight(
        base._CTX["depth_ratio"], decoded["q"]
    )
    rng = np.random.default_rng(list(seed_root) + [path])
    noise = GridWhiteNoise(
        GridWhiteNoiseParameters(
            phi_amplitude=1.0,
            correlation_length_cells=CORR_CELLS,
        ),
        base._CTX["y"],
        parameters.lambda_ref_m,
        base._CTX["surface_to_green"],
        base.COARSE_DT,
        rng,
        spatial_weight=weight,
    )
    with closure_config.template():
        solver = GPDriftSolver(
            base._CTX["y"],
            base._CTX["depth_ratio"],
            parameters.epsilon,
            parameters.mu,
            base.COARSE_DT,
            base._CTX["n_steps"],
        )
    solver.set_surface(surface, decoded["p"], variant, _CAL["s0"])
    try:
        times, surface_out, _, _ = solver.run_stochastic(
            np.zeros_like(base._CTX["y"]),
            base.OUTPUT_STRIDE,
            base._CTX["traces"],
            noise_increment=noise,
        )
    except FloatingPointError:
        return None
    times_s = np.asarray(times) * parameters.time_ref_s
    eta = (
        np.asarray(surface_out[:, : base.COARSE_N4], dtype=float)
        * parameters.a_ref_m
    )
    if not np.all(np.isfinite(eta)):
        return None
    return (times_s, eta[:, base._CTX["dense_columns"]],
            eta[:, base._CTX["gauge_columns"]])


def forward_statistics_h_batch(pool, tasks, chunksize: int = 1):
    """Evaluate many (theta, variant, seed_root) tasks.

    Returns a list aligned with `tasks` (None for failed members), equal
    to [forward_statistics_h(t) for t in tasks].
    """
    if not PATH_PARALLEL:
        return pool.map(forward_statistics_h, tasks, chunksize=chunksize)
    path_tasks = [
        (theta, variant, seed_root, path)
        for (theta, variant, seed_root) in tasks
        for path in range(FORWARD_PATHS)
    ]
    raw = pool.map(forward_path_h, path_tasks, chunksize=chunksize)
    baseline = np.asarray(base._CTX["dense_baseline"])
    out = []
    for i in range(len(tasks)):
        chunk = raw[i * FORWARD_PATHS:(i + 1) * FORWARD_PATHS]
        if any(c is None for c in chunk):
            out.append(None)
            continue
        times_s = chunk[0][0]
        count = times_s.size
        analysis, _ = base._analysis_mask(times_s, base.ANALYSIS_START_S)
        out.append(compute_statistics_incr(
            times_s,
            [c[1] for c in chunk],
            [c[2] for c in chunk],
            baseline[:count],
            analysis,
        ))
    return out


# ----------------------------------------------------------------------
# True surface (for the degeneration gate and recovery metrics ONLY --
# never enters the inversion)
# ----------------------------------------------------------------------


def true_surface_values(t0_unused: float, s0: float) -> np.ndarray:
    """m_dagger(u, s) = -(3 eps / 2) * s0 * u * s  (nondim tendency).

    Continuous identity: the deleted term is -gamma u u_y with gamma =
    (3 eps/2) d^{-1/2}; in (u, s = u_y/s0) coordinates and with the
    d^p envelope carrying p = -1/2, the surface is bilinear in (u, s).
    """

    return -(1.5 * EPSILON) * s0 * NODES[:, 0] * NODES[:, 1]


def recovery_metrics(decoded: dict, s0: float) -> dict:
    grid_u, grid_s = np.meshgrid(
        np.linspace(-2.5, 2.5, 81), np.linspace(-1.5, 1.5, 61),
        indexing="ij",
    )
    surface = GPSurface(
        decoded["node_values"], decoded["obs_noise"],
        decoded["amplitude"], decoded["l_u"], decoded["l_s"],
    )
    learned = surface(grid_u, grid_s)
    bilinear = grid_u * grid_s
    coeff = float(
        np.sum(learned * bilinear) / np.sum(bilinear**2)
    )
    residual = learned - coeff * bilinear
    projection_corr = float(np.sqrt(max(
        0.0, 1.0 - np.sum(residual**2) / max(np.sum(learned**2), 1e-300)
    )))
    true_coeff = -(1.5 * EPSILON) * s0
    zero_row = surface(np.linspace(-2.5, 2.5, 81),
                       np.zeros(81))
    return {
        "projection_corr_onto_us_plane": projection_corr,
        "bilinear_coeff": coeff,
        "bilinear_coeff_true": true_coeff,
        "bilinear_coeff_rel_error": float(
            abs(coeff - true_coeff) / abs(true_coeff)
        ),
        "m_u0_row_maxabs": float(np.max(np.abs(zero_row))),
        "p": decoded["p"],
        "p_true_reference": -0.5,
        "phi": decoded["phi"],
        "q": decoded["q"],
    }


# ----------------------------------------------------------------------
# Degeneration gate
# ----------------------------------------------------------------------


def degeneration_check(variant: str, calibration: dict) -> None:
    from sde_closure_context import CoarseModelContext, ModelAConfig

    s0 = calibration["s0"]
    config = ModelAConfig(boundary_seed=20260801)
    context = CoarseModelContext(config)
    node_values = true_surface_values(calibration["t0"], s0)
    # Hyper-parameters chosen wide/smooth; nugget small: the gate asks
    # for STATISTICS-level equivalence, not bitwise identity (the GP
    # mean is a smooth approximation of the bilinear surface).
    surface = GPSurface(node_values, 1e-2 * calibration["t0"],
                        2.0 * calibration["t0"], 1.6, 1.1)
    weight = 0.0124 * terrain_weight(context.depth_ratio, 2.35)
    bundle = np.load(TRUTH_DIR / "truth_bundle.npz", allow_pickle=True)
    truth0 = np.asarray(bundle["eta_paths_m"], dtype=float)[0]
    rng = np.random.default_rng([20260801, 0])
    noise = GridWhiteNoise(
        GridWhiteNoiseParameters(
            phi_amplitude=1.0, correlation_length_cells=CORR_CELLS
        ),
        context.y, context.parameters.lambda_ref_m,
        context.surface_to_green, config.coarse_dt, rng,
        spatial_weight=weight,
    )
    with closure_config.template():
        solver = GPDriftSolver(
            context.y, context.depth_ratio,
            context.parameters.epsilon, context.parameters.mu,
            config.coarse_dt, context.n_steps,
        )
    solver.set_surface(surface, -0.5, variant, s0)
    _, out, _, _ = solver.run_stochastic(
        np.zeros_like(context.y), config.output_stride,
        context.traces, noise_increment=noise,
    )
    eta = (
        np.asarray(out[:, : config.coarse_n4], dtype=float)
        * context.parameters.a_ref_m
    )
    t = np.asarray(bundle["times_s"], dtype=float)
    hs_t = 4.0 * np.std(truth0[t >= 600.0], axis=0)
    hs_m = 4.0 * np.std(eta[t >= 600.0], axis=0)
    rel = float(np.linalg.norm(hs_m - hs_t) / np.linalg.norm(hs_t))
    print(f"[H-check {variant}] same-seed vs twin path0: field rms "
          f"diff {np.sqrt(np.mean((eta - truth0) ** 2)):.5f} m, "
          f"Hs rel L2 {rel:.5f} (gate 0.02)")


# ----------------------------------------------------------------------
# Main EKI run (algorithms.eki engine + driver-managed checkpoint/resume)
# ----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("H2", "H3"),
                        required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--members", type=int, default=60)
    parser.add_argument("--iterations", type=int, default=14)
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()

    calibration = load_calibration()
    print(f"[SW_{arguments.variant}] calibration: "
          f"t0={calibration['t0']:.4g} s0={calibration['s0']:.4g} "
          f"({calibration['source']})")
    if arguments.check:
        degeneration_check(arguments.variant, calibration)
        return

    # SW_H_OUTTAG lets a re-run under a different Gamma type (diag/full)
    # land in its own dir and never resume a checkpoint produced under
    # another Gamma.
    output = OUT_ROOT / (f"{arguments.variant}_eki"
                         + os.environ.get("SW_H_OUTTAG", ""))
    if output.exists() and any(output.iterdir()) \
            and not (arguments.overwrite or arguments.resume):
        raise SystemExit(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)

    members = arguments.members
    max_iterations = arguments.iterations
    duration_s = v3_world.DURATION_S
    if arguments.smoke:
        members, max_iterations, duration_s = 8, 2, 240.0
    tag = f"SW_{arguments.variant}"
    t0 = calibration["t0"]
    bounds = make_bounds(t0)

    observation, _block_stats, _ = truth_statistics_and_blocks_twin(
        TRUTH_DIR / "truth_bundle.npz", dense=True, incr=True
    )
    q_obs = observation.size
    print(f"[{tag}] dims={THETA_DIM} q_obs={q_obs} J={members} N_G={N_G}")

    rng = np.random.default_rng([EKI_SEED_H, ord(arguments.variant[1])])
    thetas = clip_theta_h(
        initial_ensemble_h(members, rng, t0), bounds
    )

    pool = Pool(
        processes=arguments.processes,
        initializer=_init_worker_h,
        initargs=(duration_s, 20260801, calibration),
    )
    if PATH_PARALLEL:
        _init_worker_h(duration_s, 20260801, calibration)  # main-process ctx
        print(f"[{tag}] path-parallel evaluation: {FORWARD_PATHS} tasks "
              f"per member, {arguments.processes} processes")
    try:
        # ------------------------------------------------ Gamma (spec)
        import sw_gamma_unified as gu
        gamma_cache = output / (f"gamma_cache_{GAMMA_MODE}_p{FORWARD_PATHS}"
                                f"_nf{N_FWD_REPEATS}.npz")
        theta_repeat = np.mean(thetas, axis=0)
        started = time.perf_counter()
        repeat_stats = None
        fwd_cache = gu.fwd_cache_path(
            OUT_ROOT, tag, theta_repeat, EKI_SEED_H, FORWARD_PATHS,
            N_FWD_REPEATS, "incr")
        cached_fwd = gu.load_fwd_cache(fwd_cache, q_obs)
        if cached_fwd is not None:
            repeat_stats = list(cached_fwd)
            print(f"[{tag}] forward repeats loaded from cache "
                  f"({len(repeat_stats)} repeats, {fwd_cache.name}); "
                  f"Gamma rebuilt with current settings")
        if repeat_stats is None:
            repeat_tasks = [
                (theta_repeat, arguments.variant,
                 (EKI_SEED_H, 2999, repeat))
                for repeat in range(N_FWD_REPEATS)
            ]
            repeat_stats = [
                s for s in forward_statistics_h_batch(pool, repeat_tasks)
                if s is not None
            ]
            if len(repeat_stats) < max(3, N_FWD_REPEATS // 2):
                raise RuntimeError("too many failed Gamma repeats")
            gu.save_fwd_cache(fwd_cache, np.asarray(repeat_stats),
                              theta_repeat, note="S2 prior-mean repeats")
        ref_stats = gu.load_ref_stats(TRUTH_DIR, "incr", q_obs)
        gamma, gamma_diag_info = gu.build_gamma_unified(
            observation, ref_stats, np.asarray(repeat_stats),
            "incr", n_dense=N_DENSE,
        )
        print(gu.acceptance_line(gamma_diag_info))
        (output / "gamma_unified_diag.json").write_text(
            json.dumps(gamma_diag_info, indent=2), encoding="utf-8")
        np.savez(gamma_cache, gamma=gamma)
        gamma_inverse = np.linalg.inv(gamma)
        print(f"[{tag}] Gamma (spec, var_ref only) from "
              f"{np.asarray(ref_stats).shape[0]} reference records, "
              f"N_F={len(repeat_stats)} diagnostic repeats "
              f"({time.perf_counter()-started:.0f}s)")

        # ---------------------------------- resume state (checkpoint)
        sentinel = 10.0 * np.sqrt(np.diag(gamma))
        history_phi: list[np.ndarray] = []
        means: list[float] = []
        phi_ens_hist: list[float] = []
        best = {"phi": np.inf, "theta": None}
        counters = {"fail": 0, "eval": 0}
        checkpoint_path = output / "checkpoint.npz"
        start_iteration = 0
        if arguments.resume and checkpoint_path.exists():
            ck = np.load(checkpoint_path)
            thetas = ck["thetas"]
            history_phi = [row.copy() for row in ck["history_phi"]]
            means = [float(v) for v in ck["means"]]
            best = {"phi": float(ck["best_phi"]),
                    "theta": ck["best_theta"].copy()}
            counters["fail"] = int(ck["fail_total"])
            counters["eval"] = int(ck["eval_total"])
            rng.bit_generator.state = json.loads(str(ck["rng_state"]))
            start_iteration = int(ck["next_iteration"])
            print(f"[{tag}] resumed at iteration {start_iteration} "
                  f"(best Phi so far {best['phi']:.2f})")

        def save_checkpoint(thetas_ck, next_iteration, rng_obj):
            """Origin checkpoint layout, byte-compatible keys."""
            np.savez(
                checkpoint_path,
                thetas=thetas_ck,
                history_phi=np.asarray(history_phi),
                means=np.asarray(means),
                best_phi=best["phi"],
                best_theta=(best["theta"] if best["theta"] is not None
                            else np.full(THETA_DIM, np.nan)),
                fail_total=counters["fail"],
                eval_total=counters["eval"],
                next_iteration=next_iteration,
                rng_state=json.dumps(rng_obj.bit_generator.state),
            )

        # -------------------------------------------- engine injections
        def evaluator(theta_ens, engine_iteration):
            """One EKI evaluation: J x N_G tasks, CRN seeds shared across
            members.  A member fails only when ALL its N_G realisations
            failed; it returns a NaN row for the engine's failure policy.
            Also writes the per-iteration audit file and the pre-update
            checkpoint (so a stop leaves the evaluated state on disk)."""
            iteration = start_iteration + engine_iteration
            started = time.perf_counter()
            n_members = theta_ens.shape[0]
            tasks = [(theta_ens[m], arguments.variant,
                      spec_seed_root(EKI_SEED_H, iteration, a))
                     for m in range(n_members) for a in range(N_G)]
            flat = forward_statistics_h_batch(pool, tasks, chunksize=1)
            rows = np.empty((n_members, q_obs), dtype=float)
            n_lost, lost_members = 0, 0
            for m in range(n_members):
                ok = [flat[m * N_G + a] for a in range(N_G)
                      if flat[m * N_G + a] is not None]
                n_lost += N_G - len(ok)
                if ok:
                    rows[m] = np.mean(np.asarray(ok), axis=0)
                else:
                    rows[m] = np.nan
                    lost_members += 1
            if n_lost:
                print(f"[{tag}] {n_lost}/{n_members * N_G} realisations "
                      f"failed ({lost_members} members lost entirely)",
                      flush=True)
            counters["fail"] += lost_members
            counters["eval"] += n_members
            # Audit copy with the declared penalty rows swapped in (same
            # replacement the engine applies afterwards).  Phi is
            # recomputed with the origin's explicit-inverse formula so the
            # audit files stay bit-identical to the origin's; the engine's
            # internal Phi (Cholesky whitening of the exact Gamma, drives
            # the stop rule) differs from it at the last-ulp level only.
            g_matrix = rows.copy()
            failed = ~np.all(np.isfinite(g_matrix), axis=1)
            g_matrix[failed] = observation + sentinel
            residuals = g_matrix - observation
            phis = 0.5 * np.einsum(
                "ij,jk,ik->i", residuals, gamma_inverse, residuals
            )
            r_bar = g_matrix.mean(axis=0) - observation
            phi_ensemble = float(0.5 * r_bar @ gamma_inverse @ r_bar)
            phi_ens_hist.append(phi_ensemble)
            history_phi.append(phis.copy())
            for m in np.nonzero(~failed)[0]:
                if phis[m] < best["phi"]:
                    best["phi"] = float(phis[m])
                    best["theta"] = theta_ens[m].copy()
            valid_phis = phis[~failed] if (~failed).any() else phis
            mean_phi = float(np.mean(valid_phis))
            means.append(mean_phi)
            print(f"[{tag}] iter {iteration}: "
                  f"Phi(mean G)={phi_ensemble:.2f} "
                  f"mean Phi={mean_phi:.2f} "
                  f"min={float(np.min(valid_phis)):.2f} "
                  f"failures={lost_members} "
                  f"({time.perf_counter()-started:.0f}s)", flush=True)
            np.savez(
                output / f"iter_{iteration:03d}.npz",
                thetas=theta_ens,
                g_matrix=g_matrix,
                phis=phis,
                phi_ensemble=phi_ensemble,
            )
            # Pre-update checkpoint: keeps best/counters in sync even when
            # the stop rule fires before the theta update.  Skipped for the
            # engine's extra final-ensemble evaluation (cap case) so the
            # stored next_iteration never exceeds the iteration cap.
            if engine_iteration < n_iter_left:
                save_checkpoint(theta_ens, iteration + 1, rng)
            return rows

        def failed_mask(outputs):
            return ~np.all(np.isfinite(outputs), axis=1)

        def sentinel_row(y, gamma_diag):
            return y + 10.0 * np.sqrt(gamma_diag)

        def spec_clip(theta_ens, engine_iteration):
            return clip_theta_h(theta_ens, bounds)

        def on_update(cb):
            """Post-update checkpoint (the resumable state)."""
            iteration = start_iteration + cb["iteration"]
            save_checkpoint(cb["thetas"], iteration + 1, cb["rng"])

        n_iter_left = max(max_iterations - start_iteration, 0)
        # jitter=0.0 reproduces the origin's un-regularised Kalman-gain
        # solve.  For the objective it is redundant-but-harmless: the
        # engine always computes Phi with the exact Gamma (Cholesky
        # whitening), independent of the jitter setting.
        result = run_eki(
            thetas,
            None,
            observation,
            gamma,
            n_iter=n_iter_left,
            rng=rng,
            perturb_observations=True,
            jitter=0.0,
            verbose=True,
            ensemble_evaluator=evaluator,
            stop_rel_tol=0.01,
            stop_patience=3,
            sentinel_row_fn=sentinel_row,
            failed_mask_fn=failed_mask,
            post_update=spec_clip,
            iteration_callback=on_update,
            jitter_mode="absolute",
        )
        print(f"[{tag}] engine stop: {result.stop_reason} "
              f"({result.n_updates} updates applied)")
    finally:
        pool.close()
        pool.join()

    thetas = result.final_ensemble
    fail_total, eval_total = counters["fail"], counters["eval"]
    if eval_total and fail_total / eval_total > 0.10:
        print(f"[{tag}] WARNING: path failure rate "
              f"{fail_total/eval_total:.1%} > 10% -- pre-registered "
              f"remedy: tighten node bounds to 3*T0 and rerun")

    theta_best = best["theta"]
    if theta_best is None:
        raise SystemExit(f"[{tag}] every member failed at every iteration")
    decoded_best = decode_theta_h(theta_best)
    recovery = recovery_metrics(decoded_best, calibration["s0"])
    # Decoded ensemble statistics (model coordinates) for the spec report.
    _decoded = [decode_theta_h(t) for t in thetas]
    _scalar = [k for k, v in _decoded[0].items()
               if not isinstance(v, np.ndarray)]
    decoded_mean_spec = {k: float(np.mean([d[k] for d in _decoded]))
                         for k in _scalar}
    decoded_sd_spec = {k: float(np.std([d[k] for d in _decoded], ddof=1))
                       for k in _scalar}
    _nodes = np.array([d["node_values"] for d in _decoded])
    decoded_mean_spec["node_values"] = _nodes.mean(0).tolist()
    decoded_sd_spec["node_values"] = _nodes.std(0, ddof=1).tolist()

    # In-loop evaluations (origin meaning across resume segments): the
    # engine adds one final-ensemble evaluation when the cap is reached.
    iterations_run = int(min(len(history_phi), max_iterations))

    summary = {
        "phiq_prior": s1_phiq_prior() or "uniform U(0.003,0.03) x U(0,5)",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "driver": "sw_eki_h",
        "variant": arguments.variant,
        "calibration": calibration,
        "members": members,
        "iterations_run": iterations_run,
        "n_updates": int(result.n_updates),
        "stop_reason": result.stop_reason,
        "forward_paths": FORWARD_PATHS,
        "gamma_mode": GAMMA_MODE,
        "phi_best": best["phi"],
        "q_obs": int(q_obs),
        "theta_best": theta_best.tolist(),
        "theta_mean": np.mean(thetas, axis=0).tolist(),
        "final_parameter_spread":
            np.std(thetas, axis=0, ddof=1).tolist(),
        # Spec 2026-08-23: final ensemble mean +/- sd of the DECODED
        # (model-coordinate) parameters.
        "decoded_final_mean": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in decoded_mean_spec.items()},
        "decoded_final_sd": decoded_sd_spec,
        "algorithm": "spec_2026-08-23",
        "parameter_choice": "final_ensemble_mean",
        "N_G": N_G,
        # This resume segment's in-loop Phi(mean G) values; the engine's
        # extra final-ensemble evaluation (cap case) is reported apart.
        "phi_ensemble_history":
            [float(v) for v in phi_ens_hist[:n_iter_left]],
        "phi_ensemble_final_evaluation": (
            float(phi_ens_hist[-1])
            if len(phi_ens_hist) > n_iter_left else None
        ),
        "log_coordinates": ["phi", "obs_noise", "amplitude", "l_u", "l_s"],
        "decoded_best": {k: (v.tolist() if isinstance(v, np.ndarray)
                             else v) for k, v in decoded_best.items()},
        "recovery": recovery,
        "nodes": NODES.tolist(),
        "failure_rate": (fail_total / eval_total) if eval_total else 0,
    }
    diag_path = output / "gamma_unified_diag.json"
    if diag_path.exists():
        import sw_gamma_unified as gu
        summary.update(gu.summary_fields(
            json.loads(diag_path.read_text(encoding="utf-8"))))
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"[{tag}] best Phi={best['phi']:.2f}")
    print(f"[{tag}] recovery={json.dumps(recovery)[:400]}")


if __name__ == "__main__":
    main()
