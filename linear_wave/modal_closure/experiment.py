"""Case definition and EKI driver for the broadband linear-wave modal closure
(thesis Chapter 4): truth spectrum, 38-dim statistics vector, y/Gamma build,
N_G calibration, forward model, parallel evaluation, and the spec-2026-08-23
EKI run.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

# Bootstrap the code_rp root onto sys.path BEFORE importing the shared
# algorithms package.  run_closure.py does the same, but a spawned worker
# re-imports this module directly and must be able to resolve `algorithms`
# on its own.
_CODE_RP_ROOT = str(Path(__file__).resolve().parents[2])
if _CODE_RP_ROOT not in sys.path:
    sys.path.insert(0, _CODE_RP_ROOT)

from algorithms.eki import run_eki
from algorithms.gamma import build_gamma
from algorithms.statistics import band_energy_spectrum as _band_energy
from algorithms.statistics import cross_corr, normalized_autocorr

from . import bundle_io
from .numerics import dispersion as _disp
from .numerics import modal_propagator as _modal_propagator
from .truth import _solve_wavenumber, get_linear_data


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

# Truth: N broadband random (incommensurate) frequencies with a smooth envelope
N_COMPONENTS = 100                        # number of truth wave components (n)
F_MIN, F_MAX = 0.3, 1.8                   # frequency band (Hz)
TRUTH_FREQ_SEED = 7                       # seed for the (fixed) random frequency draw
SPEC_PEAK_HZ = 1.0                        # peak of the amplitude envelope (Hz)
SPEC_WIDTH_HZ = 0.35                      # width of the amplitude envelope (Hz)
TARGET_VAR = 1.0                          # total surface-elevation variance sum(a_j^2/2)
DEPTH = 100.0                             # water depth (m)

# Draw the truth frequencies once (random reals -> generically incommensurate
# -> the truth field is not periodic).
_freq_rng = np.random.default_rng(TRUTH_FREQ_SEED)
FREQUENCIES = np.sort(_freq_rng.uniform(F_MIN, F_MAX, N_COMPONENTS))
_envelope = np.exp(-0.5 * ((FREQUENCIES - SPEC_PEAK_HZ) / SPEC_WIDTH_HZ) ** 2)
AMPLITUDES = _envelope * np.sqrt(TARGET_VAR / np.sum(_envelope ** 2 / 2.0))

K_GAUGES = 10                             # number of gauges
DX = 2.0 / 9.0                            # 10 gauges spanning [0, 2] m (uniform spacing ~0.222 m)
GAUGES = np.arange(K_GAUGES) * DX

DT_DATA = 0.05                            # sampling step of data/statistics (s)
# Analysis-window length, set by TWO rules that pull against each other.
#
# Identifiability (report Sec. 3.3) wants a LONG window: S = |dG| /
# sqrt(var_ref) grows with T, and the rule asks for S >= 1 on every unknown.
# Measured at the solution with 24 repeats per parameter, 18 of the 20 unknowns
# cross S = 1 by 879 s, but delta_6 and delta_3 need 3725 s and 5269 s, because
# their signal sits in the variance/ACF/cross-correlation families whose sd
# falls only like T^-0.3 while the band energies' falls like T^-0.95.
#
# Stability wants a SHORT one.  Gamma = diag(var_ref) shrinks with T, and Gamma
# is what damps the Kalman step; once Gamma << C_GG the update is a full
# Gauss-Newton step, which this nonlinear forward map does not survive.  A
# bisection on Phi_final / (q/2) puts the transition between 1375 s (6.4) and
# 1500 s (35.5); at 2000 s it is 2355 and at 12000 s the inversion diverges
# outright (Phi reached 1.4e18 by the third iteration).
#
# The two rules therefore overlap only on 879-1375 s, and 20/20 is unreachable.
# 1000 s is taken: inside the overlap, Phi_final = 1.65 x q/2, and delta_3 and
# delta_6 are reported through combinations rather than individually.  This
# conflict is a result about the case, not a tuning choice -- see the thesis.
T_RECORD = 1000.0                         # length of each realization (s)
N_DATA = int(round(T_RECORD / DT_DATA))
T_BURN = 12.0                             # burn-in discarded from each simulation (s)

AUTO_LAGS = np.array([1, 2, 3, 4, 6, 8, 12, 16])         # single-gauge autocorrelation lags (steps)
CROSS_LAGS = np.array([-8, -6, -4, -2, 0, 2, 4, 6, 8])   # signed neighbour lags

# Model: M oscillators on a FIXED frequency grid spanning the band (spectral
# reconstruction).  The committed value is 10.  MODAL_CLOSURE_M_MODES overrides
# it for mode-count experiments; it is read from the environment rather than
# passed as an argument because a spawned worker re-imports this module and
# must land on the same grid as the parent.  Note the identifiability ceiling:
# q = 28 + M against 2M unknowns, so M > 28 is underdetermined however long
# the window.
M_MODES = int(os.environ.get("MODAL_CLOSURE_M_MODES", 10))
F_GRID = np.linspace(F_MIN, F_MAX, M_MODES)             # fixed grid frequencies (Hz)
SPEC_DF = (F_MAX - F_MIN) / (M_MODES - 1)              # grid spacing = periodogram band width (Hz)
OMEGA_GRID = 2 * np.pi * F_GRID                         # fixed grid angular frequencies (rad/s)
K_GRID = np.array([_solve_wavenumber(w, DEPTH) for w in OMEGA_GRID])   # wavenumbers (dispersion)
OMEGA2_GRID = OMEGA_GRID ** 2

# Legacy parameter boxes.  The spec fit runs UNBOUNDED in log coordinates, so
# these are never applied by EKI; audit.py keeps them as the fixed yardstick
# for its excursion accounting (how far the final ensemble ranged), exactly as
# in the frozen run.
DELTA_BOUNDS = [(0.02, 8.0)] * M_MODES                  # per-mode bandwidth (1/s)
SIGMA_BOUNDS_SDE = [(0.01, 30.0)] * M_MODES             # per-mode sqrt noise intensity

# 100 against 20 unknowns.  A larger M needs a larger ensemble to keep the
# empirical covariances non-degenerate, so this is overridable alongside M.
ENSEMBLE_SIZE = int(os.environ.get("MODAL_CLOSURE_ENSEMBLE", 100))
N_ITER = 20                               # EKI iterations (cap; the stop rule may fire earlier)

# Worker processes used for the embarrassingly parallel parts: the per-member
# forward evaluations, the reference records behind Gamma, and the audit's
# repeated re-evaluations.  Results are independent of this number: every
# simulation draws from its own SeedSequence keyed by (base seed, iteration,
# replicate), so the serial and parallel paths return identical arrays.
#
# Sized from the machine rather than hard-coded, because the 2026-08-23 spec
# multiplies the forward work by N_G and the run is meant to move to a large
# server.  A laptop keeps two cores free; a many-core box keeps a smaller
# fraction.  MODAL_CLOSURE_WORKERS overrides both.
def available_cpus() -> int:
    """Cores this process may actually use.

    ``os.cpu_count()`` reports the machine, not the allocation: inside a
    container or under taskset it over-reports, and 190 workers on 8 permitted
    cores is slower than 8.  The affinity mask is the honest number on Linux and
    simply absent on Windows.
    """

    if hasattr(os, "sched_getaffinity"):
        try:
            return len(os.sched_getaffinity(0))
        except OSError:
            pass
    return os.cpu_count() or 4


def _default_workers() -> int:
    override = os.environ.get("MODAL_CLOSURE_WORKERS")
    if override:
        return max(1, int(override))
    n = available_cpus()
    return max(1, n - 2) if n <= 32 else n - 4


N_WORKERS = _default_workers()


# Truth-generator jobs -- the reference records behind Gamma, and y itself --
# are the only memory-heavy stage: each holds a (TIME_BLOCK, n_components,
# n_gauges) complex phase block, about 240 MB peak per worker at the committed
# block size.  Model forwards are three orders of magnitude smaller.  Running
# 190 of the former at once would ask for ~45 GB, so this stage gets its own
# cap; it is only N_GAMMA jobs and finishes in well under a minute anyway.
TRUTH_JOB_MEMORY_GB = 0.35        # measured peak per worker, incl. interpreter


def _default_truth_workers() -> int:
    override = os.environ.get("MODAL_CLOSURE_TRUTH_WORKERS")
    if override:
        return max(1, int(override))
    # Size from the memory actually available, not from a guessed constant: a
    # 384 GB server can run all 188 of these at once, a 16 GB laptop cannot.
    budget = None
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    budget = 0.5 * int(line.split()[1]) / (1024.0 ** 2)
                    break
    except OSError:
        budget = None
    if budget is None:
        return min(N_WORKERS, 64)
    return max(1, min(N_WORKERS, int(budget / TRUTH_JOB_MEMORY_GB)))


N_TRUTH_WORKERS = _default_truth_workers()

# Uniform priors for the initial EKI ensemble.  Their midpoint is the
# "representative theta" at which the forward-side term of Gamma is evaluated,
# so it must not depend on the answer.
DELTA_PRIOR = (0.3, 3.0)
SIGMA_PRIOR = (0.5, 6.0)

# --- sizes -----------------------------------------------------------------
# N_GAMMA   reference records behind Gamma = diag(var_ref).  Each gamma_i is
#           then known to sqrt(2/(N-1)); 200 gives 10.0% against 20.2% at 50.
#           A record costs 10.2 s, so 200 of them cost ~3 min on 12 workers.
# N_G       forward runs averaged into one G_hat, per member, per iteration.
#           The spec calibrates it from ratio = var_fwd/var_ref, requiring
#           N_G >= 5*ratio so the neglected var_fwd/N_G stays under 20% of
#           var_ref.  Taken at the 75th percentile of ratio rather than its
#           maximum: the maximum is set by low-energy bands whose var_ref is
#           ~1e-10 because the truth is a sum of fixed-amplitude components and
#           barely varies between records, while a ten-mode white-noise-driven
#           model genuinely does.  That is structural, not a mis-set constant,
#           and the literal rule asks for N_G ~ 5e6.  Measured at N_GAMMA = 200
#           with probes at the log prior mean and the upper corner of the prior
#           box: ratio p50 1.16, p75 6.04, p90 638, max 1.08e6.  The rule at the
#           75th percentile therefore requires N_G >= 31; 100 is taken because
#           it clears that with margin and covers 33 of 38 statistics -- every
#           variance, every ACF lag and all 18 cross-correlations, leaving only
#           five low-energy bands.  calibrate_forward_averaging() re-measures
#           this every run and coverage_report() names what is left out.
# CALIB_K   probe evaluations per theta_probe in that calibration.
N_GAMMA = 200
N_G = 100
CALIB_K = 20
CALIB_QUANTILE = 0.75
# Second probe for the N_G calibration.  The spec asks for "prior mean AND near
# the expected optimum"; in practice the optimum comes from a previous run, so
# this holds that run's theta and run_closure.py can load it from a bundle.
#
# It matters more than it looks.  A corner of the prior box was used at first,
# on the assumption that a point far from the optimum makes the harshest
# demand.  Measured, the opposite is true: at the prior corner the model is so
# energetic that a couple of bands show ratio ~1e5 while the MEDIAN ratio is
# only 0.64, whereas at the solution every statistic is at the truth's scale
# and the model's intrinsic randomness -- white-noise driven, against a truth
# of fixed amplitudes and random phases only -- shows up everywhere, lifting
# the median to 4.64.  The 75th-percentile rule then asks for N_G >= 38 at the
# solution against 16 at the corner.  Probing only far from the optimum
# understates the requirement by a factor of two.
CALIB_PROBE_THETA = None
CALIB_SAFETY = 5.0          # the spec's factor: N_G >= CALIB_SAFETY * ratio
CALIB_TOLERANCE = 0.20      # ...which is what makes var_fwd/N_G <= 20% var_ref

# --- EKI --------------------------------------------------------------------
# Positive unknowns are held as log theta throughout and exponentiated before
# each forward run, so the update cannot drive a rate negative and no clipping
# is needed.  With no bounds there is no bound-activity to report and the
# penalty sentinel becomes a pure numerical guard.
LOG_COORDINATES = True
STOP_REL_TOL = 0.01         # stop when |dPhi|/Phi stays under this ...
STOP_PATIENCE = 3           # ... for this many consecutive iterations
N_SINGLE_REALISATIONS = 50  # unaveraged G(theta_hat) draws stored for error-bar comparisons

T_LONG = 300.0                            # long validation run length (s)
ENS_MEMBERS = 100                         # ensemble members overlaid in the stored ensembles
T_LONG_ENS = 100.0                        # long-run length per overlay member (s)
OBSERVATION_SEED = 22001                  # the single reference record behind y
EKI_SEED = 42
TRUTH_LONG_SEED = 1021
SDE_LONG_SEED = EKI_SEED + 7
SDE_ENSEMBLE_SEED0 = EKI_SEED + 1000
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "modal_closure"

# --- disjoint seed blocks ---------------------------------------------------
# The spec requires the record behind y, the records behind Gamma and the
# forward randomness to be drawn from streams that cannot overlap, so that y is
# independent of its own error bar.  Each block below is a decade wide and the
# blocks are separated by 10 000; check_seed_blocks() asserts the separation
# rather than leaving it to the reader.
GAMMA_REF_SEED = 30000        # the N_GAMMA reference records behind var_ref
FORWARD_CRN_SEED = 40000      # S(n, a): the common random numbers of G_hat
OBS_PERTURB_SEED = 50000      # eta ~ N(0, Gamma), per member per iteration
INIT_ENSEMBLE_SEED = 60000    # the log-uniform initial ensemble
SINGLE_REAL_SEED = 70000      # unaveraged G(theta_hat) draws kept for error-bar figures
CALIB_SEED = 80000            # the K probe runs of the N_G calibration
SEED_BLOCK_WIDTH = 10000


def dispersion(k):
    """Linear dispersion omega^2 = g k tanh(k h) at this experiment's depth."""

    return _disp(k, DEPTH)


def band_energy_spectrum(eta_field):
    """Periodogram band energy at this experiment's fixed grid frequencies."""

    return _band_energy(eta_field, F_GRID, SPEC_DF, DT_DATA)


def spatial_statistics(eta_field, v_field, auto_lags, cross_lags):
    """Assemble the summary-statistic vector (variances + ACF + cross-corr +
    band spectrum), in the fixed order given by statistic_labels()."""

    K = eta_field.shape[1]
    acf = np.zeros(len(auto_lags))
    for k in range(K):
        acf += normalized_autocorr(eta_field[:, k], auto_lags)
    acf /= K
    return np.concatenate([
        [float(eta_field.var()), float(v_field.var())],
        acf,
        cross_corr(eta_field, 1, cross_lags),
        cross_corr(eta_field, 2, cross_lags),
        band_energy_spectrum(eta_field),
    ])


# non-spectral part of the statistic vector
N_BASE_STATS = 2 + len(AUTO_LAGS) + 2 * len(CROSS_LAGS)

# Statistic families (name, start, stop) for the effective-DOF correction.
STAT_FAMILIES = (
    ("variance", 0, 2),
    ("acf", 2, 2 + len(AUTO_LAGS)),
    ("xcorr", 2 + len(AUTO_LAGS), N_BASE_STATS),
    ("band", N_BASE_STATS, N_BASE_STATS + M_MODES),
)


def statistic_labels() -> list[str]:
    """Names of the q statistics, in the order used by spatial_statistics.

    Defined here rather than in audit.py because this module is what fixes the
    order; audit re-exports it so the old import path keeps working.
    """

    return (
        ["var_eta", "var_v"]
        + [f"acf_lag{int(lag)}" for lag in AUTO_LAGS]
        + [f"xcorr1_lag{int(lag):+d}" for lag in CROSS_LAGS]
        + [f"xcorr2_lag{int(lag):+d}" for lag in CROSS_LAGS]
        + [f"band_{f:.4f}Hz" for f in F_GRID]
    )


# Truth data, observation y, error bars Gamma
def truth_fields(n_real, seed):
    output_times = np.arange(N_DATA) * DT_DATA
    return get_linear_data(
        gauge_locations=GAUGES, frequencies=FREQUENCIES, amplitudes=AMPLITUDES,
        output_times=output_times, depth=DEPTH, runs=n_real, random_seed=seed)


def truth_fields_at(output_times, seed):
    """Truth fields sampled on an arbitrary time grid (used by the audit
    diagnostics, which need step sizes other than DT_DATA)."""

    eta_list, v_list = get_linear_data(
        gauge_locations=GAUGES, frequencies=FREQUENCIES, amplitudes=AMPLITUDES,
        output_times=np.asarray(output_times, dtype=float), depth=DEPTH, runs=1,
        random_seed=seed)
    return eta_list[0], v_list[0]


def long_truth_field(seed):
    n = int(round(T_LONG / DT_DATA))
    output_times = np.arange(n) * DT_DATA
    eta_list, v_list = get_linear_data(
        gauge_locations=GAUGES, frequencies=FREQUENCIES, amplitudes=AMPLITUDES,
        output_times=output_times, depth=DEPTH, runs=1, random_seed=seed)
    return eta_list[0], v_list[0]


def observation_record(seed, n_data=None):
    """Statistics of ONE reference record over the full analysis window."""

    n = N_DATA if n_data is None else int(n_data)
    eta, v = truth_fields_at(np.arange(n) * DT_DATA, seed)
    return spatial_statistics(eta, v, AUTO_LAGS, CROSS_LAGS)


# --- log coordinates (2026-08-23 spec) -------------------------------------
# All twenty unknowns -- ten damping rates and ten noise amplitudes -- are
# positive, so EKI runs on xi = log theta and every forward call exponentiates.
# The Kalman update is unconstrained in xi, which is what lets the bounds and
# the clipping go away; a member can become very small but never negative.


def to_log(theta):
    return np.log(np.asarray(theta, dtype=float))


def from_log(xi):
    return np.exp(np.asarray(xi, dtype=float))


def as_physical(theta):
    """Interpret an EKI parameter vector in physical units."""

    return from_log(theta)


def log_prior_mean_parameters():
    """Geometric mean of the prior interval: the representative theta in log
    coordinates, and one of the two probes the N_G calibration requires.  It
    depends on no fitted result."""

    return np.concatenate([
        np.full(M_MODES, float(np.sqrt(DELTA_PRIOR[0] * DELTA_PRIOR[1]))),
        np.full(M_MODES, float(np.sqrt(SIGMA_PRIOR[0] * SIGMA_PRIOR[1]))),
    ])


def calibration_probes():
    """The two theta at which the N_G calibration measures var_fwd/var_ref.

    The spec asks for two probes: one representative of the prior, and one near
    the expected optimum.  The first is the geometric mean of the prior box.
    For the second, CALIB_PROBE_THETA is used when a previous run's theta is
    available; without one the energetic corner of the prior box stands in
    (least damping, most noise, hence the largest Var(q) = sigma/(2 delta w^2)),
    and run_experiment prints the warning that this understates the
    requirement.  Both are returned in PHYSICAL units, because
    calibrate_forward_averaging dispatches its probe jobs with log_coords=False.
    """

    probes = {"log_prior_mean": log_prior_mean_parameters()}
    if CALIB_PROBE_THETA is None:
        # The UPPER corner of the prior box on both coordinates.  Verified
        # against the archived M=10 calibration, which it reproduces exactly
        # (p50 0.9407, p75 2.6962, p90 56.4813, max 61843.5 -> N_G >= 14).
        probes["prior_corner"] = np.concatenate([
            np.full(M_MODES, float(DELTA_PRIOR[1])),
            np.full(M_MODES, float(SIGMA_PRIOR[1])),
        ])
    else:
        probes["previous_solution"] = np.asarray(CALIB_PROBE_THETA, dtype=float)
    return probes


def check_seed_blocks():
    """Fail loudly if two seed blocks could produce the same random stream.

    y must be independent of its own error bar, and the forward randomness must
    be independent of both.  The spec states this; this function is what makes
    it a property of the code rather than of the reader's attention.
    """

    blocks = {
        "observation_y": (OBSERVATION_SEED, 1),
        "gamma_reference": (GAMMA_REF_SEED, N_GAMMA),
        "forward_crn": (FORWARD_CRN_SEED, SEED_BLOCK_WIDTH),
        "observation_perturbation": (OBS_PERTURB_SEED, 1),
        "initial_ensemble": (INIT_ENSEMBLE_SEED, 1),
        "single_realisations": (SINGLE_REAL_SEED, N_SINGLE_REALISATIONS),
        "calibration": (CALIB_SEED, SEED_BLOCK_WIDTH),
    }
    spans = sorted((start, start + max(width, 1), name)
                   for name, (start, width) in blocks.items())
    for (lo1, hi1, a), (lo2, _, b) in zip(spans, spans[1:]):
        if hi1 > lo2:
            raise ValueError(
                f"seed blocks '{a}' [{lo1}, {hi1}) and '{b}' starting at {lo2} "
                "overlap; y would not be independent of Gamma")
    return {name: [start, start + max(width, 1)] for name, (start, width)
            in blocks.items()}


# Build the observation y and the diagonal error covariance Gamma.
#
# Returns (y, gamma_diag, parts).  `parts` exposes every term so that the audit
# module can reuse this construction instead of reimplementing it -- the two
# drifting apart is exactly the failure mode this signature prevents.


def build_reference_error_model(observation_seed=OBSERVATION_SEED,
                                n_workers=N_WORKERS,
                                n_gamma=None):
    """y and Gamma under the 2026-08-23 spec.

    Gamma = diag(var_ref) and nothing else.  No forward term, because the model
    fluctuation is averaged away by N_G instead of budgeted for; no discrepancy
    floor; no effective-DOF factor.  Gamma is then a pure statement about how
    much the reference itself moves from record to record, which is also the
    quantity the window rule of Sec. 3.3 is written in -- so the window and the
    error bar are finally the same object.

    The variance estimate itself is delegated to algorithms.gamma.build_gamma
    (structure='diagonal'): the identical ddof=1 per-component sample variance,
    with the spec's rank/condition diagnostics recorded into ``parts``.
    """

    n_gamma = N_GAMMA if n_gamma is None else int(n_gamma)
    check_seed_blocks()
    y = observation_record(observation_seed)

    # Truth-generator jobs, so the memory-capped pool rather than the full one.
    ref_workers = min(int(n_workers), N_TRUTH_WORKERS)
    reference_stats = np.asarray(_mapper(ref_workers)(
        _reference_job,
        [(GAMMA_REF_SEED + i, N_DATA) for i in range(n_gamma)]))

    # With no floor there is nothing to rescue a zero.  Say so by name BEFORE
    # handing the records to build_gamma (whose correlation diagnostics would
    # divide by the zero first); quietly clamping would reintroduce a floor
    # under another word.
    dead = np.flatnonzero(reference_stats.var(axis=0, ddof=1) <= 0.0)
    if dead.size:
        names = ", ".join(statistic_labels()[i] for i in dead)
        raise ValueError(
            f"var_ref vanished for {dead.size} statistic(s) ({names}); Gamma "
            "would be singular.  Either those statistics are constant across "
            "reference records and must be dropped from q, or N_GAMMA is too "
            "small.")

    gamma_est = build_gamma(reference_stats, structure="diagonal")
    var_ref = gamma_est.var_ref

    parts = {
        "var_ref": var_ref,
        "var_fwd": np.zeros_like(var_ref),
        "floor": np.zeros_like(var_ref),
        "gamma_sampling": var_ref.copy(),
        "floor_dominates": np.zeros(var_ref.size, dtype=bool),
        "reference_stats": reference_stats,
        "forward_stats": np.empty((0, var_ref.size)),
        "theta_forward": None,
        "forward_theta_is_prior_mean": True,
        "neff_correction": False,
        "neff_factors_by_family": {},
        "n_gamma": n_gamma,
        "relative_precision": float(np.sqrt(2.0 / (n_gamma - 1))),
        "gamma_type": "reference_variance_only",
        # Diagnostics from the shared estimator (diag Gamma: rank always q).
        "gamma_condition_number": gamma_est.condition_number,
        "gamma_corr_min_eigenvalue": gamma_est.corr_min_eigenvalue,
    }
    return y, var_ref.copy(), parts


def calibrate_forward_averaging(var_ref, theta_probes=None, k=None,
                                n_workers=N_WORKERS, quantile=None):
    """How many forward runs one G_hat must average, per the spec's rule.

    The spec probes ratio = var_fwd/var_ref at two or more theta and demands
    N_G >= CALIB_SAFETY * ratio, which keeps the neglected var_fwd/N_G below
    CALIB_TOLERANCE of var_ref.  Applied to the maximum over components, this
    case asks for N_G ~ 1e5: the low-energy bands have var_ref ~ 1e-10 because
    the truth is a sum of a hundred fixed-amplitude components whose band
    energy barely moves between records, while a ten-mode white-noise-driven
    model's genuinely does.  The rule is therefore applied at a quantile of the
    ratio, and the components it does not cover are named in the return value
    so the report can state them instead of implying full coverage.

    Kept local rather than delegated to algorithms.gamma.calibrate_n_g: that
    reference implementation applies the literal max-ratio rule (with a
    near-degeneracy drop), takes precomputed probe statistics, and has no
    sentinel filtering -- the quantile convention above is this case's
    documented deviation and lives here with its justification.
    """

    var_ref = np.asarray(var_ref, dtype=float)
    k = CALIB_K if k is None else int(k)
    quantile = CALIB_QUANTILE if quantile is None else float(quantile)
    if theta_probes is None:
        theta_probes = {"log_prior_mean": log_prior_mean_parameters()}
    map_fn = _mapper(n_workers)

    per_probe = {}
    for p, (name, theta) in enumerate(theta_probes.items()):
        stats = np.asarray(map_fn(
            _forward_job,
            [(np.asarray(theta, dtype=float), (CALIB_SEED, p, i),
              var_ref.size, 1, T_RECORD, False) for i in range(k)]))
        valid = ~np.all(stats == FORWARD_SENTINEL, axis=1)
        if valid.sum() < 2:
            raise ValueError(f"probe '{name}' produced fewer than two usable "
                             "forward runs; cannot estimate var_fwd")
        per_probe[name] = {
            "theta": np.asarray(theta, dtype=float),
            "var_fwd": stats[valid].var(axis=0, ddof=1),
            "n_valid": int(valid.sum()),
        }

    ratio = np.max([d["var_fwd"] / var_ref for d in per_probe.values()], axis=0)
    recommended = int(np.ceil(CALIB_SAFETY * np.quantile(ratio, quantile)))
    return {
        "ratio": ratio,
        "per_probe": per_probe,
        "k": k,
        "quantile": quantile,
        "recommended_n_g": recommended,
        "n_g_for_full_coverage": int(np.ceil(CALIB_SAFETY * ratio.max())),
    }


def coverage_report(ratio, n_g):
    """Which statistics the chosen N_G actually covers, and which it does not."""

    ratio = np.asarray(ratio, dtype=float)
    covered = ratio <= CALIB_TOLERANCE * float(n_g)
    labels = statistic_labels()
    by_family = {name: [int(covered[a:b].sum()), int(b - a)]
                 for name, a, b in STAT_FAMILIES}
    return {
        "n_g": int(n_g),
        "covered": int(covered.sum()),
        "total": int(ratio.size),
        "by_family": by_family,
        "uncovered": [labels[i] for i in np.flatnonzero(~covered)],
        "worst_residual_fraction": float((ratio / float(n_g)).max()),
    }


# Kept as the public name for the error model; the construction itself lives in
# build_reference_error_model above.  audit.py imports this one.
def build_observation_and_gamma(observation_seed=OBSERVATION_SEED,
                                n_workers=N_WORKERS):
    return build_reference_error_model(observation_seed, n_workers)


def report_error_model(y, gamma_diag, parts):
    """Acceptance printout for the error model."""

    print(f"  q={y.size}  N_Gamma={parts['n_gamma']}  Gamma = diag(var_ref), "
          f"reference side only (no forward term, no floor, no n_eff)")
    print(f"  gamma min/median/max = {gamma_diag.min():.3e}/"
          f"{np.median(gamma_diag):.3e}/{gamma_diag.max():.3e}   "
          f"each gamma_i known to {parts['relative_precision']:.1%}", flush=True)
    return 0.0


# Model: M fixed-grid oscillators + spatial-phase readout (exact OU propagator).
# Integrate the M fixed-grid oscillators exactly and read out the two
# observation channels at the gauges.  The second channel is the modal-skeleton
# velocity functional of Eq. (5), not a pathwise time derivative of the
# simulated eta: eta contains p_j, p_j is driven by white noise, so eta is not
# differentiable in time.  See audit.velocity_readout_audit for the exact Ito
# accounting.
def simulate_grid(delta, sqrt_sigma, *, rng, t_record=None, t_burn=T_BURN, dt=DT_DATA):
    # Resolved at call time rather than bound as a default, so the analysis
    # window length stays a single point of control.
    t_record = T_RECORD if t_record is None else float(t_record)
    delta = np.asarray(delta, dtype=float)
    sigma = np.asarray(sqrt_sigma, dtype=float) ** 2
    omega2, omega = OMEGA2_GRID, OMEGA_GRID
    M = M_MODES

    kx = np.outer(GAUGES, K_GRID)
    cq = np.cos(kx)
    cv = -np.sin(kx) / omega
    dq = np.sin(kx) * omega
    dv = np.cos(kx)

    phi, noise_root = _modal_propagator(omega2, delta, sigma, dt)
    p00, p01, p10, p11 = phi[:, 0, 0], phi[:, 0, 1], phi[:, 1, 0], phi[:, 1, 1]
    l00, l01, l10, l11 = (noise_root[:, 0, 0], noise_root[:, 0, 1],
                          noise_root[:, 1, 0], noise_root[:, 1, 1])

    n_burn = int(round(t_burn / dt))
    n_steps = int(round(t_record / dt))
    if np.any(delta <= 0.0) or np.any(sigma <= 0.0):
        raise ValueError("The modal closure requires positive damping and noise intensity")
    z_all = rng.normal(size=(n_burn + n_steps, M, 2))

    # Start at the exact stationary Gaussian law.  Burn-in is retained only as
    # a conservative numerical check and does not create the invariant measure.
    q = rng.normal(size=M) * np.sqrt(sigma / (2.0 * delta * omega2))
    qd = rng.normal(size=M) * np.sqrt(sigma / (2.0 * delta))
    q_store = np.empty((n_steps, M))
    qd_store = np.empty((n_steps, M))
    for n in range(n_burn + n_steps):
        z0, z1 = z_all[n, :, 0], z_all[n, :, 1]
        q_new = p00 * q + p01 * qd + l00 * z0 + l01 * z1
        qd_new = p10 * q + p11 * qd + l10 * z0 + l11 * z1
        q, qd = q_new, qd_new
        if not np.all(np.isfinite(q)) or np.any(np.abs(q) > 1e8):
            raise FloatingPointError("grid oscillators blew up")
        if n >= n_burn:
            q_store[n - n_burn], qd_store[n - n_burn] = q, qd
    eta = q_store @ cq.T + qd_store @ cv.T
    v = q_store @ dq.T + qd_store @ dv.T
    return eta, v


def unpack(theta):
    """Split the flat EKI parameter vector theta into (delta[M], sqrt_sigma[M])."""

    t = np.asarray(theta, dtype=float)
    return t[0:M_MODES], t[M_MODES:2 * M_MODES]


def recovered_spectrum(delta, sqrt_sigma):
    """Recovered power spectrum: modal energy Var(q_j) = sigma_j / (2 delta_j
    omega_j^2) at the grid frequencies."""

    sigma = np.asarray(sqrt_sigma, dtype=float) ** 2
    delta = np.asarray(delta, dtype=float)
    return np.where(delta > 0, sigma / (2.0 * delta * OMEGA2_GRID), 0.0)


def truth_reference():
    """Truth reference: each component's frequency, wavenumber and energy a_j^2/2."""

    omega = 2 * np.pi * FREQUENCIES
    k = np.array([_solve_wavenumber(w, DEPTH) for w in omega])
    energy = AMPLITUDES ** 2 / 2.0
    return {"freq": FREQUENCIES, "omega": omega, "k": k, "energy": energy,
            "total_var": float(energy.sum())}


def bin_truth_to_grid(truth):
    """Bin the truth component energies onto the model grid (nearest grid
    frequency)."""

    idx = np.argmin(np.abs(truth["freq"][:, None] - F_GRID[None, :]), axis=1)
    binned = np.zeros(M_MODES)
    for j, i in enumerate(idx):
        binned[i] += truth["energy"][j]
    return binned


# Sentinel returned by the forward map when a parameter vector blows up.  It is
# a penalty, not a statistic: a sentinel row contaminates the empirical C^GG
# used by the Kalman gain.  The value is exported so run_eki can count how
# often it fires; the frozen run never triggered it (0 sentinel evaluations in
# audit_metrics.json).
FORWARD_SENTINEL = 1e6


def make_forward(observation_size, n_avg=1, t_record=None):
    """Build the EKI forward map theta -> G(theta), with a sentinel penalizing
    blow-ups."""

    bad = np.full(observation_size, FORWARD_SENTINEL)

    min_len = AUTO_LAGS.max() + abs(CROSS_LAGS).max() + 5

    def forward(theta, local_rng):
        # ONE simulation per evaluation, over the same window and gauge count
        # as y.  The resulting sampling noise is carried by var_fwd in Gamma
        # rather than suppressed by averaging.
        try:
            params = unpack(theta)
            accumulated = np.zeros(observation_size)
            for _ in range(n_avg):
                eta_f, v_f = simulate_grid(*params, rng=local_rng, t_record=t_record)
                if eta_f.shape[0] < min_len:
                    return bad.copy()
                stats = spatial_statistics(eta_f, v_f, AUTO_LAGS, CROSS_LAGS)
                if not np.all(np.isfinite(stats)):
                    return bad.copy()
                accumulated += stats
            return accumulated / n_avg
        except (FloatingPointError, np.linalg.LinAlgError, ValueError):
            return bad.copy()

    return forward


# Parallel execution helpers
# --------------------------------------------------------------------------
# These are module-level so they survive pickling to spawned workers.  Each job
# carries its own seed key, which is what makes the parallel result
# reproducible and identical to the serial one.


def _forward_job(job):
    """Evaluate G(theta) once, with an explicitly keyed random stream.

    The window length travels in the job rather than being read from the
    module, because a spawned worker re-imports this module and would otherwise
    silently use the committed T_RECORD instead of whatever the caller is
    using.  The coordinate flag travels with it for the same reason: under the
    2026-08-23 spec the caller hands over log theta, and a worker must not have
    to guess.
    """

    theta, seed_key, observation_size, n_avg, t_record = job[:5]
    log_coords = job[5] if len(job) > 5 else False
    forward = make_forward(observation_size, n_avg, t_record)
    rng = np.random.default_rng(np.random.SeedSequence(list(seed_key)))
    physical = from_log(theta) if log_coords else np.asarray(theta, dtype=float)
    return forward(physical, rng)


def _reference_job(job):
    """Statistics of one independent reference record."""

    seed, n_data = job
    return observation_record(seed, n_data)


_POOL = None
_POOL_SIZE = None


def _get_pool(n_workers):
    """One shared pool per process: starting workers costs seconds, so reuse it.

    The start method is pinned to "spawn" rather than left to the platform.
    Linux would otherwise fork, and a forked worker inherits the parent's whole
    interpreter state -- including an OpenMP runtime that has already decided
    how many threads to use, and including any module constant a caller has
    reassigned since import.  Both are silent failure modes: the first
    oversubscribes ~190 workers by a factor of ~190, the second lets a worker
    disagree with its caller about the window length or the parameter
    coordinates.  Spawning gives every worker a clean import on both platforms.

    What this buys, measured: on a given machine the worker count does not
    change the result at all -- 1, 8, 96 and 188 workers return bitwise
    identical arrays, because each job carries its own SeedSequence.  ACROSS
    machines the random streams are still identical, but the arrays are not
    bitwise equal: different libm, BLAS and FFT builds (and numpy 2.4 against
    2.5) move the last one or two bits.  Windows and Ubuntu agree to ~1e-16
    relative here, which is far below anything physical, but a regression test
    that hashes a forward map will fail if it is moved between them.
    """

    global _POOL, _POOL_SIZE
    if _POOL is not None and _POOL_SIZE == int(n_workers):
        return _POOL
    shutdown_pool()
    # Belt and braces: modal_closure/__init__ sets these before numpy loads,
    # but a caller that imported this module directly would not have gone
    # through it.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    _POOL = ProcessPoolExecutor(max_workers=int(n_workers),
                                mp_context=multiprocessing.get_context("spawn"))
    _POOL_SIZE = int(n_workers)
    return _POOL


def shutdown_pool():
    global _POOL, _POOL_SIZE
    if _POOL is not None:
        _POOL.shutdown()
    _POOL, _POOL_SIZE = None, None


def _mapper(n_workers):
    """Return a map function; serial when n_workers <= 1.

    The chunk size adapts to the batch: at N_G = 20 and J = 100 an iteration is
    2000 jobs, and handing them out one at a time to ~190 workers costs 2000
    round trips.  Four chunks per worker keeps the tail balanced -- the jobs
    are near-identical in cost -- while cutting the traffic by the same factor.
    """

    if not n_workers or n_workers <= 1:
        return lambda fn, items: [fn(item) for item in items]
    pool = _get_pool(n_workers)
    workers = int(n_workers)

    def mapped(fn, items):
        items = list(items)
        chunksize = max(1, len(items) // (4 * workers))
        return list(pool.map(fn, items, chunksize=chunksize))

    return mapped


def _ensemble_evaluator(map_fn, observation_size, base_seed, n_avg, t_record,
                        log_coords=False, sentinel=None):
    """Evaluate a whole EKI ensemble, one keyed job per forward run.

    Seeding follows the 2026-08-23 spec's common random numbers: streams are
    keyed by (base, iteration, replicate) only, so all J members of one
    iteration share the SAME N_G random streams.  The residual model noise then
    acts as one common shift of every G_hat rather than as J independent
    errors: it cancels almost exactly in the ensemble anomalies that build
    C^{theta G} and C^{GG}, and it is redrawn every iteration so it also
    averages down across the run.  That is the mechanism that is supposed to
    make a Gamma without a discrepancy floor usable, so it is not optional
    decoration -- it is why a modest N_G is affordable at all.

    (The legacy per-member seeding convention, keyed by (base, iteration,
    member), was deleted in this release; no shipped caller used it.)
    """

    replicates = max(1, int(n_avg))

    def evaluate(theta_ensemble, iteration):
        members = theta_ensemble.shape[0]
        # One job per (member, replicate).  The seed key deliberately omits the
        # member index -- that omission IS the common random numbers.
        jobs = [(theta_ensemble[j], (base_seed, iteration, a),
                 observation_size, 1, t_record, log_coords)
                for j in range(members) for a in range(replicates)]
        raw = np.asarray(map_fn(_forward_job, jobs))
        raw = raw.reshape(members, replicates, observation_size)
        if sentinel is None:
            return raw.mean(axis=1)
        # A failed replicate must not be averaged into a finite-looking G_hat:
        # that would hide a blown-up member from the sentinel accounting.
        failed = np.all(raw == sentinel, axis=2)          # (members, replicates)
        averaged = raw.mean(axis=1)
        averaged[failed.any(axis=1)] = sentinel
        _checkpoint(iteration, theta_ensemble, averaged)
        return averaged

    return evaluate


def _checkpoint(iteration, theta_ensemble, outputs):
    """Write the current ensemble after each evaluation, if asked to.

    Off unless MODAL_CLOSURE_CHECKPOINT names a file.  A long run on a machine
    that reboots for updates otherwise loses everything; this at least leaves
    the last ensemble to restart from.  It is a salvage point, not a bit-exact
    resume: restarting re-keys the common-random streams from iteration zero.
    """

    path = os.environ.get("MODAL_CLOSURE_CHECKPOINT")
    if not path:
        return
    target = Path(path)
    # The temp name must itself end in '.npz': np.savez silently appends the
    # extension otherwise, and the rename would then look for a file that was
    # never written.
    tmp = target.with_name(target.stem + ".partial.npz")
    try:
        np.savez(tmp, iteration=np.array(iteration),
                 log_theta=np.asarray(theta_ensemble),
                 outputs=np.asarray(outputs),
                 m_modes=np.array(M_MODES),
                 ensemble_size=np.array(ENSEMBLE_SIZE),
                 t_record=np.array(T_RECORD))
        tmp.replace(target)          # atomic, so a kill mid-write cannot corrupt it
    except OSError as exc:
        print(f"[checkpoint] could not write {target}: {exc}", flush=True)


def initial_ensemble(rng):
    """Draw the initial EKI parameter ensemble, in log coordinates.

    Under the 2026-08-23 spec the ensemble is uniform on the prior interval in
    log coordinates and is returned as log theta.  That is a real change of
    prior against the pre-spec runs, not a reparametrisation of the same one:
    the median damping moves from 1.65 to 0.95 and the median noise amplitude
    from 3.25 to 1.73, both towards the smaller values the fit actually
    reaches.
    """

    delta_cols = rng.uniform(*np.log(DELTA_PRIOR), (ENSEMBLE_SIZE, M_MODES))
    sigma_cols = rng.uniform(*np.log(SIGMA_PRIOR), (ENSEMBLE_SIZE, M_MODES))
    return np.hstack([delta_cols, sigma_cols])


def fit(y, gamma_diag, q, n_avg=N_G, n_workers=N_WORKERS):
    """One EKI fit under the algorithm spec.

    Three disjoint streams, as the spec requires: the initial ensemble, the
    observation perturbations, and the forward common random numbers (which are
    keyed inside the evaluator, not drawn from a generator).  No bounds -- log
    coordinates make the update unconstrained by construction, so there is
    nothing left to clip.
    """

    map_fn = _mapper(n_workers)
    initial = initial_ensemble(np.random.default_rng(INIT_ENSEMBLE_SEED))
    return run_eki(initial, None, y, gamma_diag,
                   n_iter=N_ITER,
                   rng=np.random.default_rng(INIT_ENSEMBLE_SEED + 1),
                   bounds=None,
                   perturb_observations=True, verbose=True,
                   sentinel_value=FORWARD_SENTINEL,
                   stop_rel_tol=STOP_REL_TOL,
                   stop_patience=STOP_PATIENCE,
                   observation_rng=np.random.default_rng(OBS_PERTURB_SEED),
                   ensemble_evaluator=_ensemble_evaluator(
                       map_fn, q, FORWARD_CRN_SEED, n_avg, T_RECORD,
                       log_coords=True, sentinel=FORWARD_SENTINEL))


def select_members(result, n=ENS_MEMBERS):
    """The ensemble whose spread the stored overlays show, in PHYSICAL units.

    Under the 2026-08-23 spec this is the whole final-iteration ensemble in its
    original order -- no ranking, no truncation.  Ranking members by their own
    noisy objective and keeping the best is the selection step the spec
    removes, and it is what made the reported objective optimistic (report
    Sec. 5.1); doing it only for the overlays would put a selected spread next
    to an unselected mean.
    """

    thetas = result.final_ensemble
    objectives = result.objective_member_history[-1]
    return [as_physical(thetas[j]) for j in range(min(n, len(thetas)))
            if objectives[j] < 1e6]


def member_long_runs(members, seed0, t_record=T_LONG_ENS):
    """Long simulation for each selected member (members that blow up are
    skipped)."""

    fields = []
    for i, theta in enumerate(members):
        try:
            eta_f, v_f = simulate_grid(*unpack(theta), rng=np.random.default_rng(seed0 + i),
                                       t_record=t_record, t_burn=30.0)
            fields.append((eta_f, v_f))
        except (FloatingPointError, np.linalg.LinAlgError, ValueError):
            continue
    return fields


def run_experiment(output_dir: str | Path = DEFAULT_RESULTS_DIR,
                   forward_averages: int = N_G,
                   n_workers: int = N_WORKERS) -> Path:
    """Run the full modal closure EKI calculation and save every result array."""

    results_dir = Path(output_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 76)
    print(f"modal_closure - broadband modal closure with spectral fitting "
          f"(n={N_COMPONENTS}, M={M_MODES})")
    print("=" * 76)

    truth = truth_reference()
    print(f"truth: {N_COMPONENTS} random frequencies in [{F_MIN}, {F_MAX}] Hz, "
          f"total var = {truth['total_var']:.4f}")

    print(f"\n[estimating y, Gamma]  (1 reference record for y; "
          f"{N_GAMMA} independent reference records for Gamma)")
    y, gamma_diag, gamma_parts = build_observation_and_gamma(n_workers=n_workers)
    q = y.size
    print(f"  statistic dimension q = {q} (incl. {M_MODES} spectral bands); "
          f"observed var_eta = {y[0]:.4f}")
    report_error_model(y, gamma_diag, gamma_parts)

    print(f"\n[calibrating N_G]  ({CALIB_K} probe runs at each of 2 theta)")
    probes = calibration_probes()
    if CALIB_PROBE_THETA is None:
        print("  NOTE: no CALIB_PROBE_THETA given, so the second probe is a "
              "corner of the prior box.  That understates the requirement "
              "(~2x here); pass a previous run's theta.")
    calibration = calibrate_forward_averaging(
        gamma_parts["var_ref"], theta_probes=probes, n_workers=n_workers)
    ratio = calibration["ratio"]
    forward_averages = int(forward_averages)
    cover = coverage_report(ratio, forward_averages)
    calibration["coverage"] = cover
    print(f"  ratio = var_fwd/var_ref:  p50 {np.median(ratio):.2f}   "
          f"p{100 * CALIB_QUANTILE:.0f} "
          f"{np.quantile(ratio, CALIB_QUANTILE):.2f}   max {ratio.max():.1f}")
    print(f"  spec rule at the {100 * CALIB_QUANTILE:.0f}th percentile -> "
          f"N_G >= {calibration['recommended_n_g']}; at the maximum -> "
          f"{calibration['n_g_for_full_coverage']} (not affordable)")
    print(f"  using N_G = {forward_averages}: covers "
          f"{cover['covered']}/{cover['total']} statistics ("
          + ", ".join(f"{k} {v[0]}/{v[1]}"
                      for k, v in cover["by_family"].items()) + ")")
    if cover["uncovered"]:
        print(f"  NOT covered: {', '.join(cover['uncovered'])}", flush=True)
    if forward_averages < calibration["recommended_n_g"]:
        print(f"  WARNING: N_G = {forward_averages} is below the calibrated "
              f"{calibration['recommended_n_g']}.", flush=True)

    print("\n[EKI - Modal closure] ...")
    sde_res = fit(y, gamma_diag, q, forward_averages, n_workers)
    print(f"  stopped after {sde_res.n_updates} updates ({sde_res.stop_reason}); "
          f"final objective = {sde_res.final_objective:.4g}")

    # The reported parameters: the mean of the final ensemble, with no
    # selection of any kind -- see select_members for why no "best" member is
    # reported alongside it.  The spread quoted with it is the ensemble sd; in
    # log coordinates the ensemble is roughly symmetric, so both summaries are
    # stored, and the log one is what a +/- in the thesis table should read
    # from.
    final_physical = np.exp(sde_res.final_ensemble)
    sde_theta = final_physical.mean(axis=0)
    sde_theta_sd = final_physical.std(axis=0, ddof=1)
    log_mean, log_sd = sde_res.final_mean, sde_res.final_sd
    reported_objective = sde_res.final_objective

    sde_delta, sde_sigma = unpack(sde_theta)
    sde_spec = recovered_spectrum(sde_delta, sde_sigma)
    rel = sde_theta_sd / np.maximum(sde_theta, 1e-300)
    print(f"\n[fit] reported objective = {reported_objective:.4g}")
    print(f"  final ensemble spread: delta {np.median(rel[:M_MODES]):.1%} "
          f"median relative sd, sqrt_sigma {np.median(rel[M_MODES:]):.1%}")
    print(f"  recovered total energy (SDE) = {sde_spec.sum():.4f}   "
          f"truth total = {truth['total_var']:.4f}")

    # Unaveraged draws of G at the reported parameters.  G_hat averages N_G
    # runs and is therefore sqrt(N_G) smoother than y; any comparison that puts
    # a model curve next to a sqrt(var_ref) error bar has to use these instead,
    # or the model will look far more repeatable than the data it is fitted to.
    map_fn = _mapper(n_workers)
    single_outputs = np.asarray(map_fn(
        _forward_job,
        [(sde_theta, (SINGLE_REAL_SEED, i), q, 1, T_RECORD, False)
         for i in range(N_SINGLE_REALISATIONS)]))
    single_outputs = single_outputs[
        ~np.all(single_outputs == FORWARD_SENTINEL, axis=1)]
    print(f"  stored {len(single_outputs)} unaveraged G(theta_hat) draws "
          f"for the error-bar comparisons")

    print("\n[long validation + ensemble runs] ...")
    truth_eta_long, truth_v_long = long_truth_field(TRUTH_LONG_SEED)
    sde_eta_long, sde_v_long = simulate_grid(sde_delta, sde_sigma,
                                             rng=np.random.default_rng(SDE_LONG_SEED),
                                             t_record=T_LONG, t_burn=40.0)
    sde_members = select_members(sde_res)
    sde_fields = member_long_runs(sde_members, SDE_ENSEMBLE_SEED0)
    print(f"  SDE: {len(sde_fields)}/{len(sde_members)} members simulated")

    # A run whose final ensemble has no usable member has failed, and it must
    # say so here.  Left to continue, it writes a bundle with an empty ensemble
    # array and the failure surfaces later as an IndexError in an analysis
    # helper -- which reads like a bug rather than a diverged inversion.
    if not sde_fields:
        raise RuntimeError(
            f"the final ensemble has no usable member "
            f"({len(sde_members)} survived the sentinel filter, "
            f"{len(sde_fields)} simulated).  The inversion diverged: final "
            f"objective {sde_res.final_objective:.4g}, "
            f"{sde_res.total_sentinels} sentinel evaluations.  This is a "
            f"result, not a crash -- record the configuration that produced "
            f"it rather than retrying blindly.")

    print(f"[saving bundle] -> {results_dir}")
    span_v = 4.0 * float(truth_v_long.std())
    bundle_io.save_bundle(
        str(results_dir),
        experiment="modal_closure", gauges=GAUGES, dt_data=DT_DATA, depth=DEPTH,
        span_eta=np.array([-4.0, 4.0]), span_v=np.array([-span_v, span_v]),
        truth_eta_long=truth_eta_long, truth_v_long=truth_v_long,
        sde_eta_long=sde_eta_long, sde_v_long=sde_v_long,
        sde_eta_ens=np.array([f[0] for f in sde_fields]),
        sde_v_ens=np.array([f[1] for f in sde_fields]),
        sde_objective=float(reported_objective),
        grid_freq=F_GRID, truth_freq=truth["freq"], truth_energy=truth["energy"],
        spec_peak_hz=SPEC_PEAK_HZ, spec_width_hz=SPEC_WIDTH_HZ, target_var=TARGET_VAR,
        f_min=F_MIN, f_max=F_MAX,
        truth_binned=bin_truth_to_grid(truth),
        sde_member_spectrum=np.array([recovered_spectrum(*unpack(th)) for th in sde_members]),
        sde_best_spectrum=sde_spec,
        sde_members=np.array(sde_members),
        # NAMING TRAP (kept for bundle compatibility with the frozen run):
        # `sde_best` holds the FINAL-ENSEMBLE MEAN in physical units, not a
        # selected best member.  The key predates the 2026-08-23 reporting
        # convention; renaming it would break validation of the archived
        # bundle, so the explicit aliases below carry the honest names.
        sde_best=sde_theta,
        # The 2026-08-23 reporting convention, stored explicitly so no reader
        # has to infer whether `sde_best` was selected or averaged.
        sde_theta_sd=sde_theta_sd,
        sde_log_mean=log_mean, sde_log_sd=log_sd,
        sde_final_ensemble=sde_res.final_ensemble,
        sde_final_outputs=sde_res.final_outputs,
        sde_single_realisations=single_outputs,
        target_statistics=y, gamma_diag=gamma_diag,
        gamma_var_ref=gamma_parts["var_ref"],
        sde_objective_history=sde_res.objective_history,
        sde_objective_member_history=sde_res.objective_member_history,
        sde_theta_history=sde_res.theta_history,
    )
    summary = {
        "experiment": "broadband_modal_spectral_fit",
        "algorithm_spec": "EKI_algorithm_spec_2026-08-23.md",
        "n_components": N_COMPONENTS, "freq_band_Hz": [F_MIN, F_MAX],
        "m_grid_modes": M_MODES, "t_record_s": T_RECORD,
        "spectral_statistic": True, "grid_freq_Hz": F_GRID.tolist(),
        "truth_total_var": truth["total_var"],
        "var_eta_observed": float(y[0]),
        "statistic_dim": int(q),
        "sde_objective": float(reported_objective),
        "sde_recovered_spectrum": sde_spec.tolist(),
        "truth_binned_spectrum": bin_truth_to_grid(truth).tolist(),
        "n_members_overlaid": {"sde": len(sde_fields)},
        # --- conventions ---
        "y_convention": "single_window",
        "forward_convention": "average_of_N_G_common_random",
        "gamma_type": "diagonal",
        "gamma_ref_method": "independent_repeats",
        "coordinates": "log",
        "parameter_report_convention": "final_ensemble_mean_pm_sd",
        "parameter_bounds": None,          # log coordinates: unconstrained
        "prior_interval": {"delta_per_s": list(DELTA_PRIOR),
                           "sqrt_sigma": list(SIGMA_PRIOR),
                           "shape": "log_uniform"},
        "n_gamma": int(N_GAMMA),
        "n_g": int(forward_averages),
        "t_g_s": float(T_RECORD),
        "n_workers": int(n_workers),
        "gamma_relative_precision": gamma_parts["relative_precision"],
        "sde_theta_mean": sde_theta.tolist(),
        "sde_theta_sd": sde_theta_sd.tolist(),
        "sde_log_mean": np.asarray(log_mean).tolist(),
        "sde_log_sd": np.asarray(log_sd).tolist(),
        "gamma_diagnostics": {
            "min": float(gamma_diag.min()),
            "median": float(np.median(gamma_diag)),
            "max": float(gamma_diag.max()),
            "n_gamma": int(N_GAMMA),
            "relative_precision": gamma_parts["relative_precision"],
            "contains_forward_term": False,
            "contains_discrepancy_floor": False,
            "neff_correction": False,
        },
        "eki": {
            "ensemble_size": ENSEMBLE_SIZE,
            "iterations_max": N_ITER,
            "updates_applied": int(sde_res.n_updates),
            "stop_reason": sde_res.stop_reason,
            "stop_rel_tol": STOP_REL_TOL,
            "stop_patience": STOP_PATIENCE,
            "objective_history": [float(v) for v in sde_res.objective_history],
            "sentinel_evaluations": int(sde_res.total_sentinels),
        },
        "n_g_calibration": {
            "k": calibration["k"],
            "quantile": calibration["quantile"],
            "safety_factor": CALIB_SAFETY,
            "tolerance": CALIB_TOLERANCE,
            "probes": list(calibration["per_probe"].keys()),
            "ratio_percentiles": {
                f"p{p}": float(np.percentile(calibration["ratio"], p))
                for p in (50, 75, 90, 100)},
            "recommended_n_g": calibration["recommended_n_g"],
            "n_g_for_full_coverage": calibration["n_g_for_full_coverage"],
            "coverage": calibration["coverage"],
        },
        "seeds": {
            "truth_frequency": TRUTH_FREQ_SEED,
            "observation_y": OBSERVATION_SEED,
            "gamma_reference": GAMMA_REF_SEED,
            "forward_common_random": FORWARD_CRN_SEED,
            "observation_perturbation": OBS_PERTURB_SEED,
            "initial_ensemble": INIT_ENSEMBLE_SEED,
            "single_realisations": SINGLE_REAL_SEED,
            "calibration": CALIB_SEED,
            "truth_long": TRUTH_LONG_SEED,
            "sde_long": SDE_LONG_SEED,
            "sde_ensemble_start": SDE_ENSEMBLE_SEED0,
            "blocks": check_seed_blocks(),
        },
    }

    with (results_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Results + bundle.npz in {results_dir}")
    return results_dir


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    main()
