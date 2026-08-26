"""EKI engine with ensemble member resampling — the exact loop of the frozen
thesis-Chapter-4 GP run (case-local on purpose, see below).

Why this file exists instead of algorithms.eki
----------------------------------------------
The shared engine (code_rp/algorithms/eki.py) grew out of the linear-wave
engine this file also descends from, and its injection hooks cover most case
behaviour — but not this run's, for two concrete reasons:

1. **Member resampling is not expressible through the shared hooks.**  With
   ``resample_failures=True`` a member whose forward map returned the failure
   sentinel is replaced by a copy of a SURVIVING member — parameters and
   output together, drawn from a dedicated generator (seed 987654321) — after
   the evaluation and BEFORE the objective, the empirical covariances and the
   Kalman update, and the recorded pre-update ensemble ``theta_history[-1]``
   is rewritten to the repaired one.  The shared engine's
   ``sentinel_row_fn`` replaces failed OUTPUT rows with one fixed penalty row
   (it never touches theta), and ``post_update`` runs after the Kalman
   update; neither can swap theta and outputs together at the right point in
   the loop.

2. **The frozen run's objective solve is the regularised one.**  This loop
   computes Phi by solving against Gamma with its diagonal inflated by a
   relative 1e-8 (``_regularise``), which is what produced the archived
   ``phi_history`` (1.41e10 -> 80.5698).  The shared engine deliberately
   computes Phi with the EXACT Gamma (Cholesky whitening; release decision
   2026-08-25).  The difference is <= 1e-8 relative — far below any quoted
   precision — but the frozen numbers are frozen, so the construction that
   produced them is kept verbatim rather than approximated.

Everything else (perturbed-observations update, J-1 covariances, the
2026-08-23 stopping rule, the final extra evaluation with rng-state restore)
is the same mathematics as algorithms.eki.run_eki.  This file is NOT a fork
of the shared engine: it is the archived engine of one frozen run, shipped
next to the run's driver so that run stays reproducible from the file that
produced it.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

# numpy arrays
Array = np.ndarray
Bounds = Sequence[tuple[float | None, float | None]]

# ForwardMap takes a parameter vector and a random number generator, and
# returns the vector of statistics computed from one finite simulation: G(theta).
ForwardMap = Callable[[Array, np.random.Generator], Array]

# theta    = one model parameter vector
# G(theta) = statistics predicted by a simulation with that theta
# y        = observed statistics from the true system
# Gamma    = covariance / noise level of those statistics
#
# n_iter = number of EKI iterations to run
# J      = ensemble size
# p      = number of unknown parameters in theta
# q      = y.size = number of statistics in the data vector


@dataclass
class EKIResult:
    theta_history: Array  # theta - (n_iter+1, J, p)
    output_history: Array  # G(theta) - (n_iter+1, J, q)
    # Paper objective of the ensemble-mean output at each EKI evaluation.
    objective_history: Array
    # Paper objective of every single ensemble member at each EKI evaluation.
    objective_member_history: Array
    best_iteration: int  # iteration with the smallest objective value
    best_member_iteration: int  # iteration containing the best single particle
    best_member_index: int  # ensemble index of the best single particle
    best_member_parameter: Array  # theta of the best single particle
    best_member_output: Array  # G(theta) of the best single particle
    best_member_objective: float  # smallest single-particle objective
    # Number of members whose forward map returned the failure sentinel at each
    # evaluation.  A nonzero entry means the empirical C^GG that produced the
    # Kalman gain was contaminated by a penalty vector rather than a statistic.
    sentinel_counts: Array = field(default_factory=lambda: np.zeros(0, dtype=int))
    # Members replaced by a copy of a survivor at each evaluation, when
    # ``resample_failures`` is on.  Nonzero entries mean the ensemble was
    # repaired rather than left with penalty rows in C^GG.
    resampled_counts: Array = field(default_factory=lambda: np.zeros(0, dtype=int))
    # Kalman updates actually applied, and why the loop ended.  Under the
    # 2026-08-23 algorithm spec the loop may stop before ``n_iter`` because the
    # relative change of the ensemble-mean objective fell below a threshold.
    n_updates: int = 0
    stop_reason: str = "n_iter"

    @property
    def total_sentinels(self) -> int:
        return int(np.sum(self.sentinel_counts))

    # Parameter ensemble with the smallest recorded objective value.
    @property
    def best_ensemble(self) -> Array:
        return self.theta_history[self.best_iteration]

    # Mean parameter vector with the smallest recorded objective value.
    @property
    def best_mean(self) -> Array:
        return np.mean(self.best_ensemble, axis=0)

    # Best objective value recorded during the EKI iterations.
    @property
    def best_objective(self) -> float:
        return float(self.objective_history[self.best_iteration])

    # The literal last iteration (paper display convention: run the fixed
    # number of EKI iterations and use the final ensemble, no selection).
    # Note: "final" and "best" can differ -- the last step is not guaranteed
    # to have the smallest objective.
    @property
    def final_ensemble(self) -> Array:
        return self.theta_history[-1]

    @property
    def final_mean(self) -> Array:
        return np.mean(self.final_ensemble, axis=0)

    # Spread of the final ensemble.  The 2026-08-23 spec reports the learned
    # parameters as final_mean +/- final_sd, with no selection of any kind, so
    # this is the uncertainty that actually goes into the thesis table.
    @property
    def final_sd(self) -> Array:
        return np.std(self.final_ensemble, axis=0, ddof=1)

    @property
    def final_objective(self) -> float:
        return float(self.objective_history[-1])

    @property
    def final_outputs(self) -> Array:
        return self.output_history[-1]

    @property
    def best_outputs(self) -> Array:
        return self.output_history[self.best_iteration]


# Component-wise box constraints on the parameters (clip after each update).
def _clip_bounds(theta: Array, bounds: Bounds | None) -> Array:
    if bounds is None:
        return theta
    clipped = theta.copy()
    for idx, (lo, hi) in enumerate(bounds):
        if lo is not None:
            clipped[:, idx] = np.maximum(clipped[:, idx], lo)
        if hi is not None:
            clipped[:, idx] = np.minimum(clipped[:, idx], hi)
    return clipped


def _regularise(matrix: Array, jitter: float) -> Array:
    """Add ``jitter`` as a RELATIVE inflation of the diagonal, not an absolute one.

    An absolute ``matrix + jitter * I`` assumes every diagonal entry is O(1).
    Here they are not: under the 2026-08-23 spec ``Gamma = diag(var_ref)`` spans
    1.7e-12 on the 1.8 Hz band to 0.98 on the elevation variance, so the old
    absolute jitter of 1e-8 was 5875x the entire error bar of the smallest
    statistic and 85x, 26x and 3.1x of the next three.  That is a discrepancy
    floor of 1e-8 reintroduced by the back door, on exactly the low-energy bands
    the spec is most delicate about, and it moved the objective by 37%.

    Scaling the diagonal by ``1 + jitter`` instead is scale-free: it conditions
    the solve without asserting an error bar the error model does not have.
    """

    if jitter <= 0.0:
        return matrix
    out = np.array(matrix, dtype=float, copy=True)
    index = np.diag_indices_from(out)
    out[index] = out[index] * (1.0 + jitter)
    return out


def _objective_values(
    outputs: Array, observation: Array, gamma_mat: Array, jitter: float
) -> Array:
    """Compute the paper objective for every ensemble member."""

    residual = outputs - observation[None, :]
    weighted_residual = np.linalg.solve(
        _regularise(gamma_mat, jitter), residual.T
    ).T
    return 0.5 * np.sum(residual * weighted_residual, axis=1)


def _validate_gamma(gamma_mat: Array) -> None:
    """Reject an error-covariance that would make the Kalman gain meaningless.

    The wave branch originally trusted its caller here.  The check draws no
    random numbers and therefore cannot change a reproduced run; it only turns a
    silently wrong Gamma into an immediate, named failure.
    """

    if not np.all(np.isfinite(gamma_mat)):
        raise ValueError("Gamma must contain only finite entries")
    if not np.allclose(gamma_mat, gamma_mat.T, rtol=0.0, atol=1e-12):
        raise ValueError("Gamma must be symmetric")
    try:
        np.linalg.cholesky(gamma_mat)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Gamma must be positive definite") from exc


def _evaluate(
    theta: Array,
    iteration: int,
    forward_map: ForwardMap | None,
    ensemble_evaluator: Callable[[Array, int], Array] | None,
    rng: np.random.Generator,
    ensemble_size: int,
) -> Array:
    """Evaluate G for a whole ensemble.

    ``ensemble_evaluator`` lets the caller key each member's random stream
    explicitly, which is what allows the evaluations to be distributed over
    worker processes without changing the result.  The legacy path threads one
    shared generator through the members in order and is retained for callers
    that still pass a bare forward map.
    """

    if ensemble_evaluator is not None:
        outputs = np.asarray(ensemble_evaluator(theta, iteration))
        if outputs.shape[0] != ensemble_size:
            raise ValueError("ensemble_evaluator returned the wrong number of rows")
        return outputs
    if forward_map is None:
        raise ValueError("either forward_map or ensemble_evaluator must be given")
    return np.asarray([forward_map(theta[j], rng) for j in range(ensemble_size)])


def _count_sentinels(outputs: Array, sentinel_value: float | None) -> int:
    """Members whose forward map returned the constant failure vector."""

    if sentinel_value is None:
        return 0
    return int(np.sum(np.all(outputs == sentinel_value, axis=1)))


def run_eki(
    initial_ensemble: Array,
    forward_map: ForwardMap | None,
    observation: Array,
    gamma: Array,
    *,
    n_iter: int,
    rng: np.random.Generator,
    perturb_observations: bool = False,
    bounds: Bounds | None = None,
    jitter: float = 1e-8,
    verbose: bool = True,
    sentinel_value: float | None = None,
    ensemble_evaluator: Callable[[Array, int], Array] | None = None,
    stop_rel_tol: float | None = None,
    stop_patience: int = 3,
    observation_rng: np.random.Generator | None = None,
    resample_failures: bool = False,
    resample_seed: int = 987654321,
) -> EKIResult:
    """Ensemble Kalman inversion with perturbed observations.

    ``stop_rel_tol``/``stop_patience`` implement the stopping rule of the
    2026-08-23 algorithm spec: leave the loop once the relative change of the
    ensemble-mean objective has stayed below the tolerance for ``stop_patience``
    consecutive iterations.  ``None`` (the default) runs the fixed ``n_iter``.

    ``observation_rng``, when given, draws the perturbations ``eta ~ N(0,Gamma)``
    from a stream of its own.  The spec asks for the observation noise, the
    initial ensemble and the forward-model randomness to occupy disjoint seed
    blocks; sharing one generator would interleave them.

    ``resample_failures`` replaces a member whose forward map returned the
    sentinel with a copy of a surviving member, instead of leaving the penalty
    vector in the ensemble to contaminate C^GG.  Off by default so the archived
    runs reproduce; the Lorenz codebase uses resampling and this branch's own
    warning has always said the penalty scheme should be replaced by it.
    """

    # theta has shape (J, p):
    # J = number of ensemble members, p = number of unknown parameters.
    theta = np.asarray(initial_ensemble, dtype=float).copy()

    # y has shape (q,), q is the number of statistics in the data vector.
    y = np.asarray(observation, dtype=float).reshape(-1)
    ensemble_size, _ = theta.shape
    q = y.size

    gamma_mat = np.asarray(gamma, dtype=float)
    if gamma_mat.ndim == 1:
        gamma_mat = np.diag(gamma_mat)
    _validate_gamma(gamma_mat)

    sentinel_counts: list[int] = []
    theta_history = [theta.copy()]
    output_history = []
    objective_history = []
    objective_member_history = []
    best_iteration = 0
    best_objective = np.inf
    best_member_iteration = 0
    best_member_index = 0
    best_member_objective = np.inf
    best_member_parameter = theta[0].copy()
    best_member_output = np.empty(q, dtype=float)
    obs_rng = rng if observation_rng is None else observation_rng
    n_updates = 0
    stop_reason = "n_iter"
    quiet_streak = 0
    resampled_counts: list[int] = []
    resample_rng = np.random.default_rng(resample_seed)

    for i in range(n_iter):
        # Step 1: evaluate the forward map G(theta) for every ensemble member.
        # outputs[j] = G(theta[j])
        outputs = _evaluate(theta, i, forward_map, ensemble_evaluator, rng, ensemble_size)
        n_bad = _count_sentinels(outputs, sentinel_value)
        if resample_failures and n_bad and n_bad < ensemble_size:
            # Replace a failed member with a copy of a surviving one, parameters
            # and output together, rather than leaving the penalty vector in the
            # ensemble.  A sentinel row is a constant far outside the data range,
            # so it dominates C^GG and the Kalman gain it produces describes the
            # penalty, not the model.  The GP closure makes this concrete: its
            # stationarity gate fails whole members, and 46 sentinels across 31
            # ensembles were enough for run_eki to warn that C^GG was
            # contaminated.  Resampling keeps the ensemble size fixed and the
            # covariance built only from usable evaluations.
            bad = np.all(outputs == sentinel_value, axis=1)
            good = np.flatnonzero(~bad)
            donors = resample_rng.choice(good, size=int(bad.sum()))
            theta[bad] = theta[donors]
            outputs[bad] = outputs[donors]
            resampled_counts.append(int(bad.sum()))
            theta_history[-1] = theta.copy()
        else:
            resampled_counts.append(0)
        output_history.append(outputs.copy())
        sentinel_counts.append(n_bad)

        # Objective function for every single member:
        # Phi(theta) = 0.5 * (G(theta) - y)^T Gamma^{-1} (G(theta) - y).
        objective_each = _objective_values(outputs, y, gamma_mat, jitter)
        objective_member_history.append(objective_each.copy())

        # The paper-style figures compare the ensemble-mean output with y.
        # Therefore the iteration "best" is selected by the objective of
        # mean_j G(theta_j), not by the average of the individual objectives.
        output_mean = np.mean(outputs, axis=0, keepdims=True)
        objective = float(_objective_values(output_mean, y, gamma_mat, jitter)[0])
        objective_history.append(objective)

        current_iteration = len(objective_history) - 1
        if objective < best_objective:
            best_objective = objective
            best_iteration = current_iteration

        member_index = int(np.argmin(objective_each))
        member_objective = float(objective_each[member_index])
        if member_objective < best_member_objective:
            best_member_objective = member_objective
            best_member_iteration = current_iteration
            best_member_index = member_index
            best_member_parameter = theta[member_index].copy()
            best_member_output = outputs[member_index].copy()

        # Stopping rule (2026-08-23 spec): the relative change of the
        # ensemble-mean objective, judged over consecutive iterations.  Checked
        # here, before the update, so that the ensemble reported at the end is
        # the one whose objective satisfied the rule.
        rel_change = np.inf
        if stop_rel_tol is not None and len(objective_history) >= 2:
            previous = objective_history[-2]
            if np.isfinite(previous) and previous > 0.0:
                rel_change = abs(objective - previous) / previous
                quiet_streak = quiet_streak + 1 if rel_change < stop_rel_tol else 0

        if verbose:
            tail = ("" if not np.isfinite(rel_change)
                    else f", rel={rel_change:.2%} ({quiet_streak}/{stop_patience})")
            print(
                f"EKI iter {i + 1:02d}/{n_iter}: "
                f"objective={objective:.6g}, best={best_objective:.6g}, "
                f"best_member={best_member_objective:.6g}{tail}",
                flush=True,
            )

        if stop_rel_tol is not None and quiet_streak >= stop_patience:
            stop_reason = f"rel_change<{stop_rel_tol:g} for {stop_patience} iterations"
            if verbose:
                print(f"EKI stop: {stop_reason} (after {i + 1} evaluations)",
                      flush=True)
            break

        # Step 2: empirical covariances from the current ensemble,
        # C^{theta G} (p, q) and C^{GG} (q, q), unbiased J-1 normalisation.
        theta_anom = theta - np.mean(theta, axis=0, keepdims=True)
        output_anom = outputs - np.mean(outputs, axis=0, keepdims=True)
        c_tg = theta_anom.T @ output_anom / (ensemble_size - 1)
        c_gg = output_anom.T @ output_anom / (ensemble_size - 1)

        # Step 3: Kalman gain matrix K = C^{theta G} @ inv(C^{GG} + Gamma).
        system = _regularise(c_gg + gamma_mat, jitter)
        gain_t = np.linalg.solve(system, c_tg.T)
        gain = gain_t.T

        # Step 4: y_n^j = y + eta^j, where eta^j ~ N(0, Gamma) is a random
        # perturbation of the data.
        if perturb_observations:
            # Sample ensemble_size draws from N(y, Gamma).
            y_ensemble = obs_rng.multivariate_normal(y, gamma_mat, size=ensemble_size)
        else:
            # No perturbation: use the same observation y for every ensemble member.
            y_ensemble = np.broadcast_to(y, outputs.shape)

        # Step 5: update the parameters: theta^j = theta^j + K @ (y_n^j - G(theta^j)).
        innovations = y_ensemble - outputs
        theta = theta + innovations @ gain.T
        theta = _clip_bounds(theta, bounds)
        theta_history.append(theta.copy())
        n_updates += 1

    # If the last Kalman update produced a new ensemble that has not yet been
    # evaluated, evaluate it once so it can also compete for the best objective.
    if len(output_history) < len(theta_history):
        rng_state = rng.bit_generator.state
        outputs = _evaluate(theta, n_iter, forward_map, ensemble_evaluator, rng, ensemble_size)
        rng.bit_generator.state = rng_state
        output_history.append(outputs.copy())
        sentinel_counts.append(_count_sentinels(outputs, sentinel_value))
        resampled_counts.append(0)
        objective_each = _objective_values(outputs, y, gamma_mat, jitter)
        objective_member_history.append(objective_each.copy())
        output_mean = np.mean(outputs, axis=0, keepdims=True)
        objective = float(_objective_values(output_mean, y, gamma_mat, jitter)[0])
        objective_history.append(objective)

        current_iteration = len(objective_history) - 1
        if objective < best_objective:
            best_objective = objective
            best_iteration = current_iteration

        member_index = int(np.argmin(objective_each))
        member_objective = float(objective_each[member_index])
        if member_objective < best_member_objective:
            best_member_objective = member_objective
            best_member_iteration = current_iteration
            best_member_index = member_index
            best_member_parameter = theta[member_index].copy()
            best_member_output = outputs[member_index].copy()

        if verbose:
            print(
                f"EKI final evaluation: objective={objective:.6g}, "
                f"best={best_objective:.6g}, "
                f"best_member={best_member_objective:.6g}",
                flush=True,
            )

    theta_history_array = np.asarray(theta_history)
    output_history_array = np.asarray(output_history)
    sentinel_array = np.asarray(sentinel_counts, dtype=int)

    # Without resampling this keeps the original penalty-sentinel scheme so the
    # archived runs stay reproducible.  The Lorenz codebase replaced it with
    # resampling from the valid members precisely because a sentinel row biases
    # C^GG.  Make that hazard loud instead of silent.
    if verbose and sentinel_array.sum() > 0 and not resample_failures:
        print(
            f"WARNING: {int(sentinel_array.sum())} sentinel member evaluations "
            f"across {sentinel_array.size} ensembles; the empirical C^GG is "
            "contaminated and this fit should be repeated with member resampling.",
            flush=True,
        )

    return EKIResult(
        theta_history=theta_history_array,
        output_history=output_history_array,
        objective_history=np.asarray(objective_history),
        objective_member_history=np.asarray(objective_member_history),
        best_iteration=best_iteration,
        best_member_iteration=best_member_iteration,
        best_member_index=best_member_index,
        best_member_parameter=np.asarray(best_member_parameter),
        best_member_output=np.asarray(best_member_output),
        best_member_objective=float(best_member_objective),
        sentinel_counts=sentinel_array,
        resampled_counts=np.asarray(resampled_counts, dtype=int),
        n_updates=n_updates,
        stop_reason=stop_reason,
    )
