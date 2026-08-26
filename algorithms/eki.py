"""Ensemble Kalman inversion (EKI) engine shared by every dissertation case.

Origin: 2.Linear_wave_case/modal_closure/eki.py
Changes vs origin:
- added optional injection hooks (sentinel_row_fn, failed_mask_fn, post_update,
  iteration_callback) and a jitter_mode switch, all defaulting to the exact
  original behaviour (see "Extensions vs the linear-wave engine" below);
- added the CRN seed helpers crn_seed / crn_seeds, ported from
  1. Reproduce_papers/common/code/eki_spec.py (identical arithmetic);
- added clip_latent, ported from eki_spec.py, as a ready-made post_update
  building block for log-parameterised cases;
- the objective solve is DE-regularised (release decision 2026-08-25): Phi is
  computed by Cholesky-whitening with the EXACT Gamma, ported from
  eki_spec._phi, so the reported discrepancy never contains a numerical
  conditioning term; ``jitter`` now applies to the Kalman-gain solve only.
  (The origin regularised both solves; the effect on Phi was <= 1e-8 relative.)
- comments/docstrings translated and polished; core update numerics untouched.

Extensions vs the linear-wave engine
------------------------------------
Each addition is keyword-only and defaults to the original behaviour:

- ``sentinel_row_fn(observation, gamma_diag) -> (q,) row``: when given, every
  failed member's output row is REPLACED by this row before it enters the
  history, the objective and the Kalman update.  The vKdV convention is
  ``lambda y, g: y + 10.0 * np.sqrt(g)`` (a large fixed penalty in error-bar
  units).  Default ``None`` keeps the linear-wave behaviour: failed rows are
  only counted, never replaced.
- ``failed_mask_fn(outputs) -> (J,) bool mask``: the case decides what
  "failed" means (NaNs, blow-up flags, out-of-range statistics...).  Default:
  a row is failed when it is identically equal to ``sentinel_value`` (the
  original constant-sentinel detection); with ``sentinel_value=None`` nothing
  is ever flagged.
- ``post_update(thetas, iteration) -> thetas``: applied after each Kalman
  update (and after ``bounds`` clipping).  This is the clip hook: the Lorenz
  cases pass ``clip_latent``-based guards, the vKdV spec runs pass their own
  latent clip.  Default ``None`` applies nothing.
- ``iteration_callback(state: dict) -> None``: called at the end of every
  iteration in which a Kalman update was applied, with
  ``{iteration, thetas, outputs, objectives, mean_objective, rng}`` where
  ``thetas`` is the POST-update ensemble.  Intended for per-iteration
  checkpointing of long runs (write thetas + the rng state, support
  ``--resume``).  When the stopping rule fires, the loop exits before the
  update, so no callback fires for that terminal evaluation.  Default
  ``None``.
- ``jitter_mode``: ``'relative'`` (default, the wave convention: diagonal
  scaled by ``1 + jitter``, scale-free) or ``'absolute'`` (the legacy
  Lorenz/vKdV convention: ``matrix + jitter * I``).  Applied to the
  Kalman-gain solve ONLY; the objective is always computed with the exact
  Gamma.

Conventions
-----------
- Empirical covariances use the unbiased ``J - 1`` normalisation.
- The objective is the exact ``Phi = 0.5 ||Gamma^{-1/2}(y - G)||^2``, computed
  by whitening with the Cholesky factor of the un-regularised Gamma (the
  eki_spec._phi construction).  Regularisation (``jitter``/``jitter_mode``)
  conditions the Kalman-gain solve only.
- Perturbed observations: with ``perturb_observations=True`` each member sees
  ``y + eta_j``, ``eta_j ~ N(0, Gamma)`` (drawn from ``observation_rng`` when
  given, so observation noise can occupy its own seed block).
- Log-space parameters are the CALLER's responsibility: this engine evolves
  whatever coordinates it is handed; encode/decode with
  ``algorithms.parameterization`` in the case driver.
- The reported estimate is the CALLER's convention.  ``EKIResult`` records
  both the final ensemble (spec-2026-08-23 reporting: ``final_mean`` +/-
  ``final_sd``, no selection) and the best-objective iteration/member (legacy
  paper-figure selection); the case decides which to publish.
- Stopping rule (2026-08-23 spec): checked BEFORE the update, on the relative
  change of the ensemble-mean-output objective, so the ensemble reported at
  the end is the one whose objective satisfied the rule.
- After the last Kalman update the new ensemble is evaluated once more (with
  the rng state restored afterwards) so it can also compete for "best".
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


# ---------------------------------------------------------------- CRN seeds


def crn_seed(base_seed: int, iteration: int, realisation: int) -> int:
    """Common-random-numbers seed S(n, r) for one forward realisation.

    Common random numbers (CRN): within one EKI iteration ``n`` every ensemble
    member evaluates its ``N_G`` forward realisations with the SAME seed set
    ``S(n, 1..N_G)``, so differences between members reflect their parameters
    and not their noise draws; the set changes from iteration to iteration so
    the inversion never conditions on one particular noise realisation.

    Mirrors ``eki_spec.crn_seeds`` (spec-2026-08-23, section 2 step 1):
    ``S(n, r) = base_seed + 1_000_003 * (n + 1) + r`` with realisation index
    ``r`` in ``0 .. N_G - 1``.
    """

    return int(base_seed + 1_000_003 * (int(iteration) + 1) + int(realisation))


def crn_seeds(base_seed: int, iteration: int, n_g: int) -> np.ndarray:
    """Vector form of :func:`crn_seed`: the full seed set S(n, 1..N_G).

    Identical for all members within one iteration, different at each
    iteration.  Byte-for-byte the same arithmetic as ``eki_spec.crn_seeds``.
    """

    return base_seed + 1_000_003 * (iteration + 1) + np.arange(int(n_g), dtype=np.int64)


# ---------------------------------------------------------------- latent guard


LATENT_LOW, LATENT_HIGH = float(np.log(1e-12)), 80.0


def clip_latent(theta_latent: Array, positive_indices) -> tuple[Array, int]:
    """Declared numerical guard (not part of the spec's mathematics).

    ``parameterization.log_decode`` refuses latent values outside
    ``[log(1e-12), 80]`` because ``exp`` would overflow.  An unattended EKI run
    can propose such a value early on, which would abort the whole inversion.
    We clip the *evaluated* parameter into the safe range -- the ensemble
    member itself keeps the value EKI produced -- and count the occurrences so
    the run can report them.  A healthy run clips nothing.

    Typical use: inside the case's forward evaluator (clip what is decoded,
    not the ensemble), or wrapped as a ``post_update`` hook when the case
    prefers to clip the ensemble itself.
    """

    theta = np.array(theta_latent, dtype=float, copy=True)
    idx = list(positive_indices)
    if not idx:
        return theta, 0
    block = theta[..., idx]
    clipped = int(np.sum((block < LATENT_LOW) | (block > LATENT_HIGH)))
    if clipped:
        theta[..., idx] = np.clip(block, LATENT_LOW, LATENT_HIGH)
    return theta, clipped


# ---------------------------------------------------------------- result


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
    # Number of members whose forward map failed at each evaluation.  Without
    # sentinel_row_fn a nonzero entry means the empirical C^GG that produced
    # the Kalman gain was contaminated by a constant penalty vector rather
    # than a statistic; with sentinel_row_fn the failed rows carry the case's
    # declared penalty row instead.
    sentinel_counts: Array = field(default_factory=lambda: np.zeros(0, dtype=int))
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


# ---------------------------------------------------------------- internals


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


def _regularise(matrix: Array, jitter: float, mode: str = "relative") -> Array:
    """Condition the Kalman-gain solve by inflating the diagonal.

    Used for ``C^GG + Gamma`` only -- the objective is never regularised (see
    :func:`_objective_values`).

    ``mode='relative'`` (the wave convention) scales the diagonal by
    ``1 + jitter``.  An absolute ``matrix + jitter * I`` assumes every diagonal
    entry is O(1).  Under the 2026-08-23 spec ``Gamma = diag(var_ref)`` spans
    1.7e-12 on the 1.8 Hz band to 0.98 on the elevation variance, so an
    absolute jitter of 1e-8 would be 5875x the entire error bar of the
    smallest statistic -- a discrepancy floor reintroduced by the back door on
    exactly the low-energy bands the spec is most delicate about.  (When the
    origin engine still regularised the objective solve too, that absolute
    jitter moved the reported objective by 37% on the wave case.)  Scaling the
    diagonal instead is scale-free: it conditions the solve without asserting
    an error bar the error model does not have.

    ``mode='absolute'`` (the legacy Lorenz/vKdV convention) adds
    ``jitter * I``, matching ``eki_spec.run_eki_spec``'s gain solve.
    """

    if jitter <= 0.0:
        return matrix
    out = np.array(matrix, dtype=float, copy=True)
    index = np.diag_indices_from(out)
    if mode == "relative":
        out[index] = out[index] * (1.0 + jitter)
    elif mode == "absolute":
        out[index] = out[index] + jitter
    else:
        raise ValueError(f"unknown jitter_mode: {mode!r}")
    return out


def _objective_values(
    outputs: Array, observation: Array, gamma_chol: Array
) -> Array:
    """Paper objective Phi = 0.5 ||Gamma^{-1/2}(G - y)||^2, one value per row.

    Whitening with the lower Cholesky factor of the EXACT Gamma (no explicit
    inverse, no regularisation), byte-for-byte the ``eki_spec._phi``
    construction: Phi is the reported physical discrepancy measure, so no
    numerical conditioning term may enter it.  The ``jitter`` parameter
    conditions the Kalman-gain solve only.
    """

    from scipy.linalg import solve_triangular

    residual = outputs - observation[None, :]
    z = solve_triangular(gamma_chol, residual.T, lower=True)
    return 0.5 * np.sum(z * z, axis=0)


def _validate_gamma(gamma_mat: Array) -> None:
    """Reject an error-covariance that would make the Kalman gain meaningless.

    The check draws no random numbers and therefore cannot change a reproduced
    run; it only turns a silently wrong Gamma into an immediate, named failure.
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
    explicitly (e.g. via :func:`crn_seed`), which is what allows the
    evaluations to be distributed over worker processes without changing the
    result.  The legacy path threads one shared generator through the members
    in order and is retained for callers that still pass a bare forward map.
    """

    if ensemble_evaluator is not None:
        outputs = np.asarray(ensemble_evaluator(theta, iteration))
        if outputs.shape[0] != ensemble_size:
            raise ValueError("ensemble_evaluator returned the wrong number of rows")
        return outputs
    if forward_map is None:
        raise ValueError("either forward_map or ensemble_evaluator must be given")
    return np.asarray([forward_map(theta[j], rng) for j in range(ensemble_size)])


def _failed_members(
    outputs: Array,
    sentinel_value: float | None,
    failed_mask_fn: Callable[[Array], Array] | None,
) -> Array:
    """Boolean mask of failed members.

    The case decides what "failed" means via ``failed_mask_fn``; the default
    reproduces the original constant-sentinel detection (a row is failed when
    every entry equals ``sentinel_value``), and flags nothing when no sentinel
    value is given.
    """

    if failed_mask_fn is not None:
        mask = np.asarray(failed_mask_fn(outputs), dtype=bool).reshape(-1)
        if mask.shape[0] != outputs.shape[0]:
            raise ValueError("failed_mask_fn must return one flag per member")
        return mask
    if sentinel_value is None:
        return np.zeros(outputs.shape[0], dtype=bool)
    return np.all(outputs == sentinel_value, axis=1)


# ---------------------------------------------------------------- main loop


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
    sentinel_row_fn: Callable[[Array, Array], Array] | None = None,
    failed_mask_fn: Callable[[Array], Array] | None = None,
    post_update: Callable[[Array, int], Array] | None = None,
    iteration_callback: Callable[[dict], None] | None = None,
    jitter_mode: str = "relative",
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

    The optional hooks (``sentinel_row_fn``, ``failed_mask_fn``,
    ``post_update``, ``iteration_callback``, ``jitter_mode``) are documented in
    the module docstring; every default reproduces the linear-wave engine
    exactly.
    """

    if jitter_mode not in ("relative", "absolute"):
        raise ValueError(f"jitter_mode must be 'relative' or 'absolute', got {jitter_mode!r}")

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
    # Lower Cholesky factor of the exact Gamma, used for every objective
    # evaluation (whitening; no regularisation enters Phi).
    gamma_chol = np.linalg.cholesky(gamma_mat)

    # The replacement row for failed members is deterministic in (y, Gamma),
    # so build it once.  The vKdV convention is y + 10*sqrt(diag(Gamma)).
    replacement_row: Array | None = None
    if sentinel_row_fn is not None:
        replacement_row = np.asarray(
            sentinel_row_fn(y, np.diag(gamma_mat)), dtype=float
        ).reshape(-1)
        if replacement_row.shape[0] != q:
            raise ValueError("sentinel_row_fn must return a row of length q")

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

    def _apply_failure_policy(outputs: Array) -> tuple[Array, int]:
        """Flag failed members and, when a replacement row is declared, swap it in."""

        mask = _failed_members(outputs, sentinel_value, failed_mask_fn)
        count = int(np.sum(mask))
        if replacement_row is not None and count:
            outputs = np.array(outputs, dtype=float, copy=True)
            outputs[mask] = replacement_row
        return outputs, count

    for i in range(n_iter):
        # Step 1: evaluate the forward map G(theta) for every ensemble member.
        # outputs[j] = G(theta[j])
        outputs = _evaluate(theta, i, forward_map, ensemble_evaluator, rng, ensemble_size)
        outputs, n_failed = _apply_failure_policy(outputs)
        output_history.append(outputs.copy())
        sentinel_counts.append(n_failed)

        # Objective function for every single member:
        # Phi(theta) = 0.5 * (G(theta) - y)^T Gamma^{-1} (G(theta) - y).
        objective_each = _objective_values(outputs, y, gamma_chol)
        objective_member_history.append(objective_each.copy())

        # The paper-style figures compare the ensemble-mean output with y.
        # Therefore the iteration "best" is selected by the objective of
        # mean_j G(theta_j), not by the average of the individual objectives.
        output_mean = np.mean(outputs, axis=0, keepdims=True)
        objective = float(_objective_values(output_mean, y, gamma_chol)[0])
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
        system = _regularise(c_gg + gamma_mat, jitter, jitter_mode)
        gain_t = np.linalg.solve(system, c_tg.T)
        gain = gain_t.T

        # Step 4: y_n^j = y + eta^j, eta^j ~ N(0, Gamma), a fresh perturbation
        # of the data for every member (perturbed-observations EKI).
        if perturb_observations:
            y_ensemble = obs_rng.multivariate_normal(y, gamma_mat, size=ensemble_size)
        else:
            # No perturbation: the same observation y for every ensemble member.
            y_ensemble = np.broadcast_to(y, outputs.shape)

        # Step 5: update the parameters: theta^j = theta^j + K @ (y_n^j - G(theta^j)).
        innovations = y_ensemble - outputs
        theta = theta + innovations @ gain.T
        theta = _clip_bounds(theta, bounds)
        if post_update is not None:
            theta = np.asarray(post_update(theta, i), dtype=float)
            if theta.shape != theta_history[0].shape:
                raise ValueError("post_update changed the ensemble shape")
        theta_history.append(theta.copy())
        n_updates += 1

        # Checkpoint hook: the POST-update ensemble plus this iteration's
        # evaluation, and the live rng (its bit-generator state can be saved).
        if iteration_callback is not None:
            iteration_callback({
                "iteration": i,
                "thetas": theta.copy(),
                "outputs": outputs.copy(),
                "objectives": objective_each.copy(),
                "mean_objective": objective,
                "rng": rng,
            })

    # If the last Kalman update produced a new ensemble that has not yet been
    # evaluated, evaluate it once so it can also compete for the best objective.
    # The rng state is restored so this extra evaluation is invisible to any
    # subsequent draws.
    if len(output_history) < len(theta_history):
        rng_state = rng.bit_generator.state
        outputs = _evaluate(theta, n_iter, forward_map, ensemble_evaluator, rng, ensemble_size)
        rng.bit_generator.state = rng_state
        outputs, n_failed = _apply_failure_policy(outputs)
        output_history.append(outputs.copy())
        sentinel_counts.append(n_failed)
        objective_each = _objective_values(outputs, y, gamma_chol)
        objective_member_history.append(objective_each.copy())
        output_mean = np.mean(outputs, axis=0, keepdims=True)
        objective = float(_objective_values(output_mean, y, gamma_chol)[0])
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

    # Without a replacement row this keeps the original penalty-sentinel scheme
    # so archived runs stay reproducible; a sentinel row biases C^GG, so make
    # that hazard loud instead of silent.  With sentinel_row_fn the bias is a
    # declared convention (vKdV penalty rows), but a nonzero count is still
    # worth flagging.
    if verbose and sentinel_array.sum() > 0:
        print(
            f"WARNING: {int(sentinel_array.sum())} failed member evaluations "
            f"across {sentinel_array.size} ensembles; the empirical C^GG "
            "includes penalty rows rather than statistics -- check the run.",
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
        n_updates=n_updates,
        stop_reason=stop_reason,
    )
