"""Case (a) observation vector, theta packing, and prior ensemble for Lorenz 96.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make the shared `algorithms` package importable when this module is loaded
# from the case folder (the entry script run_spec.py does the same).
_CODE_RP_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_RP_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_RP_ROOT))

from algorithms.parameterization import gp_positive_indices  # noqa: E402
from algorithms.statistics import centered_first_second_moments  # noqa: E402


# Case (a) data vector (q = 44): first moments (8), centered variances (8),
# and centered covariances (28) of the first 8 slow variables, in the paper's
# figure order.  Restricting to 8 of the 36 components matches the paper's
# displayed data vector.
def case_a_statistics(samples: np.ndarray) -> np.ndarray:
    return centered_first_second_moments(samples, n_components=8)


# Split theta into GP parameters, linear damping coefficient, and sigma.
# Layout: [node values (R) | nugget | amplitude | lengthscale]
#         [+ linear coefficient if learned] [+ sqrt(sigma) if stochastic].
# The evolved stochastic parameter is sqrt(sigma); it is squared here so the
# simulator receives the noise VARIANCE sigma.
def unpack_theta_parts(
    theta: np.ndarray,
    nodes: np.ndarray,
    *,
    learn_linear_coefficient: bool,
    stochastic: bool,
    default_linear: float,
) -> tuple[np.ndarray, float, float]:
    block = nodes.size + 3
    theta_gp = np.asarray(theta[:block], dtype=float)
    index = block
    if learn_linear_coefficient:
        linear_coefficient = float(theta[index])
        index += 1
    else:
        linear_coefficient = default_linear
    sigma = float(theta[index]) ** 2 if stochastic else 0.0
    return theta_gp, linear_coefficient, sigma


# Draw the initial EKI ensemble from paper Section 4.2 ranges (no parameter
# bounds).  Column order matches unpack_theta_parts; all values are on the
# physical scale -- the driver log-encodes the positive columns before EKI.
def initial_ensemble(
    *,
    rng: np.random.Generator,
    ensemble_size: int,
    nodes: np.ndarray,
    length_low: float,
    length_high: float,
    learn_linear_coefficient: bool,
    stochastic: bool,
) -> np.ndarray:
    parts = [
        rng.uniform(-1.0, 1.0, size=(ensemble_size, nodes.size)),   # node values
        rng.uniform(0.1, 1.0, size=(ensemble_size, 1)),             # nugget
        rng.uniform(0.1, 1.0, size=(ensemble_size, 1)),             # amplitude
        rng.uniform(length_low, length_high, size=(ensemble_size, 1)),  # lengthscale
    ]
    if learn_linear_coefficient:
        parts.append(rng.uniform(1.0 / 3.0, 20.0 / 3.0, size=(ensemble_size, 1)))
    if stochastic:
        parts.append(rng.uniform(0.01, 10.0, size=(ensemble_size, 1)))  # sqrt(sigma)
    return np.hstack(parts)


def positive_parameter_indices(
    nodes: np.ndarray, *, learn_linear_coefficient: bool, stochastic: bool
) -> tuple[int, ...]:
    """Columns represented in log space during EKI.

    GP node values remain signed.  Kernel nugget/amplitude/length scale, the
    optional damping coefficient, and sqrt(sigma) are strictly positive.
    """

    indices = list(gp_positive_indices(nodes.size))
    next_index = nodes.size + 3
    if learn_linear_coefficient:
        indices.append(next_index)
        next_index += 1
    if stochastic:
        indices.append(next_index)
    return tuple(indices)
