"""Log-space encoding of strictly positive parameters for EKI.

Origin: 1. Reproduce_papers/common/code/parameterization.py
Changes vs origin: comments/docstrings only -- numerics identical.

EKI evolves parameters in an unconstrained latent space.  Strictly positive
physical parameters (noise scales, GP amplitude/length scales, damping rates)
are therefore evolved as their logarithms: ``log_encode`` maps physical ->
latent before the inversion starts, ``log_decode`` maps latent -> physical
inside every forward evaluation.  The engine (algorithms.eki) never sees this
convention -- encode/decode is the case driver's responsibility.
"""

from __future__ import annotations

import numpy as np

# Smallest physical value the latent representation may decode to; together
# with the +80 upper latent bound this keeps exp() away from under/overflow.
MIN_POSITIVE = 1e-12


def log_encode(value: np.ndarray | float) -> np.ndarray:
    """Physical -> latent for strictly positive parameters (elementwise log)."""

    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError("Strictly positive finite values are required before log encoding")
    return np.log(array)


def log_decode(value: np.ndarray | float) -> np.ndarray:
    """Latent -> physical (elementwise exp), refusing unsafe latent values."""

    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("Finite latent values are required before log decoding")
    # Prevent floating-point overflow without adding a discontinuity in the
    # scientifically relevant range.  Values outside this range are invalid;
    # see algorithms.eki.clip_latent for the declared guard EKI runs use.
    if np.any(array < np.log(MIN_POSITIVE)) or np.any(array > 80.0):
        raise ValueError("Latent positive parameter is outside the safe exponential range")
    return np.exp(array)


def encode_positive_columns(
    physical: np.ndarray, indices: list[int] | tuple[int, ...]
) -> np.ndarray:
    """Log-encode only the listed columns of a parameter vector or (J, p) ensemble."""

    encoded = np.asarray(physical, dtype=float).copy()
    if encoded.ndim not in (1, 2):
        raise ValueError("Expected a parameter vector or ensemble matrix")
    encoded[..., list(indices)] = log_encode(encoded[..., list(indices)])
    return encoded


def decode_positive_columns(latent: np.ndarray, indices: list[int] | tuple[int, ...]) -> np.ndarray:
    """Log-decode only the listed columns of a vector, ensemble, or ensemble history."""

    decoded = np.asarray(latent, dtype=float).copy()
    if decoded.ndim not in (1, 2, 3):
        raise ValueError("Expected a parameter vector, ensemble, or ensemble history")
    decoded[..., list(indices)] = log_decode(decoded[..., list(indices)])
    return decoded


def gp_positive_indices(node_count: int, *, offset: int = 0) -> tuple[int, int, int]:
    """Indices of observation noise, amplitude, and length scale in one GP block.

    The block layout is [node values (R) | nugget | amplitude | lengthscale]
    (the (v, tau, a, ell) order); node values stay on their natural scale, the
    three hyper-parameters are the strictly positive tail.
    """

    return offset + node_count, offset + node_count + 1, offset + node_count + 2
