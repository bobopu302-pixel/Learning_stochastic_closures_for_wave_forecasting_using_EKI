"""Isolated nonlinear discretisation candidates for Experiment 16 screening.

Origin: 3. KDV_nonlinear_case/high_order_nonlinear_candidates.py
Changes vs origin: comments/docstrings only (this provenance header added).

This module is intentionally separate from the frozen Experiment-14/15
solvers.  None of the classes here is a production solver.  They reuse the
same linear C6-D1/C4-D3 operator, Crank--Nicolson/Adams--Bashforth time step,
three incident traces and three linear DABC rows, while changing only the
explicit interior nonlinear drift.

The first candidate is the C6 split (often called skew-symmetric or entropy-
split) discretisation

    N_v = S[-gamma/3 * (u D1(u) + D1(u**2))],

where ``v=S*u``, ``S=d**(1/4)`` and ``gamma=(3 epsilon/2)d**(-1/2)``.  The
factor 1/3 makes the continuous expression identical to ``-gamma*u*u_y``.
The derivative matrices have empty first/last three rows, so the returned
drift is exactly zero on the six rows that are replaced by prescribed traces
and DABC constraints in the inherited marcher.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import spmatrix

from high_order_variable_depth_dabc import CoastalHighOrderCNAB2DABCSolver


def split_entropy_nonlinear_drift(
    surface: np.ndarray,
    gamma: np.ndarray,
    surface_to_green: np.ndarray,
    d1: spmatrix,
) -> np.ndarray:
    """Return the C6 split nonlinear drift in the normalised state.

    Parameters are dimensionless and one-dimensional.  ``d1`` must be the
    same C6 first-derivative matrix used by the linear operator.  Its first
    and last three rows are assumed to be empty boundary rows.
    """

    u = np.asarray(surface, dtype=float)
    coefficient = np.asarray(gamma, dtype=float)
    scale = np.asarray(surface_to_green, dtype=float)
    if u.ndim != 1:
        raise ValueError("surface must be one-dimensional")
    if coefficient.shape != u.shape or scale.shape != u.shape:
        raise ValueError("gamma and surface_to_green must match surface")
    if d1.shape != (u.size, u.size):
        raise ValueError("d1 must be square and match surface")
    if not (
        np.all(np.isfinite(u))
        and np.all(np.isfinite(coefficient))
        and np.all(np.isfinite(scale))
    ):
        raise ValueError("split-drift inputs must be finite")

    derivative_u = np.asarray(d1 @ u).ravel()
    derivative_square = np.asarray(d1 @ (u * u)).ravel()
    surface_drift = -(coefficient / 3.0) * (
        u * derivative_u + derivative_square
    )
    normalised_drift = scale * surface_drift

    # Make the boundary-row contract explicit even though the derivative
    # matrices already have zero rows there.
    normalised_drift[:3] = 0.0
    normalised_drift[-3:] = 0.0
    return normalised_drift


class CoastalHighOrderSplitCNAB2DABCSolver(
    CoastalHighOrderCNAB2DABCSolver
):
    """Experiment-16 C6 split-nonlinearity screening candidate.

    The inherited ``run`` method supplies exactly the Experiment-15 CNAB2
    recurrence and boundary treatment.  Only ``nonlinear`` is replaced.
    """

    candidate_name = "C6_entropy_split"

    def nonlinear(self, normalized: np.ndarray) -> np.ndarray:
        """Evaluate the split drift on the six-row interior stencil."""

        values = np.asarray(normalized, dtype=float)
        if values.shape != self.y.shape:
            raise ValueError("nonlinear expects one normalized field matching y")
        return split_entropy_nonlinear_drift(
            self.to_surface(values),
            self.gamma,
            self.surface_to_green,
            self.d1,
        )
