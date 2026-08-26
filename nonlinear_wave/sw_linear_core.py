"""Linear vKdV core solver for the stepwise ladder (Step 2 world).
The Step-2 world deletes the vKdV nonlinearity; ``LambdaFluxSolver`` is
the frozen implicit-midpoint solver with a generalised entropy-split
flux -S d^{-1/2}(1/3)[lambda'(u) u_y + 2 (lambda(u))_y] in place of the
nonlinearity.  With ``set_lambda(zeros)`` it IS the linear core used by

  * sw_gpr_reference.py  (exact one-interval linear propagator for the
    GPR reference track of the H2 scheme);
  * any consumer that needs "the linear operator with the deleted
    nonlinearity" on the frozen discretisation.

Extracted verbatim (2026-08-16) from the retired sw_eki_s2.py (the
known-form lambda(u) inversion, archived in
archive/stepwise_s2_retired_20260816.tar) so the H2 pipeline no longer
imports the retired driver.
"""

from __future__ import annotations

import numpy as np

from sde_closure_core import StochasticImplicitMidpointDABCSolver

U_NODES = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])
EPSILON = 0.01


class LambdaInterpolant:
    """C1 lambda(u): PCHIP through the u-nodes, linear tails.

    A piecewise-LINEAR lambda has discontinuous lambda'(u) at the nodes
    (u=0 in particular is crossed everywhere), which makes the implicit
    midpoint fixed point chatter and stall.  PCHIP gives Lipschitz
    lambda' and an ANALYTIC derivative, so the split form stays exactly
    consistent (lambda' = d lambda/du of the same interpolant).
    """

    def __init__(self, nodes: np.ndarray) -> None:
        from scipy.interpolate import PchipInterpolator

        self._pch = PchipInterpolator(
            U_NODES, np.asarray(nodes, dtype=float), extrapolate=False
        )
        self._dpch = self._pch.derivative()
        self._lo, self._hi = U_NODES[0], U_NODES[-1]
        self._val_lo = float(self._pch(self._lo))
        self._val_hi = float(self._pch(self._hi))
        self._slope_lo = float(self._dpch(self._lo))
        self._slope_hi = float(self._dpch(self._hi))

    def __call__(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u = np.asarray(u, dtype=float)
        lam = self._pch(u)
        lamp = self._dpch(u)
        below, above = u < self._lo, u > self._hi
        if np.any(below):
            lam[below] = self._val_lo + self._slope_lo * (u[below] - self._lo)
            lamp[below] = self._slope_lo
        if np.any(above):
            lam[above] = self._val_hi + self._slope_hi * (u[above] - self._hi)
            lamp[above] = self._slope_hi
        return lam, lamp


class LambdaFluxSolver(StochasticImplicitMidpointDABCSolver):
    """Linear frozen operator + generalised entropy-split lambda flux."""

    def set_lambda(self, nodes: np.ndarray) -> None:
        self._lambda = LambdaInterpolant(nodes)
        self._dinv_sqrt = self.depth_ratio ** (-0.5)

    def nonlinear(self, normalized: np.ndarray) -> np.ndarray:
        u = self.to_surface(np.asarray(normalized, dtype=float))
        lam, lamp = self._lambda(u)
        flux = (
            lamp * np.asarray(self.d1 @ u).ravel()
            + 2.0 * np.asarray(self.d1 @ lam).ravel()
        ) / 3.0
        drift = -self.surface_to_green * self._dinv_sqrt * flux
        drift[:3] = 0.0
        drift[-3:] = 0.0
        return drift
