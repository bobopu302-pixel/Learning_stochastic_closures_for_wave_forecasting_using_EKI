"""Dimension-agnostic Gaussian-process conditional mean for learned closures.

Modelling notes
---------------
- The node ``values`` are FREE PARAMETERS estimated by EKI, not data: they
  are "virtual observations" of the unknown closure function at fixed design
  nodes, and EKI moves them (together with the hyper-parameters) to fit the
  observed statistics.
- The mean interpolates the node values only in the limit nugget -> 0; for a
  finite nugget it is a smoothed regression through them, m(node_r) != v_r.
- Up to the fixed 1e-8 solver jitter, ``amplitude`` and ``nugget`` enter the
  mean ONLY through the ratio nugget^2 / amplitude^2:
  m(x) = k(x,N) [K + nugget^2 I]^{-1} v with k, K both proportional to
  amplitude^2, so multiplying amplitude and nugget by the same factor leaves
  m unchanged.  The pair is therefore not separately identifiable from the
  mean alone -- expect a ridge in (amplitude, nugget) space.
- RBF is an explicit, documented implementation assumption: the source papers
  name only a covariance function with amplitude and length scale, not its
  family.  Strict positivity is enforced by the caller's log
  parameterization; invalid values raise instead of being clamped, because a
  silent clamp makes the forward map discontinuous and hides a failed
  parameter update.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _validated_lengthscales(lengthscales, dim: int) -> np.ndarray:
    """Broadcast scalar-or-vector lengthscales to (dim,) and validate them."""

    ell = np.asarray(lengthscales, dtype=float).reshape(-1)
    if ell.size == 1:
        ell = np.full(dim, float(ell[0]))
    if ell.size != dim:
        raise ValueError(
            f"lengthscales must be scalar or length {dim}, got {ell.size} entries"
        )
    if not np.all(np.isfinite(ell)) or np.any(ell <= 0.0):
        raise ValueError("GP length scale must be finite and strictly positive")
    return ell


def product_rbf_kernel(
    x: np.ndarray, z: np.ndarray, amplitude: float, lengthscales
) -> np.ndarray:
    """Anisotropic product RBF k(x, z) = a^2 exp(-0.5 sum_k ((x_k-z_k)/l_k)^2).

    ``x`` is (n, d), ``z`` is (m, d); returns (n, m).  For d = 1 this equals
    the origin's 1-D ``rbf_kernel`` and for d = 2 the vKdV ``rbf2`` exactly.
    """

    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    amplitude = float(amplitude)
    if not np.isfinite(amplitude) or amplitude <= 0.0:
        raise ValueError("GP amplitude must be finite and strictly positive")
    ell = _validated_lengthscales(lengthscales, x.shape[1])
    scaled = (x[:, None, :] - z[None, :, :]) / ell[None, None, :]
    return amplitude**2 * np.exp(-0.5 * np.sum(scaled**2, axis=2))


def make_gp_mean(
    nodes: np.ndarray,
    values: np.ndarray,
    amplitude: float,
    lengthscales,
    nugget: float,
) -> Callable:
    """Build the GP conditional-mean callable m(x) = k(x, nodes) @ weights.

    Parameters
    ----------
    nodes : (R,) or (R, d) fixed design nodes (1-D closure curve or d-D surface).
    values : (R,) node values -- the EKI-estimated virtual observations.
    amplitude, lengthscales, nugget : kernel amplitude a, length scale(s)
        (scalar or (d,), anisotropic product RBF), and observation-noise
        standard deviation (the "nugget", tau in the (v, tau, a, ell) order).

    Representer solve (identical in both origins):
        (K_nn + (nugget^2 + 1e-8) I) w = v
    where the 1e-8 is numerical jitter, not a modelling term.

    The returned callable accepts either
    - one argument: any-shape array of scalars for d = 1 (scalar in -> float
      out, otherwise the input shape is preserved), or an (..., d) array of
      points for d > 1 (returns shape (...,)); or
    - d broadcastable coordinate arrays, e.g. ``m(u, s)`` on meshgrids for a
      2-D surface (the vKdV GPSurface calling style).

    For 1-D nodes the callable carries ``_gp_params = (nodes, weights,
    amplitude, lengthscale)`` -- the exact contract of the origin gpr.py --
    so numba-compiled simulators can inline
    m(x) = sum_r a^2 exp(-((x - node_r)/l)^2 / 2) w_r without calling back
    into Python (the run_lorenz63 fast path).  Multi-dimensional means do not
    carry the attribute.
    """

    nodes = np.asarray(nodes, dtype=float)
    if nodes.ndim == 1:
        nodes_2d = nodes[:, None]
    elif nodes.ndim == 2:
        nodes_2d = nodes
    else:
        raise ValueError(f"nodes must be (R,) or (R, d), got shape {nodes.shape}")
    n_nodes, dim = nodes_2d.shape

    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size != n_nodes:
        raise ValueError(f"expected {n_nodes} node values, got {values.size}")

    nugget = float(nugget)
    if not np.isfinite(nugget) or nugget <= 0.0:
        raise ValueError("GP observation noise (nugget) must be finite and strictly positive")

    amplitude = float(amplitude)
    ell = _validated_lengthscales(lengthscales, dim)

    # Kernel matrix on the nodes; the amplitude/lengthscale validation happens
    # inside product_rbf_kernel, mirroring the origin's rbf_kernel.
    k_nn = product_rbf_kernel(nodes_2d, nodes_2d, amplitude, ell)

    # Add observation-error variance to the diagonal.  The extra 1e-8 is
    # numerical jitter, not a paper-specified modelling term.
    k_nn = k_nn + (nugget**2 + 1e-8) * np.eye(n_nodes)

    # Representer weights: solve m(nodes) approx values for w.
    weights = np.linalg.solve(k_nn, values)

    def function(*coords):
        if len(coords) == 1:
            x_array = np.asarray(coords[0], dtype=float)
            if dim == 1:
                # Any-shape array of scalar inputs (origin 1-D behaviour).
                flat = x_array.reshape(-1, 1)
                out_shape = x_array.shape
            else:
                if x_array.ndim == 0 or x_array.shape[-1] != dim:
                    raise ValueError(
                        f"expected points with last axis {dim}, got shape {x_array.shape}"
                    )
                flat = x_array.reshape(-1, dim)
                out_shape = x_array.shape[:-1]
        elif len(coords) == dim:
            # d broadcastable coordinate arrays (GPSurface calling style).
            grids = np.broadcast_arrays(*[np.asarray(c, dtype=float) for c in coords])
            flat = np.column_stack([g.reshape(-1) for g in grids])
            out_shape = grids[0].shape
        else:
            raise ValueError(
                f"pass one points argument or {dim} coordinate arrays, got {len(coords)}"
            )
        result = product_rbf_kernel(flat, nodes_2d, amplitude, ell) @ weights
        if out_shape == ():
            return float(result[0])
        return result.reshape(out_shape)

    if dim == 1:
        # Expose the closed-form parameters so numba-compiled simulators can
        # inline the mean without calling back into Python.  Contract (kept
        # exactly as origin gpr.py): (nodes 1-D, weights, amplitude, lengthscale).
        function._gp_params = (
            nodes_2d[:, 0].copy(), weights.copy(), amplitude, float(ell[0])
        )

    return function


def make_gp_mean_from_theta(theta_gp: np.ndarray, nodes: np.ndarray) -> Callable:
    """Origin-style packed entry point: unpack theta_gp and call make_gp_mean.

    ``theta_gp`` = [node values (R) | nugget | amplitude | lengthscale(s)],
    the (v, tau, a, ell) order used by the case drivers; for d-dimensional
    nodes the tail carries 1 or d length scales.  All values are PHYSICAL
    (already decoded from log space by the caller).
    """

    nodes = np.asarray(nodes, dtype=float)
    n_nodes = nodes.shape[0]
    theta_gp = np.asarray(theta_gp, dtype=float).reshape(-1)
    if theta_gp.size < n_nodes + 3:
        raise ValueError(
            f"theta_gp needs at least {n_nodes + 3} entries "
            "(values | nugget | amplitude | lengthscale), got "
            f"{theta_gp.size}"
        )
    values = theta_gp[:n_nodes]
    nugget = float(theta_gp[n_nodes])
    amplitude = float(theta_gp[n_nodes + 1])
    lengthscales = theta_gp[n_nodes + 2:]
    return make_gp_mean(nodes, values, amplitude, lengthscales, nugget)
