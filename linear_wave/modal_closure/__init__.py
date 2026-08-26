"""Dispersion-locked stochastic modal closure for broadband linear waves.

Origin: 2.Linear_wave_case/modal_closure/__init__.py
Changes vs origin:
- provenance docstring added; BLAS/OpenMP pinning and re-exports unchanged.
"""

# Pin the BLAS/OpenMP thread count BEFORE numpy is imported anywhere in this
# package.  Every array here is small and the parallelism is across processes,
# so a threaded BLAS can only fight the workers.  This must happen at import
# time rather than when the pool is built: on Linux the default start method is
# fork, and a forked child inherits an OpenMP runtime that has already read
# these variables -- setting them later would be silently ignored and 190
# workers would each try to open 190 threads.
import os as _os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_var, "1")

from .experiment import (
    F_GRID,
    FREQUENCIES,
    GAUGES,
    M_MODES,
    N_COMPONENTS,
    simulate_grid,
    truth_reference,
)

__all__ = [
    "F_GRID",
    "FREQUENCIES",
    "GAUGES",
    "M_MODES",
    "N_COMPONENTS",
    "simulate_grid",
    "truth_reference",
]
