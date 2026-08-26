"""Reference wave fields for the GP-closure case: the deterministic-amplitude
generator the thesis run observes, and the stochastically forced variant it was
compared against, side by side.

Why this exists
---------------
The linear-wave case's reference field is a sum of many components with
FIXED amplitudes and random phases:

    eta(x, t) = sum_j a_j cos(k_j x - omega_j t + phi_j),   phi_j ~ U[0, 2pi)

Each realisation differs only by its phases.  Because the amplitudes never
change, any quantity that averages over a long record -- band energy above all
-- comes out almost the same every time: measured at T = 12 000 s, the
reference band energies repeat to 0.07-0.54% of their own value, so
`var_ref` for those statistics is 1e-12 to 1e-8.

The closure model is not like that.  It is white-noise driven, so its band
energy is genuinely random from record to record.  The ratio
`var_fwd / var_ref` therefore reaches 1e5 on the low-energy bands, no
affordable amount of forward averaging removes it, and `Gamma = diag(var_ref)`
spans twelve orders of magnitude, which is what destabilised the inversion.

The fix explored here is to give the reference the same kind of randomness the
model has: each component becomes a damped oscillator driven by white noise,

    dq_j = p_j dt
    dp_j = (-omega_j^2 q_j - delta_j p_j) dt + sqrt(sigma_j) dW_j

with `sigma_j` set so the stationary variance `sigma_j / (2 delta_j omega_j^2)`
equals the old `a_j^2 / 2`.  The mean spectrum is therefore unchanged; what
changes is that the amplitude of each component now fluctuates instead of being
frozen.  `delta_truth` controls the linewidth: small values give sharp lines
and slowly varying amplitudes, large values give broad lines.

Both generators are kept here so every comparison is like for like.  The
frozen thesis run (N = 10000 components) observes the DETERMINISTIC generator;
the stochastic one is retained because the module's constants and the
comparison studies depend on it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# The exact OU propagator and the dispersion solve are shared with the linear
# wave case rather than copied: two copies of a propagator drift apart, and a
# comparison between the two reference systems is only meaningful if the
# numerics underneath them are identical.  In the release layout that case
# lives in code_rp/linear_wave/.
_CODE_RP_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_CODE_RP_ROOT), str(_CODE_RP_ROOT / "linear_wave")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from modal_closure.numerics import modal_propagator          # noqa: E402
from modal_closure.truth import _solve_wavenumber            # noqa: E402


# --- case definition, identical to the linear wave case ---------------------
# The component count is read from the environment so a sweep can vary it and
# have SPAWNED WORKERS see the same value.  Reassigning the module attribute
# would not survive a spawn -- the worker re-imports and gets the default -- and
# every derived constant below (frequencies, amplitudes, wavenumbers) is
# computed at import time from this number.
N_COMPONENTS = int(os.environ.get("STOCHASTIC_N", "10000"))
F_MIN, F_MAX = 0.3, 1.8
TRUTH_FREQ_SEED = 7
SPEC_PEAK_HZ = 1.0
SPEC_WIDTH_HZ = 0.35
TARGET_VAR = 1.0
DEPTH = 100.0

_freq_rng = np.random.default_rng(TRUTH_FREQ_SEED)
FREQUENCIES = np.sort(_freq_rng.uniform(F_MIN, F_MAX, N_COMPONENTS))
_envelope = np.exp(-0.5 * ((FREQUENCIES - SPEC_PEAK_HZ) / SPEC_WIDTH_HZ) ** 2)
AMPLITUDES = _envelope * np.sqrt(TARGET_VAR / np.sum(_envelope ** 2 / 2.0))

K_GAUGES = 10
DX = 2.0 / 9.0
GAUGES = np.arange(K_GAUGES) * DX
DT_DATA = 0.05
T_BURN = 12.0

OMEGA = 2.0 * np.pi * FREQUENCIES
OMEGA2 = OMEGA ** 2
WAVENUMBERS = np.array([_solve_wavenumber(w, DEPTH) for w in OMEGA])

# Stationary variance each component must carry, so that the mean spectrum of
# the stochastic reference matches the deterministic one exactly.
COMPONENT_VARIANCE = AMPLITUDES ** 2 / 2.0

# Default linewidth of the stochastic reference.  0.05 1/s gives a Lorentzian
# half-width of delta/(4 pi) = 0.004 Hz, well inside the 0.167 Hz band spacing
# of the ten-mode model grid, so the mean spectrum stays where it was.
DELTA_TRUTH = 0.05

# Chunk the time axis so peak memory does not scale with the record length --
# and, since N_COMPONENTS is now a knob, so it does not scale with the component
# count either.  Both generators hold arrays of shape (block, gauges,
# components); 4e6 elements is ~32 MB per array, which at 188 workers is the
# difference between 6 GB and 240 GB.
TIME_BLOCK = max(200, 4_000_000 // (K_GAUGES * N_COMPONENTS))


def component_sigma(delta_truth=DELTA_TRUTH):
    """sigma_j such that Var(q_j) = sigma_j / (2 delta_j omega_j^2) = a_j^2/2."""

    delta = np.full(N_COMPONENTS, float(delta_truth))
    return 2.0 * delta * OMEGA2 * COMPONENT_VARIANCE


def deterministic_fields(n_steps, seed, dt=DT_DATA):
    """The existing reference: fixed amplitudes, random phases.

    Returned as (eta, v) on the gauge grid, where v is the modal-skeleton
    velocity readout -- the same observation functional the model uses, not a
    pathwise time derivative.

    Written as a complex matrix product rather than a cosine over a
    (time, gauge, component) tensor.  Using

        eta(x_g, t) = Re{ sum_j (a_j e^{i k_j x_g}) e^{i(phi_j - omega_j t)} }
                    = Re{ (E A^T)[t, g] }

    the transcendental work drops from T*G*N to T*N and the gauge expansion
    becomes a BLAS call.  That matters at large N: the tensor form allocated a
    (block, 10, 6000) array per block and was memory-bandwidth bound, running at
    46% CPU on 192 cores.  The phase column advances by the recurrence
    E[t+1] = E[t] * exp(-i omega dt), re-anchored with an exact exponential at
    each block start so the rounding cannot accumulate over the whole record.
    """

    rng = np.random.default_rng(seed)
    phase = rng.uniform(0.0, 2.0 * np.pi, N_COMPONENTS)

    # Gauge factors, complex, computed once: A for eta, B for v.
    kx = np.outer(GAUGES, WAVENUMBERS)                       # (gauge, comp)
    a_gauge = (AMPLITUDES * np.exp(1j * kx)).T               # (comp, gauge)
    b_gauge = (AMPLITUDES * OMEGA * np.exp(1j * kx)).T

    eta = np.empty((n_steps, K_GAUGES))
    v = np.empty((n_steps, K_GAUGES))
    step_factor = np.exp(-1j * OMEGA * dt)

    # E is (block, N) complex, so the block is sized against that, not against
    # the old three-dimensional tensor.
    block = max(64, 8_000_000 // max(N_COMPONENTS, 1))
    e_block = np.empty((block, N_COMPONENTS), dtype=complex)

    for start in range(0, n_steps, block):
        stop = min(start + block, n_steps)
        e_block[0] = np.exp(1j * (phase - OMEGA * (start * dt)))
        for offset in range(1, stop - start):
            e_block[offset] = e_block[offset - 1] * step_factor
        rows = e_block[:stop - start]
        eta[start:stop] = (rows @ a_gauge).real
        v[start:stop] = (rows @ b_gauge).imag
    return eta, v


def stochastic_fields(n_steps, seed, delta_truth=DELTA_TRUTH, dt=DT_DATA,
                      t_burn=T_BURN):
    """The stochastically forced reference.

    Each component is an exactly propagated OU oscillator started from its own
    stationary law, so there is no spin-up transient; the burn-in is kept only
    as a numerical check, matching the model's convention.
    """

    delta = np.full(N_COMPONENTS, float(delta_truth))
    sigma = component_sigma(delta_truth)
    phi, noise_root = modal_propagator(OMEGA2, delta, sigma, dt)
    p00, p01, p10, p11 = phi[:, 0, 0], phi[:, 0, 1], phi[:, 1, 0], phi[:, 1, 1]
    l00, l01, l10, l11 = (noise_root[:, 0, 0], noise_root[:, 0, 1],
                          noise_root[:, 1, 0], noise_root[:, 1, 1])

    rng = np.random.default_rng(seed)
    n_burn = int(round(t_burn / dt))
    q = rng.normal(size=N_COMPONENTS) * np.sqrt(sigma / (2.0 * delta * OMEGA2))
    qd = rng.normal(size=N_COMPONENTS) * np.sqrt(sigma / (2.0 * delta))

    # Same travelling-wave readout as the deterministic generator: the spatial
    # phase enters through cos(kx)/sin(kx), so both fields propagate to the
    # right and satisfy the same dispersion relation.
    kx = np.outer(GAUGES, WAVENUMBERS)
    cq, sq = np.cos(kx), np.sin(kx)

    eta = np.empty((n_steps, K_GAUGES))
    v = np.empty((n_steps, K_GAUGES))

    # Both the noise and the modal history are handled a block at a time.
    # Storing the full (n_steps, N_COMPONENTS) history and drawing the full
    # (n_steps, N_COMPONENTS, 2) noise up front costs 1.9 GB for a 4000 s record
    # at 1000 components -- times 188 workers, that is the whole machine.  Held
    # to one block, peak memory is 32 MB and stops scaling with the record
    # length or the component count.
    #
    # The random numbers are unchanged: numpy fills in C order, so consecutive
    # block draws reproduce the single large draw bit for bit (checked).  The
    # readout is not bitwise identical to the unblocked version -- a
    # (block, N) @ (N, gauges) product sums in a different order from a
    # (n_steps, N) one -- but the discrepancy is 2e-15 on eta and 1e-14 on v,
    # i.e. floating-point non-associativity, not a change of algorithm.
    total = n_burn + n_steps
    q_block = np.empty((TIME_BLOCK, N_COMPONENTS))
    qd_block = np.empty((TIME_BLOCK, N_COMPONENTS))

    for start in range(0, total, TIME_BLOCK):
        stop = min(start + TIME_BLOCK, total)
        z_block = rng.normal(size=(stop - start, N_COMPONENTS, 2))
        kept = 0
        first_kept = None
        for offset in range(stop - start):
            z0, z1 = z_block[offset, :, 0], z_block[offset, :, 1]
            q_new = p00 * q + p01 * qd + l00 * z0 + l01 * z1
            qd_new = p10 * q + p11 * qd + l10 * z0 + l11 * z1
            q, qd = q_new, qd_new
            step = start + offset
            if step >= n_burn:
                if first_kept is None:
                    first_kept = step - n_burn
                q_block[kept] = q
                qd_block[kept] = qd
                kept += 1
        if kept:
            qs = q_block[:kept]
            qds = qd_block[:kept]
            lo = first_kept
            eta[lo:lo + kept] = qs @ cq.T - (qds / OMEGA) @ sq.T
            v[lo:lo + kept] = (qs * OMEGA) @ sq.T + qds @ cq.T
    return eta, v
