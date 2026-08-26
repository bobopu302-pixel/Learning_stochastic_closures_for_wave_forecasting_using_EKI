"""v3 twin world -- env-driven patcher aligning Chapter-5 conventions with Chapter 4.

Origin: 3. KDV_nonlinear_case/v3_world.py
Changes vs origin: comments/docstrings only.

This module is LOAD-BEARING for the whole spec chain: every driver must
``import v3_world`` and call ``v3_world.ensure_patched()`` AFTER setting the
``SDE_*`` / ``SW_*`` environment variables and BEFORE the first import of any
``sde_closure_*`` module (those read the environment and bind names at import
time).  Always launch the drivers through ``v3_spec_chain.sh`` or replicate
its environment exactly.

Activated by the environment (all read once at import):

    SW_VERSION_DIR=v3spec      results/stepwise/<this>/ for truth, Gamma
                               records and every EKI output (default v2_fine)
    SW_DURATION_S=6600         total simulated time per record / forward
                               run (600 s burn-in + 6000 s analysis)
    SW_SYNTH_PERIOD_S=6600     incident TMA record synthesis period (must
                               be >= SW_DURATION_S so the incident record
                               never repeats inside one run)
    SW_FORWARD_PATHS=1         one closure-noise path per record / eval
    SW_CRN=0                   each EKI member draws its own forward seed
                               (no common random numbers) -- the spec drivers
                               use their own CRN scheme (spec_seed_root) and
                               do not consult this flag
    SW_REF_NEW_SEA_STATE=1     every Gamma reference record gets its own
                               incident phases (+ its own deterministic
                               baseline), like the Ch4 reference records

Mechanics.  Importing this module BEFORE any sde_closure_* module patches
two names in `sde_closure_context` that the frozen code looks up at
run time:

  * `_tma_inputs`  -- same body as the shared implementation in
    high_order_variable_depth_dabc_study._tma_inputs, with the
    synthesis period taken from SW_SYNTH_PERIOD_S instead of the
    hard-coded 1800 s (the incident-record sampling density is kept at
    the original 1800/32768 s);
  * `ModelAConfig` -- a frozen-dataclass subclass whose
    requested_duration_s default is SW_DURATION_S, so every
    `ModelAConfig(boundary_seed=...)` in the sw_* scripts runs the v3
    length without touching the frozen files.

Nothing is patched when the variables are absent, so the v2 chain is
reproduced bit-for-bit with the same files.
"""

from __future__ import annotations

import dataclasses
import os

VERSION_DIR = os.environ.get("SW_VERSION_DIR", "")
DURATION_S = float(os.environ.get("SW_DURATION_S", "1800"))
SYNTH_PERIOD_S = float(os.environ.get("SW_SYNTH_PERIOD_S", "1800"))
FORWARD_PATHS = int(os.environ.get("SW_FORWARD_PATHS", "16"))
CRN = os.environ.get("SW_CRN", "1") == "1"
REF_NEW_SEA_STATE = os.environ.get("SW_REF_NEW_SEA_STATE", "0") == "1"
ANALYSIS_START_S = 600.0

_ORIGINAL_SAMPLE_DT_S = 1800.0 / 32768.0   # incident-record sampling kept


def version_dir(default: str) -> str:
    return VERSION_DIR or default


def member_seed_root(eki_seed: int, iteration: int, member: int) -> tuple:
    """Forward-noise seed root for one EKI member in one iteration.

    CRN (v2): all members share (eki_seed, 3000+iteration) and only the
    path index differs.  v3 (SW_CRN=0): the member index is part of the
    root, so every member has its own independent noise stream.  The spec
    drivers use spec_seed_root instead; this helper stays because the
    validation scripts import v3_world for its constants.
    """
    if CRN:
        return (eki_seed, 3000 + iteration)
    return (eki_seed, 3000 + iteration, member)


def _rebind(name: str, original, replacement) -> None:
    """Replace `name` wherever the ORIGINAL object is currently bound.

    Frozen modules bind these names at import time
    (`from ... import _tma_inputs`), so patching a single module is not
    enough: sde_closure_eki._init_worker_with would keep calling the
    original and every forward evaluation would then be driven by a
    DIFFERENT incident wave record than the truth (observed 2026-08-24:
    devrms 7x too large, flat in space, while the marginal statistics
    still matched).  The source module is patched too, so modules
    imported later get the replacement as well.
    """
    import sys
    patched = []
    for module in list(sys.modules.values()):
        if module is None:
            continue
        try:
            if getattr(module, name, None) is original:
                setattr(module, name, replacement)
                patched.append(getattr(module, "__name__", "?"))
        except Exception:                      # pragma: no cover
            continue
    _PATCH_LOG.append((name, patched))


_PATCH_LOG: list = []


def _apply_patches() -> None:
    import sde_closure_context as ctx
    if SYNTH_PERIOD_S != 1800.0:
        from high_order_variable_depth_dabc_study import (
            CoastalParameters, StudyConfig, _tma_inputs as _orig,
            build_modal_three_trace_lifting,
        )
        from high_order_incident_lifting import PeriodicBoundarySpectrum
        from sea_state_boundary import SeaStateParameters, TMASeaState
        import numpy as np

        if SYNTH_PERIOD_S < DURATION_S:
            raise SystemExit(
                f"SW_SYNTH_PERIOD_S={SYNTH_PERIOD_S} < SW_DURATION_S="
                f"{DURATION_S}: the incident record would repeat")

        def _tma_inputs_v3(p, config, dy, dt):
            parameters = SeaStateParameters(
                significant_wave_height_m=0.3,
                peak_period_s=15.0,
                peak_enhancement_gamma=3.3,
                water_depth_m=15.0,
                frequency_min_hz=0.03,
                high_frequency_taper_start_hz=0.085,
                frequency_max_hz=0.105,
                synthesis_period_s=SYNTH_PERIOD_S,
                ramp_duration_s=p.boundary_ramp_s,
                random_seed=config.random_seed,
            )
            sea_state = TMASeaState(parameters)
            spectrum = PeriodicBoundarySpectrum(
                angular_frequency=2.0 * np.pi * sea_state.frequencies_hz
                * p.time_ref_s,
                complex_amplitude=(sea_state.amplitudes_m / p.a_ref_m
                                   * np.exp(1.0j * sea_state.phases_rad)),
                period=parameters.synthesis_period_s / p.time_ref_s,
                sample_dt=_ORIGINAL_SAMPLE_DT_S / p.time_ref_s,
                removed_mean=0.0,
                reconstruction_relative_l2=0.0,
                retained_variance_fraction=1.0,
            )
            lifting, metadata = build_modal_three_trace_lifting(
                spectrum, "discrete_c6c4", 1.0, p.mu / 6.0, dy, dt,
                p.boundary_ramp_s / p.time_ref_s, d1_order=6,
            )

            def physical_g0(time_value: float) -> float:
                return float(sea_state.truth_m(time_value * p.time_ref_s)
                             / p.a_ref_m)

            metadata = dict(metadata)
            metadata["synthesis_period_s"] = SYNTH_PERIOD_S
            metadata["n_components"] = int(sea_state.frequencies_hz.size)
            return sea_state, lifting, lifting.callables(physical_g0), metadata

        _tma_inputs_v3.__wrapped__ = _orig
        _rebind("_tma_inputs", _orig, _tma_inputs_v3)

    if DURATION_S != 1800.0:
        Base = ctx.ModelAConfig

        @dataclasses.dataclass(frozen=True)
        class ModelAConfig(Base):        # noqa: N801 (keeps the name)
            requested_duration_s: float = DURATION_S

        ModelAConfig.__module__ = Base.__module__
        ModelAConfig.__qualname__ = Base.__qualname__
        _rebind("ModelAConfig", Base, ModelAConfig)


_PATCHED = False


def ensure_patched() -> None:
    """Apply the v3 patches once.  Call AFTER all SDE_* env variables are
    set and BEFORE the first import of sde_closure_eki / sde_closure_context
    (closure_config reads the environment at import time)."""
    global _PATCHED
    if not _PATCHED:
        _apply_patches()
        _PATCHED = True


def describe() -> dict:
    return {
        "version_dir": VERSION_DIR or "(default)",
        "duration_s": DURATION_S,
        "analysis_start_s": ANALYSIS_START_S,
        "synthesis_period_s": SYNTH_PERIOD_S,
        "forward_paths": FORWARD_PATHS,
        "common_random_numbers": CRN,
        "ref_records_new_sea_state": REF_NEW_SEA_STATE,
        "rebound": {name: mods for name, mods in _PATCH_LOG},
    }
