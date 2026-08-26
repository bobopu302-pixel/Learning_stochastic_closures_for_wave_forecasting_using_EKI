"""Result-bundle writer for the modal closure workflow.

Origin: 2.Linear_wave_case/modal_closure/plotting.py
Changes vs origin:
- renamed from plotting.py: the module never plotted anything itself, and this
  release ships no figures, so the misnomer is fixed;
- the replot() helper (which invoked the matplotlib figure code in the old
  diagnostics.py) is deleted; save_bundle is unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_bundle(results_dir: str | Path, **arrays: np.ndarray) -> Path:
    """Write every array needed to validate and re-analyse one completed run."""

    directory = Path(results_dir)
    directory.mkdir(parents=True, exist_ok=True)
    bundle_path = directory / "bundle.npz"
    np.savez(bundle_path, **arrays)
    return bundle_path
