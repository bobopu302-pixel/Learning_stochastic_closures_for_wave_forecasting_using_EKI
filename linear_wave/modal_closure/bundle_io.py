"""Result-bundle writer for the modal closure workflow.
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
