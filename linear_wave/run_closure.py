"""Command-line entry point for the modal closure workflow (thesis Chapter 4).

Origin: 2.Linear_wave_case/run_closure.py
Changes vs origin:
- added the sys.path bootstrap to the code_rp root so `from algorithms ...`
  resolves when running from the case folder;
- the --mode replot choice and every figure-related code path are removed
  (this release ships computation and data saving only); replay now means
  validate + write derived_metrics.json;
- imports updated: metrics (renamed from diagnostics) instead of
  diagnostics/plotting.

Examples (run from the linear_wave case folder)
-----------------------------------------------
Validate a stored bundle and rewrite the derived metrics::

    python run_closure.py --mode replay

Quantify the velocity readout, the spectral deconvolution and the objective
calibration::

    python run_closure.py --mode audit

Run the full SDE EKI calibration::

    python run_closure.py --mode recompute --output results/modal_closure_new
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# code_rp root, so the shared `algorithms` package resolves.  experiment.py
# repeats this bootstrap for the sake of spawned worker processes.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from modal_closure import audit, metrics
from modal_closure import experiment as case
from modal_closure.experiment import DEFAULT_RESULTS_DIR, N_WORKERS, run_experiment


PROJECT_ROOT = Path(__file__).resolve().parent
PRIMARY_RESULTS = PROJECT_ROOT / "results" / "modal_closure"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("replay", "validate", "audit", "recompute"),
        default="replay",
        help="replay is the safe default; recompute runs the full EKI calculation",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=PRIMARY_RESULTS,
        help="preserved result directory used by replay/validate/audit",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="new result directory used only by --mode recompute",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="allow --mode recompute to replace an existing result bundle",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=N_WORKERS,
        help="worker processes for the parallel stages (1 = serial)",
    )
    parser.add_argument(
        "--audit-repeats",
        type=int,
        default=audit.N_OUT_OF_SAMPLE,
        help="independent re-evaluations used by --mode audit",
    )
    parser.add_argument(
        "--n-g",
        type=int,
        default=None,
        help="forward runs averaged into one G_hat (default 100)",
    )
    parser.add_argument(
        "--n-gamma",
        type=int,
        default=None,
        help="independent reference records behind Gamma (default 200)",
    )
    parser.add_argument(
        "--t-record",
        type=float,
        default=None,
        help="analysis window T_y = T_G in seconds (default 1000)",
    )
    parser.add_argument(
        "--calib-probe-from",
        type=Path,
        default=None,
        help=(
            "results directory whose identified theta becomes the N_G "
            "calibration's near-optimum probe (the spec's second probe)"
        ),
    )
    parser.add_argument(
        "--calibrate-only",
        action="store_true",
        help="build Gamma, run the N_G calibration, print it and stop",
    )
    args = parser.parse_args()

    # Overrides applied before anything reads the module constants.  Worker
    # processes re-import the module and would not see these, which is why every
    # value that affects a forward run travels inside its job tuple.
    if args.n_g is not None:
        case.N_G = args.n_g
    if args.n_gamma is not None:
        case.N_GAMMA = args.n_gamma
    if args.calib_probe_from is not None:
        probe_bundle = args.calib_probe_from.resolve() / "bundle.npz"
        with np.load(probe_bundle, allow_pickle=False) as stored:
            # Bundle naming trap: `sde_best` is the final-ensemble mean.
            case.CALIB_PROBE_THETA = np.array(stored["sde_best"], dtype=float)
        print(f"N_G calibration probe loaded from {probe_bundle}")
    if args.t_record is not None:
        # N_DATA is derived from T_RECORD at import, so both must move together
        # or the reference records and the forward runs would use different
        # window lengths.  Workers receive each of them inside their job tuple.
        case.T_RECORD = float(args.t_record)
        case.N_DATA = int(round(case.T_RECORD / case.DT_DATA))

    if args.calibrate_only:
        y, gamma, parts = case.build_observation_and_gamma(n_workers=args.workers)
        case.report_error_model(y, gamma, parts)
        probes = case.calibration_probes()
        calib = case.calibrate_forward_averaging(
            parts["var_ref"], theta_probes=probes, n_workers=args.workers)
        ratio = calib["ratio"]
        print(f"\nratio = var_fwd/var_ref over probes "
              f"{list(probes)}, K = {calib['k']}")
        for p in (50, 75, 90, 95, 100):
            print(f"  p{p:<3d} {np.percentile(ratio, p):12.2f}"
                  f"   -> N_G >= {int(np.ceil(case.CALIB_SAFETY * np.percentile(ratio, p)))}")
        for n_g in (5, 10, 20, 50, 100):
            c = case.coverage_report(ratio, n_g)
            print(f"  N_G={n_g:4d}  covers {c['covered']:2d}/{c['total']}  " +
                  "  ".join(f"{k} {v[0]}/{v[1]}" for k, v in c["by_family"].items()))
        case.shutdown_pool()
        return

    results_dir = args.results.resolve()
    bundle_path = results_dir / "bundle.npz"

    if args.mode == "recompute":
        output_bundle = args.output.resolve() / "bundle.npz"
        if output_bundle.exists() and not args.force:
            raise FileExistsError(
                f"Refusing to overwrite {output_bundle}; pass --force explicitly"
            )
        run_experiment(args.output, n_workers=args.workers)
        return

    data = metrics.load_bundle(bundle_path)
    checks = metrics.validate_bundle(data)
    print(
        "validated bundle: "
        f"truth={checks['n_truth_components']} components, "
        f"model={checks['n_model_modes']} modes, "
        f"record={checks['long_record_seconds']:.0f} s"
    )

    if args.mode == "validate":
        return

    if args.mode == "audit":
        audit.run_audit(bundle_path, results_dir, n_out_of_sample=args.audit_repeats,
                        n_truth_reference=args.audit_repeats, n_workers=args.workers)
        return

    # --mode replay: validation (above) + the derived metrics, no figures.
    metrics_path = metrics.write_metrics(bundle_path)
    print(f"wrote metrics: {metrics_path}")


if __name__ == "__main__":
    main()
