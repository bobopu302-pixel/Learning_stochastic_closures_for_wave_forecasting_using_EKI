"""Three-way physical comparison for S2, plus the nonlinearity-signal test.

Origin: 3. KDV_nonlinear_case/v3_s2_threeway.py
Changes vs origin:
- deleted the four matplotlib figures (D7 profiles, D8 invariant, D9
  spectra, D10 nonlinearity signal) -- this release ships no plotting.
  threeway_data.npz keeps its origin layout; the data that only existed
  inside the D8/D9 figure loops (per-gauge invariant-measure histograms
  and spectra of all four field sets) is now saved to a NEW companion
  file threeway_pdf_spectra.npz; the D10 displacement curves and the
  recovered-fraction numbers were already printed and are additionally
  stored in threeway_data.npz under d10_* keys (additive keys only);
- comments/docstrings translated and polished.

Compares four fields on identical statistics, computed the same way for
all of them so the comparison needs no reference to the inversion's own
statistic definitions:

    observed        the truth record that formed y (1 path)
    inversion       final ensemble mean of the S2 posterior
    true params     m^dagger with p = -1/2 -- the model-class floor
    no nonlinearity m == 0, true noise -- the control

The decisive quantity is the *nonlinearity signal*: how far the true
nonlinear term moves each statistic away from the m == 0 control, and
how much of that displacement the learned term reproduces.

Consumes (in <run>/analysis/): validation_fields.npz,
validation_fields_truth.npz, validation_fields_zero.npz -- produced by
v3_s2_validation.py and v3_s2_val_truth.py.

    python v3_s2_threeway.py --branch diag
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path as _Path

import numpy as np

_HERE = _Path(__file__).resolve().parent
ROOT = _HERE / "results" / "stepwise" / os.environ.get("SW_VERSION_DIR",
                                                       "v3spec")


def moments(field: np.ndarray) -> tuple:
    """sigma, skewness, kurtosis over time, averaged over paths.

    `field` is (paths, time, station) or (time, station).
    """
    f = np.atleast_3d(field) if field.ndim == 3 else field[None, ...]
    mu = f.mean(axis=1, keepdims=True)
    d = f - mu
    var = (d ** 2).mean(axis=1)
    sd = np.sqrt(var)
    skew = (d ** 3).mean(axis=1) / np.maximum(sd ** 3, 1e-30)
    kurt = (d ** 4).mean(axis=1) / np.maximum(var ** 2, 1e-30) - 3.0
    return sd.mean(0), skew.mean(0), kurt.mean(0)


def pdf(sample: np.ndarray, edges: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(sample.ravel(), bins=edges, density=True)
    return hist


def psd(series: np.ndarray, dt: float) -> tuple:
    """Welch-free periodogram averaged over paths, per station."""
    x = series if series.ndim == 3 else series[None, ...]
    n = x.shape[1]
    win = np.hanning(n)[None, :, None]
    xw = (x - x.mean(axis=1, keepdims=True)) * win
    spec = np.abs(np.fft.rfft(xw, axis=1)) ** 2
    norm = (win[0, :, 0] ** 2).sum() * n / (n * dt) / n
    freq = np.fft.rfftfreq(n, dt)
    return freq, spec.mean(0) / max(norm, 1e-30)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--branch", default="diag")
    a = p.parse_args()

    ana = ROOT / f"H2_eki_{a.branch}" / "analysis"
    learned = np.load(ana / "validation_fields.npz")
    truthp = np.load(ana / "validation_fields_truth.npz")
    zero = np.load(ana / "validation_fields_zero.npz")

    t = np.asarray(learned["times_s"], dtype=float)
    dt = float(np.median(np.diff(t)))
    dense_x = np.asarray(learned["dense_x"], dtype=float)
    gauge_x = np.asarray(learned["gauge_x"], dtype=float)
    obs_dense = np.asarray(learned["dense_truth"], dtype=float)
    obs_gauge = np.asarray(learned["gauge_truth"], dtype=float)

    # (name, dense field, gauge field) for each of the four sets; the
    # tags below key the saved arrays.
    sets = [
        ("observed record", obs_dense, obs_gauge),
        ("inversion (S2)", np.asarray(learned["dense_model"], float),
         np.asarray(learned["gauge_model"], float)),
        ("true parameters", np.asarray(truthp["dense_model"], float),
         np.asarray(truthp["gauge_model"], float)),
        ("no nonlinearity (m == 0)",
         np.asarray(zero["dense_model"], float),
         np.asarray(zero["gauge_model"], float)),
    ]
    tags = {"observed record": "obs", "inversion (S2)": "inv",
            "true parameters": "tru", "no nonlinearity (m == 0)": "zero"}
    zname = "no nonlinearity (m == 0)"
    print("[3way] %d dense stations, %d gauges, %d samples, dt = %.4f s"
          % (dense_x.size, gauge_x.size, t.size, dt), flush=True)
    for name, d, _g in sets:
        print("   %-28s paths = %d" % (name, 1 if d.ndim == 2 else d.shape[0]),
              flush=True)

    mom = {name: moments(d) for name, d, _g in sets}

    # ------------------------- invariant measures + spectra (data only)
    n_g = min(4, gauge_x.size)
    pdf_store: dict[str, np.ndarray] = {}
    psd_store: dict[str, np.ndarray] = {}
    edges_store = []
    freq = None
    for j in range(n_g):
        lim = 4.0 * float(np.std(obs_gauge[:, j]))
        edges = np.linspace(-lim, lim, 61)
        edges_store.append(edges)
        for name, _d, g in sets:
            sample = g[..., j]
            key = f"pdf_{tags[name]}_g{j}"
            pdf_store[key] = pdf(sample, edges)
            f_hz, sp = psd(g[..., j] if g.ndim == 3 else g[:, j][:, None],
                           dt)
            freq = f_hz
            psd_store[f"psd_{tags[name]}_g{j}"] = sp[:, 0]
    np.savez_compressed(
        ana / "threeway_pdf_spectra.npz",
        gauge_x=gauge_x[:n_g], dt=dt,
        pdf_edges=np.stack(edges_store),
        psd_freq_hz=freq,
        **pdf_store, **psd_store)

    # ------------------------------------- the nonlinearity signal (D10)
    recovered = {}
    d10 = {}
    for k, key in enumerate(("sigma", "skew", "kurt")):
        base_k = mom[zname][k]
        d_true = mom["true parameters"][k] - base_k
        d_inv = mom["inversion (S2)"][k] - base_k
        denom = float(np.sqrt((d_true ** 2).mean()))
        num = float(np.sqrt((d_inv ** 2).mean()))
        proj = float((d_inv @ d_true) / max((d_true @ d_true), 1e-30))
        recovered[key] = (num / max(denom, 1e-30), proj)
        d10[f"d10_true_{key}"] = d_true
        d10[f"d10_learned_{key}"] = d_inv

    # ------------------------------------------------------------ report
    print("\n[3way] displacement from the m == 0 control")
    print("   %-8s %14s %14s %10s" % ("stat", "|true| (rms)", "|learned|",
                                      "projection"))
    for k, key in enumerate(("sigma", "skew", "kurt")):
        base_k = mom[zname][k]
        d_true = mom["true parameters"][k] - base_k
        d_inv = mom["inversion (S2)"][k] - base_k
        print("   %-8s %14.5g %14.5g %10.3f"
              % (key, np.sqrt((d_true ** 2).mean()),
                 np.sqrt((d_inv ** 2).mean()), recovered[key][1]))
    print("\n[3way] distance to the observed record (rms over stations)")
    for name, _d, _g in sets[1:]:
        row = []
        for k, key in enumerate(("sigma", "skew", "kurt")):
            row.append(float(np.sqrt(((mom[name][k]
                                       - mom["observed record"][k]) ** 2
                                      ).mean())))
        print("   %-28s sigma %.5f   skew %.4f   kurt %.4f"
              % (name, row[0], row[1], row[2]))

    np.savez_compressed(
        ana / "threeway_data.npz", dense_x=dense_x, gauge_x=gauge_x,
        times_s=t, dt=dt,
        **{("%s_%s" % (tags[name], key)): mom[name][k]
           for name, _d, _g in sets
           for k, key in enumerate(("sigma", "skew", "kurt"))},
        **d10,
        recovered_fraction=json.dumps(
            {k: {"rms_ratio": v[0], "projection": v[1]}
             for k, v in recovered.items()}),
    )
    print("\n[3way] threeway_data.npz + threeway_pdf_spectra.npz written "
          "to %s" % ana)


if __name__ == "__main__":
    main()
