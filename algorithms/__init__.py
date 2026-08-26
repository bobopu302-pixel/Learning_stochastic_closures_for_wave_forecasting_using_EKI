"""Shared algorithm layer for the dissertation cases (Lorenz, linear wave, vKdV).
"""

from .eki import (
    EKIResult,
    clip_latent,
    crn_seed,
    crn_seeds,
    run_eki,
)
from .gamma import (
    CalibrationResult,
    GammaEstimate,
    build_gamma,
    calibrate_n_g,
)
from .gpr import (
    make_gp_mean,
    make_gp_mean_from_theta,
    product_rbf_kernel,
)
from .parameterization import (
    decode_positive_columns,
    encode_positive_columns,
    gp_positive_indices,
    log_decode,
    log_encode,
)
from .statistics import (
    band_energy_spectrum,
    centered_first_second_moments,
    cov_from_samples,
    cross_corr,
    demeaned_acf,
    first_second_moments,
    gauge_acf,
    histogram_density,
    normalized_autocorr,
    raw_first_second_moments,
    xcorr_pair,
)

__all__ = [
    # eki
    "EKIResult",
    "run_eki",
    "crn_seed",
    "crn_seeds",
    "clip_latent",
    # gamma
    "GammaEstimate",
    "build_gamma",
    "CalibrationResult",
    "calibrate_n_g",
    # gpr
    "make_gp_mean",
    "make_gp_mean_from_theta",
    "product_rbf_kernel",
    # parameterization
    "log_encode",
    "log_decode",
    "encode_positive_columns",
    "decode_positive_columns",
    "gp_positive_indices",
    # statistics
    "normalized_autocorr",
    "gauge_acf",
    "xcorr_pair",
    "cross_corr",
    "band_energy_spectrum",
    "demeaned_acf",
    "centered_first_second_moments",
    "raw_first_second_moments",
    "first_second_moments",
    "cov_from_samples",
    "histogram_density",
]
