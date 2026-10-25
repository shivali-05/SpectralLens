"""Feature extraction modules for SpectralLens."""

from spectrallens.features.spectral import (
    FEATURE_NAMES as SPECTRAL_FEATURE_NAMES,
)
from spectrallens.features.spectral import (
    N_SPECTRAL_FEATURES,
    extract_features,
    extract_hybrid_features,
    extract_spectral_features,
)
from spectrallens.features.temporal import (
    FEATURE_NAMES as TEMPORAL_FEATURE_NAMES,
)
from spectrallens.features.temporal import (
    N_TEMPORAL_FEATURES,
    extract_temporal_features,
)

FEATURE_SET_VERSION = "v1"
HYBRID_FEATURE_NAMES = TEMPORAL_FEATURE_NAMES + SPECTRAL_FEATURE_NAMES
N_HYBRID_FEATURES = len(HYBRID_FEATURE_NAMES)

__all__ = [
    "FEATURE_SET_VERSION",
    "HYBRID_FEATURE_NAMES",
    "N_HYBRID_FEATURES",
    "N_SPECTRAL_FEATURES",
    "N_TEMPORAL_FEATURES",
    "SPECTRAL_FEATURE_NAMES",
    "TEMPORAL_FEATURE_NAMES",
    "extract_features",
    "extract_hybrid_features",
    "extract_spectral_features",
    "extract_temporal_features",
]
