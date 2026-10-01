"""ShiftWatch: estimate a deployed classifier's accuracy before labels arrive."""

from shiftwatch.estimators import (CBPE, DoC, ErrorPredictor, ESTIMATORS,
                                   MeanConfidence, NLLEstimator, TemperatureScaled,
                                   build_all, true_accuracy)
from shiftwatch.metrics import SliceResult, summary_table
from shiftwatch.sidecar import RollingEstimate

__version__ = "0.1.0"
__all__ = [
    "CBPE", "DoC", "ErrorPredictor", "ESTIMATORS", "MeanConfidence",
    "NLLEstimator", "TemperatureScaled", "build_all", "true_accuracy", "SliceResult",
    "summary_table", "RollingEstimate",
]