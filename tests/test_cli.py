import numpy as np
import pytest

from shiftwatch.cli import DEFAULT_ESTIMATOR, build_parser
from shiftwatch.estimators import ESTIMATORS


def test_sidecar_defaults_to_nll():
    # NLL is the only estimator that detected every >=5-point drop (6/6 on both datasets)
    # with zero false alarms on both; see the README results tables.
    assert DEFAULT_ESTIMATOR == "nll"
    for argv in (["serve"], ["replay"]):
        assert build_parser().parse_args(argv).estimator == "nll"


def test_estimator_flag_overrides_default():
    assert build_parser().parse_args(["serve", "--estimator", "error_predictor"]).estimator == "error_predictor"


def test_default_estimator_is_registered_and_recovers_source_accuracy():
    # NLL calibrates its temperature so that, on the source data, the estimate equals the
    # source accuracy. (A source with 100% accuracy has no such temperature and falls back to T=1.)
    assert DEFAULT_ESTIMATOR in ESTIMATORS
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(2000, 5)) * 2
    labels = logits.argmax(axis=1).copy()
    noisy = rng.random(2000) < 0.15
    labels[noisy] = rng.integers(0, 5, noisy.sum())
    accuracy = float((logits.argmax(axis=1) == labels).mean())
    est = ESTIMATORS[DEFAULT_ESTIMATOR]().fit(logits, labels)
    assert est.estimate(logits) == pytest.approx(accuracy, abs=0.01)
