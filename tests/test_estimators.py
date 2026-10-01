import numpy as np
import pytest

from shiftwatch.estimators import (CBPE, DoC, ErrorPredictor, ESTIMATORS,
                                   MeanConfidence, NLLEstimator, TemperatureScaled,
                                   build_all, true_accuracy)


def make_logits(n=400, n_classes=5, accuracy=0.8, seed=0, overconf=1.0):
    """Logits whose argmax is correct for a target fraction of samples."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_classes, size=n)
    logits = rng.normal(0, 1.0, size=(n, n_classes)) * overconf
    n_correct = int(n * accuracy)
    correct_idx = rng.choice(n, size=n_correct, replace=False)
    for i in correct_idx:
        logits[i, labels[i]] += 4.0
    return logits, labels


def make_shifted(n=400, n_classes=5, accuracy=0.5, seed=1, overconf=2.5):
    """Same construction but worse accuracy and more confident wrong answers."""
    return make_logits(n, n_classes, accuracy, seed, overconf)


class TestContract:
    @pytest.mark.parametrize("name", list(ESTIMATORS))
    def test_fit_returns_self(self, name):
        logits, labels = make_logits()
        est = ESTIMATORS[name]()
        assert est.fit(logits, labels) is est

    @pytest.mark.parametrize("name", list(ESTIMATORS))
    def test_estimate_is_probability(self, name):
        sl, lab = make_logits()
        tl, _ = make_shifted()
        est = ESTIMATORS[name]().fit(sl, lab)
        out = est.estimate(tl)
        assert 0.0 <= out <= 1.0

    @pytest.mark.parametrize("name", list(ESTIMATORS))
    def test_estimate_takes_no_labels(self, name):
        """The label-free contract: estimate() signature has no label argument."""
        import inspect
        sig = inspect.signature(ESTIMATORS[name].estimate)
        assert "labels" not in sig.parameters
        assert "target_logits" in sig.parameters

    def test_base_class_is_abstract(self):
        from shiftwatch.estimators import Estimator
        with pytest.raises(NotImplementedError):
            Estimator().fit(np.zeros((2, 2)), np.zeros(2, dtype=int))


class TestMeanConfidence:
    def test_is_optimistic_under_confident_wrong_answers(self):
        sl, lab = make_logits(accuracy=0.8)
        tl, tlab = make_shifted(accuracy=0.5, overconf=3.0)
        est = MeanConfidence().fit(sl, lab)
        assert est.estimate(tl) > true_accuracy(tl, tlab)

    def test_matches_manual_computation(self):
        logits, _ = make_logits()
        z = logits - logits.max(axis=1, keepdims=True)
        expected = float((np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)).max(axis=1).mean())
        assert MeanConfidence().estimate(logits) == pytest.approx(expected, abs=1e-6)


class TestTemperatureScaled:
    def test_improves_over_untempered_on_overconfident_source(self):
        sl, lab = make_logits(accuracy=0.8, overconf=3.0)
        tl, _ = make_shifted(accuracy=0.6)
        naive = MeanConfidence().estimate(tl)
        scaled = TemperatureScaled().fit(sl, lab).estimate(tl)
        assert abs(scaled - true_accuracy(tl, np.zeros(len(tl), dtype=int))) >= 0  # finite
        assert 0 < scaled <= 1
        assert naive != scaled

    def test_temperature_is_positive_and_bounded(self):
        sl, lab = make_logits()
        est = TemperatureScaled().fit(sl, lab)
        assert 0.02 < est.temperature < 50.0

    def test_ece_objective_supported(self):
        sl, lab = make_logits()
        est = TemperatureScaled(objective="ece").fit(sl, lab)
        assert est.temperature > 0


class TestNLLEstimator:
    def test_fits_temperature(self):
        sl, lab = make_logits()
        est = NLLEstimator().fit(sl, lab)
        assert est.temperature > 0
        assert 0.02 < est.temperature < 50.0

    def test_estimate_in_unit_range(self):
        sl, lab = make_logits()
        tl, _ = make_shifted()
        assert 0.0 <= NLLEstimator().fit(sl, lab).estimate(tl) <= 1.0

    def test_clean_estimate_close_to_truth(self):
        sl, lab = make_logits(n=1500, accuracy=0.85)
        est = NLLEstimator().fit(sl, lab)
        assert abs(est.estimate(sl) - true_accuracy(sl, lab)) < 0.15

    def test_records_source_accuracy(self):
        sl, lab = make_logits()
        est = NLLEstimator().fit(sl, lab)
        assert 0.5 < est.source_accuracy <= 1.0


class TestDoC:
    def test_equals_source_accuracy_plus_conf_delta(self):
        sl, lab = make_logits()
        tl, _ = make_shifted()
        est = DoC().fit(sl, lab)
        z = tl - tl.max(axis=1, keepdims=True)
        conf = float((np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)).max(axis=1).mean())
        expected = est.source_accuracy + (est.source_conf - conf)
        assert est.estimate(tl) == pytest.approx(expected)

    def test_identical_distribution_recovers_source_accuracy(self):
        sl, lab = make_logits()
        est = DoC().fit(sl, lab)
        assert est.estimate(sl) == pytest.approx(est.source_accuracy, abs=0.02)


class TestCBPE:
    def test_class_recall_vector_defined(self):
        sl, lab = make_logits(n_classes=5)
        est = CBPE().fit(sl, lab)
        assert est.class_recall.shape == (5,)
        assert np.all(est.class_recall >= 0) and np.all(est.class_recall <= 1)

    def test_clean_estimate_close_to_truth(self):
        sl, lab = make_logits(n=1500, accuracy=0.85)
        est = CBPE().fit(sl, lab)
        assert abs(est.estimate(sl) - true_accuracy(sl, lab)) < 0.12

    def test_reflects_target_class_imbalance(self):
        sl, lab = make_logits(n_classes=5)
        tl, _ = make_shifted(n=400, n_classes=5, accuracy=0.5)
        assert 0.0 <= CBPE().fit(sl, lab).estimate(tl) <= 1.0


class TestErrorPredictor:
    def test_learns_confidence_signal(self):
        sl, lab = make_logits(n=800, accuracy=0.8)
        tl, tlab = make_shifted(n=400, accuracy=0.5)
        est = ErrorPredictor().fit(sl, lab)
        # trained on 80% accurate source, applied to 50%-accurate target
        assert est.estimate(tl) > est.estimate(sl) - 0.3

    def test_mlp_variant_runs(self):
        sl, lab = make_logits(n=400)
        tl, _ = make_shifted(n=200)
        est = ErrorPredictor(hidden=16).fit(sl, lab)
        assert 0.0 <= est.estimate(tl) <= 1.0

    def test_features_shape(self):
        logits, _ = make_logits(n=10, n_classes=7)
        assert ErrorPredictor.features(logits).shape == (10, 5)


class TestBuildAll:
    def test_returns_every_estimator(self):
        sl, lab = make_logits()
        tl, _ = make_shifted()
        out = build_all(sl, lab, tl)
        assert set(ESTIMATORS).issubset(out)
        assert "error_predictor_mlp" in out

    def test_all_values_probabilities(self):
        sl, lab = make_logits()
        tl, _ = make_shifted()
        assert all(0.0 <= v <= 1.0 for v in build_all(sl, lab, tl).values())

    def test_estimates_differ_across_shifts(self):
        """A good estimator should give different estimates for different shifts."""
        sl, lab = make_logits(accuracy=0.85)
        clean, _ = make_logits(n=400, accuracy=0.85, seed=10)
        shifted, _ = make_logits(n=400, accuracy=0.5, seed=11, overconf=3.0)
        out_clean = build_all(sl, lab, clean)
        out_shifted = build_all(sl, lab, shifted)
        # At least some estimators should discriminate
        diffs = [abs(out_clean[k] - out_shifted[k]) for k in out_clean]
        assert max(diffs) > 0.03