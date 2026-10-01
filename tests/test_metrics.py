import pytest

from shiftwatch.metrics import (SliceResult, alert_decisions, alert_detection_rate,
                                false_alarm_rate, mean_absolute_error,
                                signed_bias, summary_table)


def slice_result(name, true_acc, est_acc, source_acc=0.90):
    return SliceResult(slice_name=name, true_accuracy=true_acc,
                       estimates=est_acc, n=100, source_accuracy=source_acc)


ESTIMATORS = ["a", "b", "c"]


class TestSliceResult:
    def test_true_drop(self):
        r = slice_result("s", 0.70, {"a": 0.75}, source_acc=0.90)
        assert r.true_drop == pytest.approx(0.20)

    def test_errors_abs(self):
        r = slice_result("s", 0.70, {"a": 0.80, "b": 0.60})
        assert r.error("a") == pytest.approx(0.10)
        assert r.error("b") == pytest.approx(0.10)
        assert r.errors == {"a": pytest.approx(0.10), "b": pytest.approx(0.10)}


class TestAverages:
    def setup_method(self):
        self.results = [
            slice_result("s1", 0.80, {"a": 0.82, "b": 0.70}),
            slice_result("s2", 0.60, {"a": 0.55, "b": 0.62}),
        ]

    def test_mae(self):
        assert mean_absolute_error(self.results, "a") == pytest.approx(0.035)
        assert mean_absolute_error(self.results, "b") == pytest.approx(0.06)

    def test_signed_bias(self):
        # a: +0.02, -0.05 -> -0.015
        assert signed_bias(self.results, "a") == pytest.approx(-0.015)

    def test_empty_is_nan(self):
        import math
        assert math.isnan(mean_absolute_error([], "a"))


class TestAlerting:
    def test_true_positive_when_both_drop(self):
        results = [slice_result("drop", 0.70, {"a": 0.70})]
        d = alert_decisions(results, "a", threshold=0.05)
        assert d["tp"] == 1 and d["fn"] == 0

    def test_false_negative_when_drop_missed(self):
        results = [slice_result("drop", 0.70, {"a": 0.89})]
        d = alert_decisions(results, "a", threshold=0.05)
        assert d["fn"] == 1

    def test_false_positive_when_no_drop_but_estimated(self):
        results = [slice_result("clean", 0.90, {"a": 0.80})]
        d = alert_decisions(results, "a", threshold=0.05)
        assert d["fp"] == 1

    def test_true_negative(self):
        results = [slice_result("clean", 0.90, {"a": 0.90})]
        assert alert_decisions(results, "a", threshold=0.05)["tn"] == 1

    def test_detection_rate_none_without_positives(self):
        results = [slice_result("clean", 0.90, {"a": 0.90})]
        assert alert_detection_rate(results, "a") is None

    def test_false_alarm_rate_none_without_negatives(self):
        results = [slice_result("drop", 0.60, {"a": 0.60})]
        assert false_alarm_rate(results, "a") is None

    def test_rates_complement_sensibly(self):
        results = [
            slice_result("d1", 0.60, {"a": 0.60}),   # tp
            slice_result("d2", 0.60, {"a": 0.89}),   # fn
            slice_result("c1", 0.90, {"a": 0.80}),   # fp
            slice_result("c2", 0.90, {"a": 0.91}),   # tn
        ]
        assert alert_detection_rate(results, "a") == pytest.approx(0.5)
        assert false_alarm_rate(results, "a") == pytest.approx(0.5)


class TestSummaryTable:
    def test_sorted_by_mae(self):
        results = [
            slice_result("s1", 0.80, {"good": 0.80, "bad": 0.50}),
            slice_result("s2", 0.60, {"good": 0.61, "bad": 0.90}),
        ]
        rows = summary_table(results, ["good", "bad"])
        assert rows[0]["estimator"] == "good"
        assert rows[0]["mae"] < rows[1]["mae"]

    def test_row_keys(self):
        rows = summary_table([slice_result("s", 0.7, {"a": 0.7})], ["a"])
        assert set(rows[0]) == {"estimator", "mae", "bias", "detection_rate",
                                "false_alarm_rate", "tp", "fp", "tn", "fn"}