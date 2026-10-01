"""How estimators are scored: accuracy of the accuracy estimate, and alerting."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SliceResult:
    """One shifted slice: what every estimator said vs what was true."""

    slice_name: str
    true_accuracy: float
    estimates: dict[str, float]
    n: int
    source_accuracy: float
    alerts: dict[str, bool] = field(default_factory=dict)

    @property
    def true_drop(self) -> float:
        return self.source_accuracy - self.true_accuracy

    @property
    def errors(self) -> dict[str, float]:
        return {k: abs(v - self.true_accuracy) for k, v in self.estimates.items()}

    def error(self, estimator: str) -> float:
        return abs(self.estimates[estimator] - self.true_accuracy)


def mean_absolute_error(results: list[SliceResult], estimator: str) -> float:
    if not results:
        return float("nan")
    return float(np.mean([r.error(estimator) for r in results]))


def signed_bias(results: list[SliceResult], estimator: str) -> float:
    """Mean signed error. Positive = the estimator is optimistic."""
    if not results:
        return float("nan")
    return float(np.mean([r.estimates[estimator] - r.true_accuracy for r in results]))


def alert_decisions(results: list[SliceResult], estimator: str,
                    threshold: float = 0.05) -> dict[str, int]:
    """Confusion of would-alert vs should-alert at a drop threshold.

    should_alert: true drop >= threshold.
    would_alert:  estimated drop >= threshold, where the estimated drop is
    source accuracy minus the estimate.
    """
    tp = fp = tn = fn = 0
    for r in results:
        should = r.true_drop >= threshold
        would = (r.source_accuracy - r.estimates[estimator]) >= threshold
        if should and would:
            tp += 1
        elif should and not would:
            fn += 1
        elif not should and would:
            fp += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def alert_detection_rate(results: list[SliceResult], estimator: str,
                         threshold: float = 0.05) -> float | None:
    d = alert_decisions(results, estimator, threshold)
    n_should = d["tp"] + d["fn"]
    return None if n_should == 0 else d["tp"] / n_should


def false_alarm_rate(results: list[SliceResult], estimator: str,
                     threshold: float = 0.05) -> float | None:
    d = alert_decisions(results, estimator, threshold)
    n_no_alert = d["fp"] + d["tn"]
    return None if n_no_alert == 0 else d["fp"] / n_no_alert


def summary_table(results: list[SliceResult], estimators: list[str],
                  threshold: float = 0.05) -> list[dict]:
    rows = []
    for est in estimators:
        detect = alert_detection_rate(results, est, threshold)
        fa = false_alarm_rate(results, est, threshold)
        rows.append({
            "estimator": est,
            "mae": round(mean_absolute_error(results, est), 4),
            "bias": round(signed_bias(results, est), 4),
            "detection_rate": None if detect is None else round(detect, 3),
            "false_alarm_rate": None if fa is None else round(fa, 3),
            **{k: v for k, v in alert_decisions(results, est, threshold).items()},
        })
    return sorted(rows, key=lambda r: r["mae"])