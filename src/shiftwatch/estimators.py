"""Label-free accuracy estimators.

All estimators share one contract:

    fit(logits_source_val, labels_source_val)   # labels allowed here, once
    estimate(logits_target) -> float            # labels never allowed here

That split is the whole point of the project, so it is enforced in the base
class rather than by convention.

Estimators implemented:
  mean_confidence      - naive baseline: mean max softmax
  temp_scaled          - temperature-scaled mean confidence
  atc                  - adaptive temperature scaling (Garg et al., 2022)
  doc                  - difference of confidence (Liang et al., 2023)
  cbpe                 - confidence-based performance estimation (NannyML)
  error_predictor      - logistic regression on confidence/entropy features
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import log_softmax, softmax


def _softmax(z: np.ndarray) -> np.ndarray:
    return softmax(z, axis=-1)


def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error, top-label confidence bins."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


class Estimator:
    name = "base"

    def fit(self, source_logits: np.ndarray, source_labels: np.ndarray) -> "Estimator":
        raise NotImplementedError

    def estimate(self, target_logits: np.ndarray) -> float:
        raise NotImplementedError


class MeanConfidence(Estimator):
    """Naive baseline: predicted accuracy = mean top-class probability."""

    name = "mean_confidence"

    def fit(self, source_logits, source_labels):
        return self

    def estimate(self, target_logits):
        return float(_softmax(target_logits).max(axis=1).mean())


class TemperatureScaled(Estimator):
    """Temperature fitted on labelled source validation, then mean confidence."""

    name = "temp_scaled"

    def __init__(self, objective: str = "nll"):
        self.objective = objective
        self.temperature = 1.0

    def _loss(self, log_t: float, logits, labels) -> float:
        t = float(np.exp(log_t))
        scaled = log_softmax(logits / t, axis=-1)
        if self.objective == "nll":
            return float(-scaled[np.arange(len(labels)), labels].mean())
        probs = np.exp(scaled)
        return float(_ece(probs, labels))

    def fit(self, source_logits, source_labels):
        res = minimize_scalar(self._loss, bounds=(-3.0, 3.0), args=(source_logits, source_labels),
                              method="bounded")
        self.temperature = float(np.exp(res.x))
        return self

    def estimate(self, target_logits):
        probs = _softmax(target_logits / self.temperature)
        return float(probs.max(axis=1).mean())


class NLLEstimator(Estimator):
    """Calibrated negative-log-likelihood estimator.

    ATC's self-agreement requires sampling, so under greedy decoding it
    degenerates (argmax is temperature-invariant). This estimator keeps the
    same calibration spirit but uses NLL: fit T on labelled source data so
    that exp(-mean NLL) equals the source accuracy, then estimate target
    accuracy as exp(-mean NLL) under that T.
    """

    name = "nll"

    def fit(self, source_logits, source_labels):
        self.source_accuracy = float((source_logits.argmax(axis=1) == source_labels).mean())

        def gap(log_t):
            t = float(np.exp(log_t))
            log_probs = log_softmax(source_logits / t, axis=-1)
            return float(np.exp(-(-log_probs.max(axis=1)).mean())) - self.source_accuracy

        lo, hi = -4.0, 4.0
        if gap(lo) * gap(hi) > 0:
            self.temperature = 1.0
        else:
            for _ in range(60):
                mid = (lo + hi) / 2
                if gap(lo) * gap(mid) <= 0:
                    hi = mid
                else:
                    lo = mid
            self.temperature = float(np.exp((lo + hi) / 2))
        return self

    def estimate(self, target_logits):
        log_probs = log_softmax(target_logits / self.temperature, axis=-1)
        nll = -log_probs.max(axis=1).mean()
        return float(np.exp(-nll))


class DoC(Estimator):
    """Difference of confidence (Liang et al., 2023).

    accuracy_hat = source_accuracy + mean_conf(source) - mean_conf(target)
    """

    name = "doc"

    def fit(self, source_logits, source_labels):
        self.source_accuracy = float((source_logits.argmax(axis=1) == source_labels).mean())
        self.source_conf = float(_softmax(source_logits).max(axis=1).mean())
        return self

    def estimate(self, target_logits):
        target_conf = float(_softmax(target_logits).max(axis=1).mean())
        return self.source_accuracy + (self.source_conf - target_conf)


class CBPE(Estimator):
    """Confidence-based performance estimation (NannyML-style).

    Estimates per-class recall from calibrated class probabilities and
    averages by the *target* predicted class distribution, so predicted
    accuracy reflects target-side class imbalance.
    """

    name = "cbpe"

    def fit(self, source_logits, source_labels):
        self.temperature_scaler = TemperatureScaled().fit(source_logits, source_labels)
        probs = _softmax(source_logits / self.temperature_scaler.temperature)
        pred = probs.argmax(axis=1)
        n_classes = probs.shape[1]
        tp = np.bincount(pred[pred == source_labels], minlength=n_classes).astype(float)
        n_pred = np.bincount(pred, minlength=n_classes).astype(float)
        self.class_recall = np.where(n_pred > 0, tp / np.maximum(n_pred, 1), 0.0)
        return self

    def estimate(self, target_logits):
        probs = _softmax(target_logits / self.temperature_scaler.temperature)
        pred = probs.argmax(axis=1)
        n_classes = probs.shape[1]
        counts = np.bincount(pred, minlength=n_classes).astype(float)
        total = counts.sum()
        if total == 0:
            return 0.0
        return float((counts / total * self.class_recall).sum())


class ErrorPredictor(Estimator):
    """Small learned error predictor (logistic regression by default).

    Trained on labelled source validation to predict correctness from
    confidence-shaped features, then averaged on target.
    """

    name = "error_predictor"

    FEATURES = ["max_prob", "margin", "neg_entropy", "logit_max_minus_mean", "rank2_prob"]

    def __init__(self, hidden: int = 0):
        self.hidden = hidden

    @staticmethod
    def features(logits: np.ndarray) -> np.ndarray:
        probs = _softmax(logits)
        sorted_p = np.sort(probs, axis=1)
        max_p = sorted_p[:, -1]
        second_p = sorted_p[:, -2] if probs.shape[1] > 1 else np.zeros_like(max_p)
        entropy = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=1)
        return np.column_stack([
            max_p,
            max_p - second_p,
            -entropy,
            logits.max(axis=1) - logits.mean(axis=1),
            second_p,
        ])

    def fit(self, source_logits, source_labels):
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        x = self.features(source_logits)
        y = (source_logits.argmax(axis=1) == source_labels).astype(int)
        if self.hidden:
            model = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(self.hidden,),
                                                                 max_iter=800, random_state=0))
        else:
            model = make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=2000, random_state=0))
        model.fit(x, y)
        self.model = model
        return self

    def estimate(self, target_logits):
        x = self.features(target_logits)
        return float(self.model.predict_proba(x)[:, 1].mean())


ESTIMATORS = {
    cls.name: cls
    for cls in [MeanConfidence, TemperatureScaled, NLLEstimator, DoC, CBPE, ErrorPredictor]
}


def build_all(source_logits, source_labels, target_logits) -> dict[str, float]:
    """Fit every estimator on source, estimate target accuracy for each."""
    out = {}
    for name, cls in ESTIMATORS.items():
        out[name] = cls().fit(source_logits, source_labels).estimate(target_logits)
    out["error_predictor_mlp"] = (ErrorPredictor(hidden=16)
                                  .fit(source_logits, source_labels).estimate(target_logits))
    return out


def true_accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    return float((logits.argmax(axis=1) == labels).mean())