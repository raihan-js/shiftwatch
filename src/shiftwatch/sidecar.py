"""Monitoring sidecar: log probabilities in, rolling estimated accuracy out.

This is the deployable half of ShiftWatch. It never sees labels. It keeps a
rolling window of top-class probabilities and republishes the estimated
accuracy with the default estimator (NLL), plus Prometheus gauges and
an alert flag when the estimated drop crosses a threshold.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from prometheus_client import CollectorRegistry, Gauge, generate_latest
from pydantic import BaseModel, Field

from shiftwatch.estimators import Estimator


class Observation(BaseModel):
    probs: list[float] = Field(..., min_length=2)


@dataclass
class RollingEstimate:
    """Rolling window of predictions with a label-free accuracy estimate."""

    window: int = 500
    estimator: Estimator | None = None
    threshold: float = 0.05
    source_accuracy: float | None = None
    probs: deque = field(default_factory=deque)
    n_seen: int = 0

    def push(self, probs: np.ndarray | list[float]) -> None:
        """Record one prediction's full probability vector (unlabelled)."""
        p = np.asarray(probs, dtype=float)
        if p.ndim != 1:
            raise ValueError("expected a single probability vector")
        self.probs.append(p)
        self.n_seen += 1
        while len(self.probs) > self.window:
            self.probs.popleft()

    def estimated_accuracy(self) -> float | None:
        if self.estimator is None or not self.probs:
            return None
        logits = np.log(np.clip(np.stack(self.probs), 1e-12, None))
        return self.estimator.estimate(logits)

    def estimated_drop(self) -> float | None:
        est = self.estimated_accuracy()
        if est is None or self.source_accuracy is None:
            return None
        return self.source_accuracy - est

    def alerting(self) -> bool:
        drop = self.estimated_drop()
        return bool(drop is not None and drop >= self.threshold)


def build_gauges(registry: CollectorRegistry) -> dict[str, Gauge]:
    return {
        "accuracy": Gauge("shiftwatch_estimated_accuracy",
                          "Label-free estimated accuracy", registry=registry),
        "drop": Gauge("shiftwatch_estimated_drop",
                      "Estimated accuracy drop vs source", registry=registry),
        "alerting": Gauge("shiftwatch_alerting",
                          "1 when estimated drop exceeds threshold", registry=registry),
        "seen": Gauge("shiftwatch_predictions_seen",
                      "Predictions observed since start", registry=registry),
        "window": Gauge("shiftwatch_window_filled",
                        "Predictions currently in the window", registry=registry),
    }


def update_gauges(gauges: dict[str, Gauge], rolling: RollingEstimate) -> None:
    est = rolling.estimated_accuracy()
    drop = rolling.estimated_drop()
    if est is not None:
        gauges["accuracy"].set(est)
    if drop is not None:
        gauges["drop"].set(drop)
    gauges["alerting"].set(1.0 if rolling.alerting() else 0.0)
    gauges["seen"].set(rolling.n_seen)
    gauges["window"].set(len(rolling.probs))


def render_metrics(rolling: RollingEstimate) -> bytes:
    registry = CollectorRegistry()
    update_gauges(build_gauges(registry), rolling)
    return generate_latest(registry)


def create_app(rolling: RollingEstimate):
    """FastAPI app around a RollingEstimate. Labels are never accepted."""
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="shiftwatch-sidecar", version="0.1.0")

    @app.post("/observe")
    def observe(obs: Observation) -> dict:
        total = sum(obs.probs)
        if total <= 0:
            raise HTTPException(422, "probabilities must sum to a positive number")
        rolling.push(np.asarray(obs.probs, dtype=float) / total)
        return {"n_seen": rolling.n_seen, "window": len(rolling.probs)}

    @app.get("/estimate")
    def estimate() -> dict:
        est = rolling.estimated_accuracy()
        return {
            "estimated_accuracy": est,
            "estimated_drop": rolling.estimated_drop(),
            "source_accuracy": rolling.source_accuracy,
            "alerting": rolling.alerting(),
            "n_seen": rolling.n_seen,
            "window_filled": len(rolling.probs),
        }

    @app.get("/metrics")
    def metrics() -> bytes:
        return render_metrics(rolling)

    return app