import numpy as np
import pytest

from shiftwatch.estimators import MeanConfidence, TemperatureScaled
from shiftwatch.sidecar import RollingEstimate, build_gauges, create_app, render_metrics, update_gauges


def make_probs(n, n_classes=4, acc=0.75, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        p = rng.dirichlet(np.ones(n_classes) * 0.4)
        if i % 4 < acc * 4:
            p = p * 0.3
            p[i % 4] += 0.7
        rows.append(p / p.sum())
    return np.stack(rows)


class TestRollingEstimate:
    def test_counts_observations(self):
        r = RollingEstimate(window=10)
        for p in make_probs(5):
            r.push(p)
        assert r.n_seen == 5 and len(r.probs) == 5

    def test_window_slides(self):
        r = RollingEstimate(window=3)
        for p in make_probs(10):
            r.push(p)
        assert len(r.probs) == 3
        assert r.n_seen == 10

    def test_estimate_none_before_push(self):
        r = RollingEstimate(estimator=MeanConfidence())
        assert r.estimated_accuracy() is None

    def test_estimate_none_without_estimator(self):
        r = RollingEstimate()
        for p in make_probs(5):
            r.push(p)
        assert r.estimated_accuracy() is None

    def test_estimate_with_fitted_estimator(self):
        probs = make_probs(100)
        logits = np.log(probs)
        labels = probs.argmax(axis=1)
        est = TemperatureScaled().fit(logits, labels)
        r = RollingEstimate(window=100, estimator=est, source_accuracy=0.9)
        for p in probs:
            r.push(p)
        assert 0.0 <= r.estimated_accuracy() <= 1.0

    def test_alerting_flag(self):
        probs = make_probs(50, acc=0.9)
        est = MeanConfidence()
        r = RollingEstimate(window=50, estimator=est, source_accuracy=0.99, threshold=0.05)
        for p in probs:
            r.push(p)
        assert r.alerting() is True

    def test_no_alert_when_drop_small(self):
        probs = np.tile([0.97, 0.01, 0.01, 0.01], (50, 1))
        r = RollingEstimate(window=50, estimator=MeanConfidence(),
                            source_accuracy=0.995, threshold=0.05)
        for p in probs:
            r.push(p)
        assert r.alerting() is False

    def test_rejects_non_vector(self):
        with pytest.raises(ValueError):
            RollingEstimate().push(np.ones((2, 2)))


class TestPrometheus:
    def test_metrics_render(self):
        probs = make_probs(30)
        r = RollingEstimate(window=30, estimator=MeanConfidence(), source_accuracy=0.9)
        for p in probs:
            r.push(p)
        out = render_metrics(r).decode()
        assert "shiftwatch_estimated_accuracy" in out
        assert "shiftwatch_predictions_seen" in out
        assert "shiftwatch_alerting" in out

    def test_gauges_updated(self):
        from prometheus_client import CollectorRegistry
        r = RollingEstimate(window=10, estimator=MeanConfidence(), source_accuracy=0.9)
        for p in make_probs(10):
            r.push(p)
        registry = CollectorRegistry()
        gauges = build_gauges(registry)
        update_gauges(gauges, r)
        assert gauges["seen"]._value.get() == 10
        assert gauges["window"]._value.get() == 10


class TestFastAPI:
    def setup_method(self):
        from fastapi.testclient import TestClient
        self.r = RollingEstimate(window=100, estimator=MeanConfidence(),
                                 source_accuracy=0.9)
        self.client = TestClient(create_app(self.r))

    def test_observe_accepts_probs(self):
        resp = self.client.post("/observe", json={"probs": [0.7, 0.2, 0.1]})
        assert resp.status_code == 200
        assert resp.json()["n_seen"] == 1

    def test_observe_rejects_empty(self):
        assert self.client.post("/observe", json={"probs": []}).status_code == 422

    def test_observe_rejects_zero_sum(self):
        assert self.client.post("/observe", json={"probs": [0.0, 0.0]}).status_code == 422

    def test_estimate_endpoint(self):
        for p in make_probs(10):
            self.client.post("/observe", json={"probs": p.tolist()})
        body = self.client.get("/estimate").json()
        assert body["n_seen"] == 10
        assert 0.0 <= body["estimated_accuracy"] <= 1.0
        assert isinstance(body["alerting"], bool)

    def test_metrics_endpoint(self):
        for p in make_probs(5):
            self.client.post("/observe", json={"probs": p.tolist()})
        assert "shiftwatch_estimated_accuracy" in self.client.get("/metrics").text

    def test_no_label_field_accepted(self):
        """The sidecar has no route that takes a label: that is the design."""
        routes = {r.path for r in create_app(self.r).routes}
        assert routes == {"/observe", "/estimate", "/metrics", "/openapi.json",
                          "/docs", "/docs/oauth2-redirect", "/redoc"}