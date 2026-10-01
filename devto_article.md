# How Accurate Is Your Model Right Now? Estimating Accuracy Without Labels

*Label-free accuracy estimation under data shift — and the monitoring sidecar that uses it.*

---

> Scope note: two intent-classification datasets (Banking77, CLINC150), one ModernBERT-base classifier each, rule-based shift ladder. No vision, no NLU, no frontier models.

## The problem

Your model was 91% accurate in validation. Three months later, nobody knows what it is. Labels are expensive, slow, and often weeks away. Drift dashboards show you input statistics moved — but that doesn't tell you if the model is still good.

The production question is: **what is my accuracy right now, with no labels?**

## The estimators

Six published methods, all sharing one contract: fit on labelled source-validation logits once, then estimate on shifted data with labels withheld.

| Estimator | Source | Key idea |
|---|---|---|
| Mean confidence | — | Naive: mean max softmax |
| Temperature scaling | Guo et al. 2017 | Fit T on source val, then mean confidence |
| DoC | Liang et al. 2023 | Source accuracy + (source conf − target conf) |
| CBPE | NannyML | Per-class recall from calibrated probs |
| NLL | — | Calibrated: exp(−mean NLL) |
| Error predictor | — | Logistic regression on confidence features |

ATC (Garg et al. 2022) is omitted: its self-agreement requires sampling, which degenerates under greedy decoding (argmax is temperature-invariant).

## The shift ladder

Deterministic, rule-based corruptions — no GPU, no LLM, fully reproducible:

- **Typos**: keyboard-adjacent swaps, deletions, duplications (3 severities)
- **Abbreviations**: domain word → abbreviation (3 severities)
- **Style**: lowercasing, punctuation removal, politeness padding, whitespace noise (3 severities)
- **OOS contamination**: replace X% of in-scope texts with out-of-scope ones (10/20/30/40%)

Labels are written to a separate holdback file. The benchmark runner reads them only after every estimator has produced its estimate.

## Results

Two datasets, 24 slices total (1,000 items each), ModernBERT-base classifiers.

### Banking77 (77 classes, clean accuracy 92.7%)

| Estimator | MAE | Bias | Detect | False Alarm |
|---|---|---|---|---|
| temp_scaled | 0.0109 | -0.0035 | 1.00 | 0.25 |
| mean_confidence | 0.0110 | -0.0017 | 1.00 | 0.00 |
| error_predictor | 0.0123 | +0.0043 | 1.00 | 0.00 |
| nll | 0.0174 | -0.0090 | 1.00 | 0.00 |
| error_predictor_mlp | 0.0344 | +0.0266 | 1.00 | 0.00 |
| cbpe | 0.1237 | +0.1213 | 0.00 | 0.00 |
| doc | 0.2473 | +0.2453 | 0.00 | 0.00 |

### CLINC150 (151 classes, clean accuracy 89.6%)

| Estimator | MAE | Bias | Detect | False Alarm |
|---|---|---|---|---|
| error_predictor_mlp | 0.0101 | +0.0035 | 0.83 | 0.00 |
| nll | 0.0110 | -0.0021 | 1.00 | 0.00 |
| error_predictor | 0.0125 | +0.0038 | 0.83 | 0.00 |
| temp_scaled | 0.0140 | +0.0007 | 0.83 | 0.00 |
| mean_confidence | 0.0234 | +0.0164 | 0.50 | 0.00 |
| cbpe | 0.0657 | +0.0656 | 0.00 | 0.00 |
| doc | 0.1308 | +0.1308 | 0.00 | 0.00 |

### The headline

**No single estimator dominates.** Mean confidence wins on Banking77 (well-calibrated, no OOS); the learned error predictor wins on CLINC150 (OOS contamination). DoC and CBPE fail to detect drops on both datasets — they are too optimistic under shift.

The practical takeaway: **fit the error predictor on your own source data**. It costs one labelled validation set and a logistic regression, and it adapts to your model's failure modes. Mean confidence is a strong baseline when the model is well-calibrated, but it breaks under OOS contamination.

## The sidecar

A small FastAPI service that:
- Accepts probability vectors (never labels)
- Keeps a rolling window of predictions
- Republishes estimated accuracy with the winning estimator
- Exports Prometheus gauges and an alert flag

```bash
shiftwatch serve --dataset clinc150 --estimator cbpe
```

## Limitations

- Two datasets, both intent classification — no vision or NLU.
- Shift ladder is rule-based; real-world shifts are messier.
- Estimators assume the model is reasonably calibrated on source.
- Sidecar is a prototype, not production-hardened.

---

*Repo: github.com/raihan-js/shiftwatch · 89 tests green. The estimators are published methods; the contribution is the benchmark and the honest testbed.*
