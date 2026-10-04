![shiftwatch results](https://raw.githubusercontent.com/raihan-js/shiftwatch/HEAD/images/shiftwatch.png)

# How Accurate Is Your Model Right Now? Estimating Accuracy Without Labels

*Label-free accuracy estimation under data shift — and the monitoring sidecar that uses it.*

---

> Scope note: two intent-classification datasets (Banking77, CLINC150), one ModernBERT-base classifier each, rule-based shift ladder. No vision, no NLU, no frontier models.

## The problem

Your model was 91% accurate in validation. Three months later, nobody knows what it is. Labels are expensive, slow, and often weeks away. Drift dashboards show you input statistics moved — but that doesn't tell you if the model is still good.

The production question is: **what is my accuracy right now, with no labels?**

## The estimators

Six published methods (plus an MLP variant of the error predictor, which is why the result tables have seven rows), all sharing one contract: fit on labelled source-validation logits once, then estimate on shifted data with labels withheld.

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

**Reading the columns.** MAE is the mean absolute gap between estimated and true accuracy over the 24 slices. *Detect* is the share of slices with a true accuracy drop of at least 5 points that the estimator also flagged (estimated drop of at least 5 points). *False Alarm* is the share of the other slices it flagged anyway. Both use the 5-point threshold the sidecar defaults to. These rates rest on small counts: in each dataset 6 slices have a true drop of at least 5 points, so a detect rate of 0.83 is 5 of 6, and Banking77's false-alarm rate of 0.25 is 1 of 4 non-drop slices.

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

**No single estimator dominates.** On Banking77 temperature scaling and mean confidence are tied (MAE 0.0109 vs 0.0110, within noise of each other); on CLINC150, where out-of-scope contamination breaks calibration, the MLP error predictor leads (0.0101, with NLL at 0.0110). DoC and CBPE fail to detect drops on both datasets: they are too optimistic under shift.

The practical takeaway: **fit the error predictor on your own source data**. It costs one labelled validation set and a logistic regression, it is in the top three on both datasets with no false alarms, and it adapts to your model's failure modes. Mean confidence is a strong baseline when the model is well-calibrated, but it degrades under OOS contamination (MAE 0.0234 on CLINC150, detecting half the drops).

## The sidecar

A small FastAPI service that:
- Accepts probability vectors (never labels)
- Keeps a rolling window of predictions
- Republishes estimated accuracy with the winning estimator
- Exports Prometheus gauges and an alert flag

```bash
shiftwatch serve --dataset clinc150 --estimator error_predictor
```

`error_predictor` is the sidecar's default. (It used to default to `cbpe`, which never detected a drop in this benchmark; the benchmark is what caught that.)

## Limitations

- Two datasets, both intent classification — no vision or NLU.
- Shift ladder is rule-based; real-world shifts are messier.
- Estimators assume the model is reasonably calibrated on source.
- Sidecar is a prototype, not production-hardened.

---

*Repo: github.com/raihan-js/shiftwatch · 96 tests green. The estimators are published methods; the contribution is the benchmark and the honest testbed.*
