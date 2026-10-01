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

[TBD after benchmark run]

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
