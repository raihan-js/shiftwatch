# ShiftWatch

Estimate a deployed classifier's accuracy after a data shift, before any labels arrive.

## The problem

Drift dashboards alert on input statistics that don't track failure. The production question is: **what is my accuracy right now, with no labels?**

## The approach

Benchmark six label-free accuracy estimators on a controlled shift ladder:

| Estimator | Source | Key idea |
|---|---|---|
| Mean confidence | — | Naive: mean max softmax |
| Temperature scaling | Guo et al. 2017 | Fit T on source val, then mean confidence |
| DoC | Liang et al. 2023 | Source accuracy + (source conf − target conf) |
| CBPE | NannyML | Per-class recall from calibrated probs |
| NLL | — | Calibrated: exp(−mean NLL) |
| Error predictor | — | Logistic regression on confidence features |

ATC (Garg et al. 2022) is omitted: its self-agreement requires sampling, which degenerates under greedy decoding.

## The shift ladder

Deterministic, rule-based corruptions (no GPU, no LLM):

- **Typos**: keyboard-adjacent swaps, deletions, duplications (3 severities)
- **Abbreviations**: domain word → abbreviation (3 severities)
- **Style**: lowercasing, punctuation removal, politeness padding, whitespace noise (3 severities)
- **OOS contamination**: replace X% of in-scope texts with out-of-scope ones (10/20/30/40%)

Two datasets: Banking77 (77 classes) and CLINC150 (150 classes + OOS).

## The sidecar

A small FastAPI service that:
- Accepts probability vectors (never labels)
- Keeps a rolling window of predictions
- Republishes estimated accuracy with the winning estimator
- Exports Prometheus gauges and an alert flag

## Results

See `data/results/` for the benchmark table. Key finding: [TBD after run].

## Estimator notes

- **Mean confidence** is optimistic under confident-wrong shifts.
- **Temperature scaling** calibrates on source but may not transfer under shift.
- **DoC** captures confidence degradation but assumes source accuracy is known.
- **CBPE** captures class distribution shift but not confidence degradation.
- **NLL** is similar to temperature scaling but uses log-probabilities.
- **Error predictor** learns from source features but may not transfer under shift.

## Usage

```bash
# Train
PYTHONPATH=src python scripts/train_classifier.py --dataset banking77

# Build ladder
PYTHONPATH=src python scripts/build_ladder.py --dataset banking77

# Benchmark
PYTHONPATH=src python scripts/run_benchmark.py --dataset banking77

# Serve
shiftwatch serve --dataset clinc150 --estimator cbpe
```

## Limitations

- Two datasets, both intent classification — no vision or NLU.
- Shift ladder is rule-based; real-world shifts are messier.
- Estimators assume the model is reasonably calibrated on source.
- Sidecar is a prototype, not production-hardened.

## License

MIT
