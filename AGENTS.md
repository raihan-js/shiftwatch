# AGENTS.md — shiftwatch

Portfolio project for Raihan Sikder. Target roles: Noeon Research (Senior ML Engineer, LLMOps), PayPay Card, Money Forward, Treasure AI, Citadel AI.

## Project: ShiftWatch

Estimate a deployed classifier's accuracy after a data shift, before any labels arrive. Benchmark label-free accuracy estimators (mean confidence, temperature scaling, DoC, CBPE, NLL, learned error predictor) on a controlled shift ladder, and ship the best as a monitoring sidecar.

## Why this project

Drift dashboards alert on input statistics that don't track failure. The production question is "what is my accuracy right now, with no labels?" The estimators are published methods (ATC/Garg 2022, DoC/Liang 2023, CBPE/NannyML), so the contribution is the benchmark and the honest testbed.

## Stack

Python, PyTorch, Hugging Face Transformers, ModernBERT-base, scikit-learn, FastAPI, Prometheus client.

## Compute

One RTX 3060 12GB. Fine-tuning ModernBERT-base on Banking77 and CLINC150 takes ~10 min each. No rented GPU needed.

## Datasets

- PolyAI/banking77 (77 intent classes)
- clinc/clinc_oos_plus (150 intent classes + OOS)

## Shift ladder

Deterministic, rule-based corruptions (no GPU, no LLM):
- Typos: keyboard-adjacent swaps, deletions, duplications (severity 1/2/3)
- Abbreviations: domain word → abbreviation (severity 1/2/3)
- Style: lowercasing, punctuation removal, politeness padding, whitespace noise (severity 1/2/3)
- OOS contamination: replace X% of in-scope texts with out-of-scope ones (10/20/30/40%)

## Estimators

| Estimator | Source | Key idea |
|---|---|---|
| MeanConfidence | — | Naive: mean max softmax |
| TemperatureScaled | Guo et al. 2017 | Fit T on source val, then mean confidence |
| DoC | Liang et al. 2023 | Source accuracy + (source conf − target conf) |
| CBPE | NannyML | Per-class recall from calibrated probs |
| NLLEstimator | — | Calibrated NLL: exp(−mean NLL) |
| ErrorPredictor | — | Logistic regression on confidence features |

ATC (Garg et al. 2022) is omitted: its self-agreement requires sampling, which degenerates under greedy decoding (argmax is temperature-invariant).

## Metric

MAE between estimated and true accuracy across all shifted slices. Alert detection rate and false alarm rate at a 5-point-drop threshold.

## Milestones

1. **Public shift ladder** (5d) — Banking77 and CLINC150 classifiers + shift ladder with held-back labels.
2. **Estimator benchmark** (6d) — All estimators, MAE table, alert analysis.
3. **Sidecar and release** (5d) — FastAPI + Prometheus, HF dataset, write-up.

## Conventions

- Python 3.10+, pytest for all estimators and metrics.
- Labels are written to a separate holdback file; the benchmark runner reads them only after all estimates are produced.
- The sidecar never accepts labels — that is the design.
- Every result cites the exact model checkpoint and dataset revision.

## Development

```bash
cd shiftwatch
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,gpu]"
pytest tests/ -v

# Train classifiers
PYTHONPATH=src python scripts/train_classifier.py --dataset banking77
PYTHONPATH=src python scripts/train_classifier.py --dataset clinc150

# Build ladder
PYTHONPATH=src python scripts/build_ladder.py --dataset banking77
PYTHONPATH=src python scripts/build_ladder.py --dataset clinc150

# Run benchmark
PYTHONPATH=src python scripts/run_benchmark.py --dataset banking77
PYTHONPATH=src python scripts/run_benchmark.py --dataset clinc150

# Serve sidecar
shiftwatch serve --dataset clinc150 --estimator cbpe
```

## Current status

- 89 tests passing
- Shift ladder: 9 lexical/style shifts + 4 OOS ratios per dataset
- Estimators: 6 implemented (mean confidence, temp scaling, DoC, CBPE, NLL, error predictor)
- Sidecar: FastAPI + Prometheus, rolling window, alert flag
- Pending: model training, benchmark run, HF dataset, write-up
