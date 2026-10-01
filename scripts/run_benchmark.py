#!/usr/bin/env python3
"""Benchmark label-free accuracy estimators against true accuracy.

Protocol, in order:
  1. Fit every estimator on clean source-validation logits (labels allowed).
  2. For each shifted slice, run the classifier to get logits, then estimate.
  3. Only then read the holdback labels and compute true accuracy.

Step 3 is enforced by writing estimates to disk before holdback is opened.

Usage:
  PYTHONPATH=src python scripts/run_benchmark.py --dataset banking77
  PYTHONPATH=src python scripts/run_benchmark.py --dataset clinc150
"""
import argparse
import json
from pathlib import Path

import numpy as np

from shiftwatch.estimators import ESTIMATORS, ErrorPredictor, build_all, true_accuracy
from shiftwatch.metrics import SliceResult, summary_table

RESULTS = Path("data/results")


def logits_for(model, tok, texts: list[str], batch_size: int, device) -> np.ndarray:
    import torch
    out = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            enc = tok(chunk, truncation=True, max_length=64, padding=True,
                      return_tensors="pt").to(device)
            out.append(model(**enc).logits.float().cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["banking77", "clinc150"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--threshold", type=float, default=0.05)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    src = np.load(Path("data/source") / f"{args.dataset}_clean.npz")
    source_logits, source_labels = src["logits"], src["labels"]
    source_acc = true_accuracy(source_logits, source_labels)
    print(f"source (clean val) accuracy: {source_acc:.4f}", flush=True)

    model_dir = Path("data/models") / f"modernbert-{args.dataset}"
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device).eval()

    ladder_files = sorted(Path("data/ladder").glob(f"{args.dataset}_*.jsonl"))
    print(f"{len(ladder_files)} slices", flush=True)

    # --- step 1+2: estimate every slice, labels untouched ---
    estimates_by_slice = {}
    for path in ladder_files:
        slice_id = path.stem[len(args.dataset) + 1:]
        rows = [json.loads(l) for l in open(path)]
        logits = logits_for(model, tok, [r["text"] for r in rows], args.batch_size, device)
        est = build_all(source_logits, source_labels, logits)
        estimates_by_slice[slice_id] = est
        print(f"  {slice_id:10s} " + "  ".join(f"{k}={v:.3f}" for k, v in list(est.items())[:3]),
              flush=True)
        np.savez(RESULTS / f"{args.dataset}_{slice_id}_logits.npz", logits=logits)

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{args.dataset}_estimates.json").write_text(json.dumps(estimates_by_slice, indent=2))
    print("estimates written before any label was read", flush=True)

    # --- step 3: now the holdback ---
    results = []
    for path in ladder_files:
        slice_id = path.stem[len(args.dataset) + 1:]
        hold = [json.loads(l) for l in
                open(Path("data/holdback") / f"{args.dataset}_{slice_id}.jsonl")]
        logits = np.load(RESULTS / f"{args.dataset}_{slice_id}_logits.npz")["logits"]
        label_to_id = model.config.label2id
        gold = np.array([label_to_id[h["label"]] for h in hold])
        results.append(SliceResult(
            slice_name=slice_id,
            true_accuracy=true_accuracy(logits, gold),
            estimates=estimates_by_slice[slice_id],
            n=len(gold),
            source_accuracy=source_acc,
        ))

    estimator_names = list(ESTIMATORS) + ["error_predictor_mlp"]
    table = summary_table(results, estimator_names, threshold=args.threshold)

    print(f"\n{'slice':12s} {'n':>5s} {'true':>7s} {'drop':>7s} " +
          "".join(f"{e[:11]:>12s}" for e in estimator_names))
    for r in results:
        print(f"{r.slice_name:12s} {r.n:5d} {r.true_accuracy:7.3f} {r.true_drop:7.3f} " +
              "".join(f"{r.estimates[e]:12.3f}" for e in estimator_names))

    print(f"\n=== estimator ranking ({args.dataset}, {len(results)} slices) ===")
    print(f"{'estimator':22s} {'MAE':>7s} {'bias':>7s} {'detect':>7s} {'falseAlarm':>11s}")
    for row in table:
        d = "n/a" if row["detection_rate"] is None else f"{row['detection_rate']:.2f}"
        f = "n/a" if row["false_alarm_rate"] is None else f"{row['false_alarm_rate']:.2f}"
        print(f"{row['estimator']:22s} {row['mae']:7.4f} {row['bias']:+7.4f} {d:>7s} {f:>11s}")

    (RESULTS / f"{args.dataset}_benchmark.json").write_text(json.dumps({
        "dataset": args.dataset,
        "source_accuracy": source_acc,
        "threshold": args.threshold,
        "slices": [{"slice": r.slice_name, "n": r.n, "true_accuracy": r.true_accuracy,
                    "true_drop": r.true_drop, "estimates": r.estimates} for r in results],
        "ranking": table,
    }, indent=2))
    print(f"\nsaved {RESULTS / f'{args.dataset}_benchmark.json'}")


if __name__ == "__main__":
    main()