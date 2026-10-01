#!/usr/bin/env python3
"""Build the shift ladder. Labels are written to a separate holdback file.

Split discipline: data/ladder/<dataset>_<slice>.jsonl holds only the shifted
text plus the slice id. data/holdback/<dataset>_<slice>.jsonl holds the gold
label per row. The benchmark runner reads holdback only after every estimator
has produced its estimate, and the label-free sidecar never sees it at all.

Usage:
  PYTHONPATH=src python scripts/build_ladder.py --dataset banking77
  PYTHONPATH=src python scripts/build_ladder.py --dataset clinc150
"""
import argparse
import json
from pathlib import Path

from shiftwatch.shifts import SHIFTS, apply_shift, mix_out_of_scope

SEED = 17


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["banking77", "clinc150"])
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()

    src = Path("data/source")
    rows = [json.loads(l) for l in open(src / f"{args.dataset}_clean_texts.jsonl")]
    rows = rows[:args.n]
    texts = [r["text"] for r in rows]
    labels = [r["label"] for r in rows]
    print(f"{args.dataset}: {len(rows)} base rows", flush=True)

    ladder, holdback = Path("data/ladder"), Path("data/holdback")
    ladder.mkdir(parents=True, exist_ok=True)
    holdback.mkdir(parents=True, exist_ok=True)

    oos_pool = []
    if args.dataset == "clinc150":
        oos_pool = [r["text"] for r in rows if r["label"] == "oos"]

    def emit(slice_id: str, shifted_texts: list[str], gold: list[str] | None,
             gold_labels: list[str | None]):
        with open(ladder / f"{args.dataset}_{slice_id}.jsonl", "w") as f:
            for i, (t, g) in enumerate(zip(shifted_texts, gold_labels)):
                f.write(json.dumps({"idx": i, "text": t, "gold_label": g}) + "\n")
        with open(holdback / f"{args.dataset}_{slice_id}.jsonl", "w") as f:
            for i, g in enumerate(gold_labels):
                f.write(json.dumps({"idx": i, "label": g}) + "\n")

    # clean control slice
    emit("clean", texts, None, labels)

    # lexical + style shifts at three severities
    for name, sev, _fn in SHIFTS:
        sid = f"{name}{sev}"
        emit(sid, apply_shift(texts, name, sev, seed=SEED), None, labels)

    # out-of-scope contamination, only where an OOS pool exists
    if oos_pool:
        for ratio in (0.1, 0.2, 0.3, 0.4):
            mixed, replaced_idx = mix_out_of_scope(texts, oos_pool, ratio, seed=SEED)
            gold: list[str | None] = []
            for i, orig in enumerate(labels):
                gold.append("oos" if i in replaced_idx else orig)
            emit(f"oos{int(ratio * 100)}", mixed, None, gold)

    slices = sorted(p.stem for p in ladder.glob(f"{args.dataset}_*.jsonl"))
    print(f"wrote {len(slices)} slices: {slices}", flush=True)


if __name__ == "__main__":
    main()