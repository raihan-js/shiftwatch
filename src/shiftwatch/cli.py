"""ShiftWatch CLI: build, benchmark, serve, replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def cmd_bench(args: argparse.Namespace) -> int:
    import runpy
    import sys
    sys.argv = ["run_benchmark.py", "--dataset", args.dataset]
    runpy.run_path("scripts/run_benchmark.py", run_name="__main__")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import numpy as np
    import uvicorn

    from shiftwatch.estimators import ESTIMATORS
    from shiftwatch.sidecar import RollingEstimate, create_app

    src = np.load(Path("data/source") / f"{args.dataset}_clean.npz")
    source_logits, source_labels = src["logits"], src["labels"]
    est = ESTIMATORS[args.estimator]().fit(source_logits, source_labels)
    rolling = RollingEstimate(window=args.window, estimator=est, threshold=args.threshold,
                              source_accuracy=float((source_logits.argmax(axis=1) == source_labels).mean()))
    uvicorn.run(create_app(rolling), host=args.host, port=args.port)
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    """Stream a ladder slice through the sidecar, in slices, and report alerts."""
    import numpy as np

    from shiftwatch.estimators import ESTIMATORS
    from shiftwatch.sidecar import RollingEstimate

    src = np.load(Path("data/source") / f"{args.dataset}_clean.npz")
    source_logits, source_labels = src["logits"], src["labels"]
    source_acc = float((source_logits.argmax(axis=1) == source_labels).mean())
    est = ESTIMATORS[args.estimator]().fit(source_logits, source_labels)

    results = Path("data/results")
    slices = args.slices or [p.stem for p in sorted(Path("data/ladder").glob(f"{args.dataset}_*.jsonl"))]
    rolling = RollingEstimate(window=args.window, estimator=est,
                              threshold=args.threshold, source_accuracy=source_acc)
    rows = []
    for slice_id in slices:
        npz = results / f"{args.dataset}_{slice_id[len(args.dataset) + 1:]}_logits.npz"
        if not npz.exists():
            npz = results / f"{args.dataset}_{slice_id}_logits.npz"
        logits = np.load(npz)["logits"]
        rolling.probs.clear()
        for row in logits:
            rolling.push(np.exp(row - row.max()))
        rows.append({"slice": slice_id, "estimated": rolling.estimated_accuracy(),
                     "estimated_drop": rolling.estimated_drop(), "alerting": rolling.alerting()})
        print(f"{slice_id:12s} est={rows[-1]['estimated']:.3f} "
              f"drop={rows[-1]['estimated_drop']:+.3f} "
              f"alert={'YES' if rows[-1]['alerting'] else 'no'}", flush=True)
    Path("data/results").mkdir(exist_ok=True)
    (Path("data/results") / f"{args.dataset}_replay.json").write_text(json.dumps(rows, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="shiftwatch")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bench", help="run the estimator benchmark")
    b.add_argument("--dataset", required=True, choices=["banking77", "clinc150"])
    b.set_defaults(func=cmd_bench)

    s = sub.add_parser("serve", help="run the monitoring sidecar")
    s.add_argument("--dataset", default="clinc150")
    s.add_argument("--estimator", default="cbpe")
    s.add_argument("--window", type=int, default=500)
    s.add_argument("--threshold", type=float, default=0.05)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8077)
    s.set_defaults(func=cmd_serve)

    r = sub.add_parser("replay", help="stream ladder slices through the estimator")
    r.add_argument("--dataset", default="clinc150")
    r.add_argument("--estimator", default="cbpe")
    r.add_argument("--slices", nargs="*", default=None)
    r.add_argument("--window", type=int, default=500)
    r.add_argument("--threshold", type=float, default=0.05)
    r.set_defaults(func=cmd_replay)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())