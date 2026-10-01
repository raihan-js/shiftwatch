#!/usr/bin/env python3
"""Upload the shift ladder and benchmark results to Hugging Face.

Usage:
  PYTHONPATH=src python scripts/publish_hf.py
"""
import json
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "raihan-js/shiftwatch-ladder"


def main() -> None:
    api = HfApi()
    api.create_repo(REPO_ID, repo_type="dataset", exist_ok=True)

    files = []
    for pattern in ["data/ladder/*.jsonl", "data/holdback/*.jsonl",
                    "data/results/*.json", "data/source/*_meta.json"]:
        files.extend(Path().glob(pattern))

    for f in files:
        api.upload_file(path_or_fileobj=str(f), path_in_repo=str(f),
                        repo_id=REPO_ID, repo_type="dataset")
        print(f"  uploaded {f}")

    readme = Path("data/README.md")
    if not readme.exists():
        readme.write_text("""---
license: apache-2.0
task_categories:
- text-classification
tags:
- distribution-shift
- label-free-evaluation
pretty_name: ShiftWatch Ladder
---

# ShiftWatch Ladder

Shift ladder for label-free accuracy estimation. Two datasets (Banking77, CLINC150),
rule-based corruptions (typos, abbreviations, style, OOS mixing), labels held back.

Each slice: `data/ladder/<dataset>_<slice>.jsonl` (shifted text + gold label)
and `data/holdback/<dataset>_<slice>.jsonl` (labels only, read after estimation).
""")
    api.upload_file(path_or_fileobj=str(readme), path_in_repo="README.md",
                    repo_id=REPO_ID, repo_type="dataset")
    print(f"Uploaded {len(files)} files to {REPO_ID}")


if __name__ == "__main__":
    main()