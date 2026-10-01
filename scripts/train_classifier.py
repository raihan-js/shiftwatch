#!/usr/bin/env python3
"""Fine-tune ModernBERT-base classifiers and cache source logits.

The cached source-validation logits are the ONLY place labels are allowed to
be seen. Every estimator in the benchmark is fitted on them and then applied to
shifted data with labels withheld.

Usage:
  PYTHONPATH=src python scripts/train_classifier.py --dataset banking77
  PYTHONPATH=src python scripts/train_classifier.py --dataset clinc150
"""
import argparse
import json
from pathlib import Path

import numpy as np

MODEL = "answerdotai/ModernBERT-base"
DATA_DIR = Path("data")
SEED = 0


def load_dataset_splits(name: str):
    from datasets import load_dataset
    if name == "banking77":
        ds = load_dataset("PolyAI/banking77")
        return ds["train"], ds["test"], "text", "label"
    if name == "clinc150":
        ds = load_dataset("clinc_oos_plus", "plus")
        return ds["train"], ds["test"], "text", "label"
    raise ValueError(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["banking77", "clinc150"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-eval", type=int, default=2000)
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_ds, test_ds, text_col, label_col = load_dataset_splits(args.dataset)
    id2label = {i: train_ds.features[label_col].names[i]
                for i in range(len(train_ds.features[label_col].names))}
    n_classes = len(id2label)
    print(f"{args.dataset}: {len(train_ds)} train / {len(test_ds)} test / {n_classes} classes",
          flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=n_classes, id2label=id2label, label2id={v: k for k, v in id2label.items()})

    def tokenize(batch):
        return tok(batch[text_col], truncation=True, max_length=args.max_len)

    train_tok = train_ds.map(tokenize, batched=True)
    eval_tok = test_ds.map(tokenize, batched=True)
    if args.max_train:
        train_tok = train_tok.select(range(min(args.max_train, len(train_tok))))
    eval_tok = eval_tok.select(range(min(args.max_eval, len(eval_tok))))

    model_dir = DATA_DIR / "models" / f"modernbert-{args.dataset}"
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(model_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=64,
            learning_rate=args.lr,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="no",
            logging_steps=50,
            report_to=[],
            seed=SEED,
            dataloader_num_workers=2,
        ),
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=DataCollatorWithPadding(tok),
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(model_dir))
    tok.save_pretrained(str(model_dir))
    print(f"eval: {metrics}", flush=True)

    # Cache logits for the clean eval slice: these carry labels and are the
    # estimator fitting set. Shifted slices are built separately.
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = DataLoader(eval_tok, batch_size=64, collate_fn=DataCollatorWithPadding(tok))
    all_logits, all_labels, all_texts = [], [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits.float().cpu().numpy()
            all_logits.append(logits)
            all_labels.append(labels.numpy())
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    acc = float((logits.argmax(axis=1) == labels).mean())

    out = DATA_DIR / "source"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{args.dataset}_clean.npz", logits=logits, labels=labels)
    with open(out / f"{args.dataset}_clean_texts.jsonl", "w") as f:
        for i, row in enumerate(eval_tok):
            f.write(json.dumps({"idx": i, "text": row[text_col],
                                "label": id2label[int(labels[i])]}) + "\n")

    meta = {"dataset": args.dataset, "base_model": MODEL, "n_classes": n_classes,
            "n_eval": int(len(labels)), "clean_accuracy": acc,
            "eval_metrics": {k: float(v) for k, v in metrics.items()},
            "seed": SEED, "max_len": args.max_len, "epochs": args.epochs}
    (out / f"{args.dataset}_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()