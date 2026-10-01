"""Controlled shift ladder: deterministic, rule-based corruptions.

Every severity is a pure function of (text, severity, seed) so the ladder is
reproducible without storing perturbed copies and without a GPU. Real labels
are never touched here - they are held back by the benchmark runner.
"""

from __future__ import annotations

import random
import re

# Domain abbreviations that appear in short customer-service text.
ABBREVIATIONS = {
    "account": ["acct", "acc"],
    "balance": ["bal", "banc"],
    "card": ["crd"],
    "charge": ["chg", "chrg"],
    "payment": ["pmt", "paymt"],
    "transaction": ["txn", "trans"],
    "transfer": ["xfer", "trf"],
    "deposit": ["dep", "dpt"],
    "withdrawal": ["wd", "wdl"],
    "interest": ["int"],
    "statement": ["stmt", "stmnt"],
    "fee": ["fees"],
    "refund": ["ref", "rfnd"],
    "cash": ["csh"],
    "credit": ["cr", "cred"],
    "debit": ["db", "dbt"],
    "loan": ["ln"],
    "mortgage": ["mtg", "mortg"],
    "insurance": ["ins", "insur"],
    "cancel": ["cxl", "cncl"],
    "confirm": ["cnfm", "cfrm"],
    "complaint": ["cmplnt", "compl"],
    "question": ["ques"],
    "available": ["avail", "avl"],
    "please": ["pls", "plz"],
    "amount": ["amt"],
    "date": ["dt"],
    "number": ["num"],
    "problem": ["prblm"],
    "help": ["hlp"],
}

_TYPO_KEYBOARD_NEIGHBOURS = {
    "a": "qwsz", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wsdr",
    "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb", "i": "ujko", "j": "huikmn",
    "k": "jiolm", "l": "kop", "m": "njk", "n": "bhjm", "o": "iklp",
    "p": "ol", "q": "wa", "r": "edft", "s": "awedxz", "t": "rfgy",
    "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu",
    "z": "asx",
}


def add_typos(text: str, severity: int, seed: int = 0) -> str:
    """Keyboard-adjacent swaps, deletions and duplications.

    severity is the fraction of eligible words to corrupt (0.05, 0.10, 0.20).
    """
    if severity <= 0:
        return text
    rng = random.Random(f"typo:{seed}:{severity}")
    words = text.split()
    out = []
    for word in words:
        if len(word) < 4 or rng.random() > severity:
            out.append(word)
            continue
        op = rng.choice(["swap", "drop", "dup"])
        idx = rng.randrange(1, len(word) - 1)
        ch = word[idx].lower()
        if op == "swap" and ch in _TYPO_KEYBOARD_NEIGHBOURS:
            repl = rng.choice(_TYPO_KEYBOARD_NEIGHBOURS[ch])
            out.append(word[:idx] + repl + word[idx + 1:])
        elif op == "drop":
            out.append(word[:idx] + word[idx + 1:])
        else:
            out.append(word[:idx] + ch + word[idx:])
    return " ".join(out)


def abbreviate(text: str, severity: int, seed: int = 0) -> str:
    """Replace domain words with their abbreviations.

    severity is the fraction of abbreviations applied (0.3, 0.6, 1.0 of hits).
    """
    if severity <= 0:
        return text
    rng = random.Random(f"abbr:{seed}:{severity}")
    tokens = re.split(r"(\s+)", text)
    hits = [i for i, t in enumerate(tokens) if t.lower().strip(".,!?") in ABBREVIATIONS]
    if not hits:
        return text
    n_apply = max(1, int(round(len(hits) * severity)))
    for i in rng.sample(hits, n_apply):
        bare = tokens[i].lower().strip(".,!?")
        repl = rng.choice(ABBREVIATIONS[bare])
        tokens[i] = tokens[i].replace(bare, repl)
    return "".join(tokens)


def style_shift(text: str, severity: int, seed: int = 0) -> str:
    """Lower-casing, dropped punctuation, politeness padding, whitespace noise."""
    if severity <= 0:
        return text
    rng = random.Random(f"style:{seed}:{severity}")
    out = text
    if severity >= 1:
        out = out.lower()
        out = re.sub(r"[.,!?;:]", "", out)
    if severity >= 2 and rng.random() < 0.5:
        out = f"hi there , {out.strip().lower()}"
    if severity >= 3:
        out = re.sub(r"\s+", "  ", out).strip()
        if rng.random() < 0.3:
            out = out.upper()
    return out


def mix_out_of_scope(texts: list[str], oos_texts: list[str], ratio: float,
                     seed: int = 0) -> tuple[list[str], list[str | None]]:
    """Replace a `ratio` share of in-scope texts with out-of-scope ones.

    Returns (texts, gold) where gold[i] is None for an injected OOS item.
    """
    if ratio <= 0:
        return list(texts), [None] * len(texts)
    rng = random.Random(f"oos:{seed}:{ratio}")
    n_inject = int(round(len(texts) * ratio))
    idx = rng.sample(range(len(texts)), min(n_inject, len(texts)))
    out = list(texts)
    gold: list[str | None] = [None] * len(texts)
    for j, i in enumerate(sorted(idx)):
        out[i] = oos_texts[(j + seed) % len(oos_texts)]
    return out, gold


# Ladder definition: (shift_name, severity_label, callable)
SHIFTS = [
    ("typo", 1, lambda t, s: add_typos(t, 0.05, s)),
    ("typo", 2, lambda t, s: add_typos(t, 0.10, s)),
    ("typo", 3, lambda t, s: add_typos(t, 0.20, s)),
    ("abbrev", 1, lambda t, s: abbreviate(t, 0.3, s)),
    ("abbrev", 2, lambda t, s: abbreviate(t, 0.6, s)),
    ("abbrev", 3, lambda t, s: abbreviate(t, 1.0, s)),
    ("style", 1, lambda t, s: style_shift(t, 1, s)),
    ("style", 2, lambda t, s: style_shift(t, 2, s)),
    ("style", 3, lambda t, s: style_shift(t, 3, s)),
]


def apply_shift(texts: list[str], name: str, severity: int, seed: int = 0) -> list[str]:
    for shift, sev, fn in SHIFTS:
        if shift == name and sev == severity:
            return [fn(t, seed) for t in texts]
    raise KeyError(f"unknown shift {name}/{severity}")