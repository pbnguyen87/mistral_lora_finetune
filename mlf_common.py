"""Shared helpers for mistral_lora_finetune (imported by the scripts via sys.path)."""

from __future__ import annotations

import gzip
import json
import math
import os
import re
from pathlib import Path
from typing import Iterable, Iterator

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")  # stale ~/.cache/huggingface/token -> 401 on public repos
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MISTRAL_VOCAB = 32000
VI_LETTERS = "ăâêôơưđàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ"
_VI_RE = re.compile(f"[{VI_LETTERS}{VI_LETTERS.upper()}]")
_PUNCT_RE = re.compile(r"[.,!?;:()\[\]\"'“”‘’…\-–—/]")


def has_vietnamese_letter(s: str) -> bool:
    return bool(_VI_RE.search(s))


def vi_letter_ratio(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    return (sum(1 for c in letters if _VI_RE.match(c)) / len(letters)) if letters else 0.0


def syllables(text: str) -> int:
    """Whitespace tokens after stripping punctuation; for Vietnamese each token is one syllable."""
    return len([w for w in _PUNCT_RE.sub(" ", text).split() if w])


def tokens_per_syllable(tok, text: str) -> float:
    return len(tok.encode(text, add_special_tokens=False)) / max(syllables(text), 1)


def bits_per_byte(total_loss_nats: float, n_bytes: int) -> float:
    return total_loss_nats / (n_bytes * math.log(2))


def load_yaml(path: str | os.PathLike) -> dict:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def iter_jsonl(path: str | os.PathLike) -> Iterator[dict]:
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | os.PathLike, rows: Iterable[dict]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if p.suffix == ".gz" else open
    n = 0
    with opener(p, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_text_column(path: str | os.PathLike, text_col: str | None = None) -> Iterator[str]:
    """Yield text from .txt / .jsonl(.gz) / .csv / .tsv / .parquet with an auto-detected text column."""
    p = Path(path)
    suf = p.suffix.lower()
    if suf in (".txt", ".md", ".markdown"):
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.strip():
                    yield line.strip()
    elif suf in (".jsonl", ".gz"):
        for r in iter_jsonl(p):
            col = text_col or next((c for c in ("text", "segment_text", "comment", "sentence", "content") if c in r), None)
            if col and r.get(col):
                yield str(r[col])
    elif suf in (".csv", ".tsv", ".parquet"):
        import pandas as pd

        df = pd.read_parquet(p) if suf == ".parquet" else pd.read_csv(p, sep="\t" if suf == ".tsv" else ",", dtype=str, keep_default_na=False)
        col = text_col or next((c for c in df.columns if c.lower() in ("text", "segment_text", "comment", "sentence", "content", "review", "normalized")), df.columns[0])
        for v in df[col].astype(str):
            if v.strip():
                yield v
    else:
        raise ValueError(f"unsupported file type: {p}")
