"""Filter raw shards, split a per-language held-out set, tokenize and pack to fixed-length blocks.

Input : data/raw/<lang>/<source>/*.jsonl.gz          (from download_corpora.py)
Output: data/heldout/<lang>.jsonl                      (held-out text per language, written once)
        data/packed/<lang>/<source>/                   (datasets.save_to_disk, column input_ids[seq_len])

Filters (per document): length bounds, alphabetic ratio, duplicate-line ratio, top-2-gram ratio, PII
masking (emails, phone numbers), a cheap language check (Vietnamese diacritic ratio for vi; fastText
lid.176 for every language when --fasttext-model is given), and optional MinHash near-dedup (--dedup).
CulturaX is already MinHash-deduplicated, so --dedup is meant for the small local sources.

    python data/filter_and_pack.py --mix data/mix.yaml --tokenizer tokenizer/out/mistral_vi
    python data/filter_and_pack.py --mix data/mix.yaml --tokenizer tokenizer/out/mistral_vi --only spoken_cs_vi --dedup
"""

from __future__ import annotations

import argparse
import collections
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlf_common import ROOT, iter_jsonl, load_yaml, vi_letter_ratio, write_jsonl  # noqa: E402

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d .-]{7,}\d)(?!\d)")


def doc_ok(text: str, lang: str, min_chars: int, max_chars: int) -> bool:
    n = len(text)
    if n < min_chars or n > max_chars:
        return False
    alpha = sum(c.isalpha() for c in text) / n
    if alpha < 0.6:
        return False
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines and len(set(lines)) / len(lines) < 0.7:
        return False
    words = text.split()
    if len(words) >= 20:
        grams = collections.Counter(zip(words, words[1:]))
        if grams.most_common(1)[0][1] / max(len(words) - 1, 1) > 0.1:
            return False
    if lang == "vi" and vi_letter_ratio(text) < 0.08:  # Vietnamese prose has ~15-25 % diacritic letters
        return False
    return True


def mask_pii(text: str) -> str:
    return PHONE.sub("<phone>", EMAIL.sub("<email>", text))


class Dedup:
    def __init__(self, threshold: float = 0.8, num_perm: int = 128):
        from datasketch import MinHash, MinHashLSH

        self.MinHash, self.lsh, self.num_perm = MinHash, MinHashLSH(threshold=threshold, num_perm=num_perm), num_perm
        self.n = 0

    def seen(self, text: str) -> bool:
        words = text.lower().split()
        m = self.MinHash(num_perm=self.num_perm)
        for i in range(max(len(words) - 4, 1)):
            m.update(" ".join(words[i:i + 5]).encode("utf-8"))
        if self.lsh.query(m):
            return True
        self.lsh.insert(f"d{self.n}", m)
        self.n += 1
        return False


class FastTextLID:
    def __init__(self, model_path: str):
        import fasttext

        self.m = fasttext.load_model(model_path)

    def lang(self, text: str) -> str:
        lab, _ = self.m.predict(text.replace("\n", " ")[:2000])
        return lab[0].replace("__label__", "")


def pack_and_save(texts, tok, seq_len: int, out_dir: Path, shard_docs: int = 50000) -> int:
    from datasets import Dataset, concatenate_datasets

    eos = tok.eos_token_id
    buf, blocks, parts = [], [], []

    def flush_blocks():
        nonlocal blocks
        if blocks:
            parts.append(Dataset.from_dict({"input_ids": blocks}))
            blocks = []

    batch = []
    for t in texts:
        batch.append(t)
        if len(batch) == 512:
            for ids in tok(batch, add_special_tokens=False)["input_ids"]:
                buf.extend(ids + [eos])
            batch = []
            while len(buf) >= seq_len:
                blocks.append(buf[:seq_len]); buf = buf[seq_len:]
            if len(blocks) >= shard_docs:
                flush_blocks()
    if batch:
        for ids in tok(batch, add_special_tokens=False)["input_ids"]:
            buf.extend(ids + [eos])
        while len(buf) >= seq_len:
            blocks.append(buf[:seq_len]); buf = buf[seq_len:]
    flush_blocks()
    if not parts:
        return 0
    ds = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_dir))
    return len(ds)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mix", default=str(ROOT / "data" / "mix.yaml"))
    ap.add_argument("--tokenizer", required=True, help="extended tokenizer dir (tokenizer/extend_tokenizer.py output)")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--max-chars", type=int, default=200_000)
    ap.add_argument("--dedup", action="store_true")
    ap.add_argument("--fasttext-model", default=None, help="path to lid.176.ftz for language id")
    ap.add_argument("--seq-len", type=int, default=None)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    mix = load_yaml(a.mix)
    seq_len = a.seq_len or int(mix.get("seq_len", 4096))
    raw = (ROOT / mix.get("raw_dir", "data/raw")).resolve()
    packed = (ROOT / mix.get("packed_dir", "data/packed")).resolve()
    held = (ROOT / mix.get("heldout_dir", "data/heldout")).resolve()
    held.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.tokenizer, token=False)
    lid = FastTextLID(a.fasttext_model) if a.fasttext_model else None
    held_budget = {k: int(v) * 4 for k, v in mix.get("heldout_tokens", {}).items()}  # chars ≈ 4 × tokens
    held_have = {lang: (sum(len(r["text"]) for r in iter_jsonl(held / f"{lang}.jsonl")) if (held / f"{lang}.jsonl").exists() else 0) for lang in held_budget}

    for src in mix["sources"]:
        if a.only and src["name"] not in a.only:
            continue
        lang, name = src["lang"], src["name"]
        in_dir = raw / lang / name
        files = sorted(in_dir.glob("shard-*.jsonl.gz"))
        if not files:
            print(f"[skip] {name}: no raw shards in {in_dir}")
            continue
        out_dir = packed / lang / name
        if (out_dir / "dataset_info.json").exists():
            print(f"[skip] {name}: already packed")
            continue
        dd = Dedup() if a.dedup else None
        stats = collections.Counter()
        held_rows = []

        def kept_texts():
            for f in files:
                for r in iter_jsonl(f):
                    stats["in"] += 1
                    t = r["text"]
                    if not doc_ok(t, lang, a.min_chars, a.max_chars):
                        stats["filtered"] += 1; continue
                    if lid and lid.lang(t) != lang:
                        stats["lid"] += 1; continue
                    if dd and dd.seen(t):
                        stats["dup"] += 1; continue
                    t = mask_pii(t)
                    if held_have.get(lang, 0) < held_budget.get(lang, 0):
                        held_have[lang] += len(t); held_rows.append({"text": t, "lang": lang, "source": name}); stats["heldout"] += 1
                        continue
                    stats["kept"] += 1
                    yield t

        n_blocks = pack_and_save(kept_texts(), tok, seq_len, out_dir)
        if held_rows:
            p = held / f"{lang}.jsonl"
            with open(p, "a", encoding="utf-8") as f:
                import json
                for r in held_rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[done] {name}: in {stats['in']:,}  filtered {stats['filtered']:,}  lid {stats['lid']:,}  dup {stats['dup']:,}  "
              f"heldout {stats['heldout']:,}  kept {stats['kept']:,}  -> {n_blocks:,} blocks x {seq_len} = {n_blocks * seq_len / 1e6:.1f} M tokens")


if __name__ == "__main__":
    main()
