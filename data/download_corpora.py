"""Stream the sources in data/mix.yaml into local jsonl.gz shards: data/raw/<lang>/<source>/shard-NNNNN.jsonl.gz

Each row: {"text": ..., "lang": ..., "source": ...}. HF sources are streamed and stopped at `max_gb`
(uncompressed UTF-8 bytes of text); local sources are copied through unchanged. Resumable: finished
sources are skipped when a `_DONE` marker exists.

    python data/download_corpora.py --mix data/mix.yaml                     # everything
    python data/download_corpora.py --mix data/mix.yaml --only culturax_vi wiki_vi
    python data/download_corpora.py --mix data/mix.yaml --max-gb-override 0.05   # tiny test pull
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlf_common import ROOT, load_yaml, read_text_column, write_jsonl  # noqa: E402

SHARD_BYTES = 256 * 1024 * 1024  # uncompressed text per shard


def _shard_writer(out_dir: Path, lang: str, source: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {"idx": len(list(out_dir.glob("shard-*.jsonl.gz"))), "buf": [], "bytes": 0, "total": 0, "docs": 0}

    def flush():
        if state["buf"]:
            write_jsonl(out_dir / f"shard-{state['idx']:05d}.jsonl.gz", state["buf"])
            state["idx"] += 1
            state["buf"] = []
            state["bytes"] = 0

    def add(text: str):
        b = len(text.encode("utf-8"))
        state["buf"].append({"text": text, "lang": lang, "source": source})
        state["bytes"] += b
        state["total"] += b
        state["docs"] += 1
        if state["bytes"] >= SHARD_BYTES:
            flush()

    return add, flush, state


def pull_hf(src: dict, out_dir: Path, max_bytes: int) -> dict:
    from datasets import load_dataset

    h = src["hf"]
    token = os.environ.get("HF_TOKEN") or None
    ds = load_dataset(h["path"], h.get("name"), split=h.get("split", "train"), streaming=True, token=token)
    add, flush, st = _shard_writer(out_dir, src["lang"], src["name"])
    field = h.get("text_field", "text")
    t0 = time.time()
    for i, row in enumerate(ds):
        t = row.get(field)
        if not t:
            continue
        add(t)
        if st["total"] >= max_bytes:
            break
        if i % 20000 == 0 and i:
            print(f"  [{src['name']}] {st['docs']:>9,d} docs  {st['total'] / 1e9:6.2f} GB  {time.time() - t0:6.0f}s", flush=True)
    flush()
    return {"docs": st["docs"], "gb": st["total"] / 1e9}


def pull_local(src: dict, out_dir: Path) -> dict:
    add, flush, st = _shard_writer(out_dir, src["lang"], src["name"])
    for pat in src["local"]:
        files = sorted(glob.glob(str((ROOT / pat).resolve()))) if not os.path.isabs(pat) else sorted(glob.glob(pat))
        if not files:
            print(f"  [{src['name']}] WARNING no files match {pat}")
        for f in files:
            for t in read_text_column(f):
                add(t)
    flush()
    return {"docs": st["docs"], "gb": st["total"] / 1e9}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mix", default=str(ROOT / "data" / "mix.yaml"))
    ap.add_argument("--only", nargs="*", help="source names to pull")
    ap.add_argument("--max-gb-override", type=float, default=None, help="cap every HF source at this many GB (testing)")
    a = ap.parse_args()
    mix = load_yaml(a.mix)
    raw = (ROOT / mix.get("raw_dir", "data/raw")).resolve()
    for src in mix["sources"]:
        if a.only and src["name"] not in a.only:
            continue
        out_dir = raw / src["lang"] / src["name"]
        if (out_dir / "_DONE").exists():
            print(f"[skip] {src['name']} already done")
            continue
        stale = list(out_dir.glob("shard-*.jsonl.gz"))
        if stale:  # interrupted earlier: streaming restarts at the beginning, so drop partial shards
            for f in stale:
                f.unlink()
            print(f"[pull] {src['name']}: removed {len(stale)} partial shard(s) from an interrupted run")
        print(f"[pull] {src['name']} ({src['lang']})", flush=True)
        try:
            if "hf" in src:
                gb = a.max_gb_override if a.max_gb_override is not None else float(src.get("max_gb", 1))
                r = pull_hf(src, out_dir, int(gb * 1e9))
            else:
                r = pull_local(src, out_dir)
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {src['name']}: {type(e).__name__}: {str(e)[:200]}")
            continue
        (out_dir / "_DONE").write_text(f"{r}\n")
        print(f"[done] {src['name']}: {r['docs']:,} docs, {r['gb']:.2f} GB -> {out_dir}")


if __name__ == "__main__":
    main()
