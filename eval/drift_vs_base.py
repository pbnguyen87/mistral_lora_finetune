"""How far did the shared embedding rows move, and how do the new rows look? (PLAN.md §5)

Compares the input embedding of a trained/exported checkpoint with the base model's: per-row cosine over
the shared ids, a few named tokens, and norms of the new rows. Reads only the embedding tensor from each
safetensors index, so it is cheap even for 7B checkpoints.

    python eval/drift_vs_base.py --base mistralai/Mistral-7B-v0.1 --tuned checkpoints/mistral-7b-vi-final \\
        --tokenizer checkpoints/mistral-7b-vi-final
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlf_common import MISTRAL_VOCAB  # noqa: E402

EMBED_KEYS = ("model.embed_tokens.weight", "embed_tokens.weight", "transformer.wte.weight")


def load_embedding(path_or_repo: str):
    """Return the input-embedding tensor from a local HF dir or a Hub repo without loading the model."""
    import torch
    from safetensors import safe_open

    p = Path(path_or_repo)
    if not p.is_dir():
        from huggingface_hub import snapshot_download

        p = Path(snapshot_download(path_or_repo, allow_patterns=["*.json", "*.safetensors"], token=False))
    idx = p / "model.safetensors.index.json"
    if idx.is_file():
        wm = json.loads(idx.read_text())["weight_map"]
        key = next(k for k in EMBED_KEYS if k in wm)
        f = p / wm[key]
    else:
        f, key = p / "model.safetensors", None
    with safe_open(str(f), "pt") as sf:
        key = key or next(k for k in EMBED_KEYS if k in sf.keys())
        return sf.get_tensor(key).float()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--tuned", required=True)
    ap.add_argument("--tokenizer", required=True, help="the EXTENDED tokenizer (to name rows)")
    ap.add_argument("--first-new-id", type=int, default=MISTRAL_VOCAB)
    ap.add_argument("--probe", nargs="*", default=["▁the", "▁là", "请", "▁deep", "▁und", "▁que"])
    a = ap.parse_args()
    import torch
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer, token=False)
    B, T = load_embedding(a.base), load_embedding(a.tuned)
    n = min(B.shape[0], a.first_new_id)
    cos = torch.nn.functional.cosine_similarity(B[:n], T[:n], dim=1)
    nz = B[:n].norm(dim=1) > 0
    print(f"base {tuple(B.shape)}  tuned {tuple(T.shape)}  shared rows compared: {n}")
    print(f"shared-row cosine: mean {cos[nz].mean():.4f}  p10 {cos[nz].quantile(.1):.4f}  median {cos[nz].median():.4f}  identical rows {(B[:n] == T[:n]).all(1).sum().item()}")
    v = tok.get_vocab()
    for tkn in a.probe:
        if tkn in v and v[tkn] < n:
            print(f"  {tkn!r:10s} id {v[tkn]:6d} cos {cos[v[tkn]]:.4f}")
    if T.shape[0] > n:
        new = T[n:]
        print(f"new rows: {new.shape[0]}  mean norm {new.norm(dim=1).mean():.4f}  vs shared base {B[:n][nz].norm(dim=1).mean():.4f} / shared tuned {T[:n][nz].norm(dim=1).mean():.4f}")
        if T.shape[0] >= n + 5:
            print("  examples:", [tok.convert_ids_to_tokens(i) for i in range(n, n + 8)])


if __name__ == "__main__":
    main()
