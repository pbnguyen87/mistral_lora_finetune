"""Resize a Mistral checkpoint to the extended vocabulary and initialise the new rows (PLAN.md §2).

New input-embedding rows and new lm_head rows = mean of the Mistral rows of each new piece's Mistral
sub-pieces (the same rule used in Confucius4-TTS/vistral_finetune). Pad pieces get the mean of all old
rows. Writes a full HF checkpoint (safetensors, sharded) + the extended tokenizer, ready for train.py.

    python train/init_new_rows.py --base mistralai/Mistral-7B-v0.1 --base-tokenizer ../Confucius4-TTS/checkpoints \\
        --new-tokenizer tokenizer/out/mistral_vi --out checkpoints/mistral-7b-vi-init
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlf_common import MISTRAL_VOCAB  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="HF id or dir of Mistral-7B-v0.1 (or any Mistral-arch model)")
    ap.add_argument("--base-tokenizer", default=None, help="dir with the ORIGINAL tokenizer (default: --base)")
    ap.add_argument("--new-tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--first-new-id", type=int, default=None, help="default: size of the base tokenizer")
    ap.add_argument("--renorm-new-rows", action="store_true", help="rescale each new row to the mean norm of the old rows (sub-piece averaging shrinks norms)")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[a.dtype]
    old_tok = AutoTokenizer.from_pretrained(a.base_tokenizer or a.base, token=False)
    new_tok = AutoTokenizer.from_pretrained(a.new_tokenizer, token=False)
    first_new = a.first_new_id or len(old_tok)
    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=dt, token=False)
    old_vocab_rows = model.get_input_embeddings().weight.shape[0]
    print(f"[init] model embedding rows {old_vocab_rows}, old tokenizer {len(old_tok)}, new tokenizer {len(new_tok)}, first new id {first_new}")
    if old_vocab_rows < first_new:
        sys.exit("[init] model has fewer embedding rows than the base tokenizer")

    try:
        model.resize_token_embeddings(len(new_tok), pad_to_multiple_of=None, mean_resizing=False)  # rows are overwritten below
    except TypeError:  # older transformers
        model.resize_token_embeddings(len(new_tok), pad_to_multiple_of=None)
    emb = model.get_input_embeddings().weight.data
    head = model.get_output_embeddings().weight.data
    tied = emb.data_ptr() == head.data_ptr()
    with torch.no_grad():
        old_e, old_h = emb[:first_new].float(), head[:first_new].float()
        mean_e, mean_h = old_e.mean(0), old_h.mean(0)
        n_sub, n_pad = [], 0
        for tid in range(first_new, len(new_tok)):
            piece = new_tok.convert_ids_to_tokens(tid)
            if piece.startswith("<pad_extra_"):
                emb[tid] = mean_e.to(emb.dtype)
                if not tied:
                    head[tid] = mean_h.to(head.dtype)
                n_pad += 1
                continue
            sub = [s for s in old_tok.encode(piece.replace("▁", " "), add_special_tokens=False) if s < first_new]
            if not sub:
                sub = [old_tok.unk_token_id or 0]
            n_sub.append(len(sub))
            emb[tid] = old_e[sub].mean(0).to(emb.dtype)
            if not tied:
                head[tid] = old_h[sub].mean(0).to(head.dtype)
    if a.renorm_new_rows:
        with torch.no_grad():
            for W, old in ((emb, old_e), (head, old_h)) if not tied else ((emb, old_e),):
                tgt = old.norm(dim=1).mean()
                rows = W[first_new:].float(); W[first_new:] = (rows * (tgt / rows.norm(dim=1, keepdim=True).clamp_min(1e-8))).to(W.dtype)
        print(f"[init] new rows rescaled to old-row mean norm")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")
    new_tok.save_pretrained(out)
    info = {"base": a.base, "first_new_id": first_new, "new_vocab": len(new_tok), "new_pieces": len(n_sub), "pad_pieces": n_pad,
            "tied_embeddings": tied, "mean_subpieces_per_new_piece": round(sum(n_sub) / max(len(n_sub), 1), 2), "dtype": a.dtype}
    (out / "init_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"[init] {len(n_sub)} new pieces initialised from on average {info['mean_subpieces_per_new_piece']} sub-pieces, {n_pad} pad rows; tied={tied}")
    print(f"[init] wrote {out}")


if __name__ == "__main__":
    main()
