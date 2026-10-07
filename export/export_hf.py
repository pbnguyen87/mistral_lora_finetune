"""Reassemble a full HF checkpoint from an init model + trained new rows (+ merged LoRA adapter).

    # after stage 1
    python export/export_hf.py --model checkpoints/mistral-7b-vi-init --new-rows runs/stage1/final/new_rows.safetensors \\
        --out checkpoints/mistral-7b-vi-stage1
    # after stage 2
    python export/export_hf.py --model checkpoints/mistral-7b-vi-stage1 --new-rows runs/stage2/final/new_rows.safetensors \\
        --adapter runs/stage2/final --out checkpoints/mistral-7b-vi-final

The output has a safetensors index with `model.embed_tokens.weight`, so
Confucius4-TTS/vistral_finetune/build_hybrid_embedding.py --vistral-repo <out> can consume it directly.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--new-rows", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", choices=["keep", "bf16", "fp16", "fp32"], default="keep")
    a = ap.parse_args()
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model, token=False)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype="auto", token=False)
    if a.new_rows:
        nr = load_file(a.new_rows)
        with open(a.new_rows, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]; meta = json.loads(f.read(hlen)).get("__metadata__", {})
        first = int(meta["first_new_id"])
        emb, head = model.get_input_embeddings().weight, model.get_output_embeddings().weight
        with torch.no_grad():
            emb[first:first + nr["embed_new"].shape[0]] = nr["embed_new"].to(emb.dtype)
            if head.data_ptr() != emb.data_ptr():
                head[first:first + nr["head_new"].shape[0]] = nr["head_new"].to(head.dtype)
        print(f"[export] applied {nr['embed_new'].shape[0]} new rows from id {first}")
    if a.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
        print(f"[export] merged LoRA adapter {a.adapter}")
    if a.dtype != "keep":
        model = model.to({"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[a.dtype])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")
    tok.save_pretrained(out)
    idx = out / "model.safetensors.index.json"
    ok = idx.is_file() and "model.embed_tokens.weight" in json.loads(idx.read_text())["weight_map"]
    print(f"[export] wrote {out}  index with embed_tokens: {ok if idx.is_file() else 'single-file checkpoint'}")


if __name__ == "__main__":
    main()
