"""Bits-per-byte per language on the held-out sets, for any HF checkpoint (optionally + adapter / new rows).

    python eval/eval_bpb.py --model checkpoints/mistral-7b-vi-init --langs vi en fr de es it
    python eval/eval_bpb.py --model checkpoints/mistral-7b-vi-stage1 --adapter runs/stage2/final --new-rows runs/stage2/final/new_rows.safetensors
    python eval/eval_bpb.py --model mistralai/Mistral-7B-v0.1 --tokenizer ../Confucius4-TTS/checkpoints   # base reference
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlf_common import ROOT, bits_per_byte, iter_jsonl  # noqa: E402


def load_model(model_path: str, tokenizer_path: str | None, adapter: str | None, new_rows: str | None, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path or model_path, token=False)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, token=False)
    if new_rows:
        from safetensors.torch import load_file

        nr = load_file(new_rows)
        with open(new_rows, "rb") as f:  # metadata holds first_new_id
            import struct
            hlen = struct.unpack("<Q", f.read(8))[0]; meta = json.loads(f.read(hlen)).get("__metadata__", {})
        first = int(meta.get("first_new_id", model.get_input_embeddings().weight.shape[0] - nr["embed_new"].shape[0]))
        with torch.no_grad():
            model.get_input_embeddings().weight[first:first + nr["embed_new"].shape[0]] = nr["embed_new"].to(model.dtype)
            if model.get_output_embeddings().weight.data_ptr() != model.get_input_embeddings().weight.data_ptr():
                model.get_output_embeddings().weight[first:first + nr["head_new"].shape[0]] = nr["head_new"].to(model.dtype)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    return tok, model.to(device).eval()


def evaluate(model, tok, heldout_dir: Path, langs, max_docs: int, seq_len: int, device: str) -> dict:
    import torch

    out = {}
    with torch.no_grad():
        for lang in langs:
            p = heldout_dir / f"{lang}.jsonl"
            if not p.exists():
                out[lang] = None; continue
            nats, nbytes, n = 0.0, 0, 0
            for _, r in zip(range(max_docs), iter_jsonl(p)):
                ids = tok(r["text"], add_special_tokens=False, truncation=True, max_length=seq_len)["input_ids"]
                if len(ids) < 2:
                    continue
                x = torch.tensor([ids], device=device)
                nats += model(input_ids=x, labels=x).loss.item() * (len(ids) - 1)
                nbytes += len(tok.decode(ids[1:]).encode("utf-8")); n += 1
            out[lang] = {"bpb": round(bits_per_byte(nats, max(nbytes, 1)), 4), "docs": n, "bytes": nbytes}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--new-rows", default=None)
    ap.add_argument("--heldout-dir", default=str(ROOT / "data" / "heldout"))
    ap.add_argument("--langs", nargs="+", default=["vi", "en", "fr", "de", "es", "it"])
    ap.add_argument("--max-docs", type=int, default=200)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--out", default=None, help="write results json")
    a = ap.parse_args()
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, model = load_model(a.model, a.tokenizer, a.adapter, a.new_rows, device)
    res = evaluate(model, tok, Path(a.heldout_dir), a.langs, a.max_docs, a.seq_len, device)
    print(f"{'lang':5s} {'bits/byte':>10s} {'docs':>6s}")
    for lang, r in res.items():
        print(f"{lang:5s} {('%.4f' % r['bpb']) if r else 'n/a':>10s} {(r['docs'] if r else 0):6d}")
    if a.out:
        Path(a.out).write_text(json.dumps({"model": a.model, "adapter": a.adapter, "new_rows": a.new_rows, "results": res}, indent=2))


if __name__ == "__main__":
    main()
