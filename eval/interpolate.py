"""Weight interpolation between the base model and the tuned (exported, merged) model (PLAN.md §4 item 4).

For each alpha: W = (1 - alpha) * base + alpha * tuned on every tensor with matching shape; the new
embedding/head rows (absent from the base) are kept from the tuned model. Reports bits-per-byte per
language for each alpha and optionally saves the best mix by a weighted objective.

    python eval/interpolate.py --base mistralai/Mistral-7B-v0.1 --tuned checkpoints/mistral-7b-vi-final \\
        --alphas 0.5 0.7 0.85 1.0 --langs vi en fr de es it --save-best checkpoints/mistral-7b-vi-interp
Memory: two copies of the model in RAM/VRAM (bf16: ~29 GB for 7B).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlf_common import ROOT  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_bpb import evaluate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--tuned", required=True, help="full merged checkpoint with the extended vocab")
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.5, 0.7, 0.85, 1.0])
    ap.add_argument("--langs", nargs="+", default=["vi", "en", "fr", "de", "es", "it"])
    ap.add_argument("--heldout-dir", default=str(ROOT / "data" / "heldout"))
    ap.add_argument("--max-docs", type=int, default=100)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--objective", default="vi:1.0,en:1.0,fr:0.25,de:0.25,es:0.25,it:0.25", help="weights for a BPB sum to minimise")
    ap.add_argument("--save-best", default=None)
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(a.tuned, token=False)
    base = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=dtype, token=False)
    tuned = AutoModelForCausalLM.from_pretrained(a.tuned, torch_dtype=dtype, token=False).to(device)
    base_sd = {k: v.to(device) for k, v in base.state_dict().items()}
    tuned_sd = {k: v.clone() for k, v in tuned.state_dict().items()}
    del base
    weights = {k: float(v) for k, v in (kv.split(":") for kv in a.objective.split(","))}

    results, best = [], None
    for alpha in a.alphas:
        sd = {}
        for k, t in tuned_sd.items():
            b = base_sd.get(k)
            if b is None or b.shape != t.shape:
                if b is not None and b.dim() == 2 and b.shape[1] == t.shape[1] and b.shape[0] < t.shape[0]:
                    mixed = t.clone(); mixed[: b.shape[0]] = (1 - alpha) * b + alpha * t[: b.shape[0]]
                    sd[k] = mixed
                else:
                    sd[k] = t
            else:
                sd[k] = (1 - alpha) * b + alpha * t
        tuned.load_state_dict(sd, strict=False)
        res = evaluate(tuned.eval(), tok, Path(a.heldout_dir), a.langs, a.max_docs, a.seq_len, device)
        score = sum(weights.get(l, 0) * (r["bpb"] if r else 0) for l, r in res.items())
        results.append({"alpha": alpha, "score": round(score, 4), **{l: (r["bpb"] if r else None) for l, r in res.items()}})
        print(f"alpha {alpha:.2f}  objective {score:.4f}  " + "  ".join(f"{l} {r['bpb']:.4f}" for l, r in res.items() if r), flush=True)
        if best is None or score < best[0]:
            best = (score, alpha, {k: v.cpu() for k, v in sd.items()})
    print("\nbest alpha:", best[1], "objective", round(best[0], 4))
    if a.save_best:
        tuned.load_state_dict(best[2], strict=False)
        tuned.save_pretrained(a.save_best, safe_serialization=True, max_shard_size="5GB"); tok.save_pretrained(a.save_best)
        Path(a.save_best, "interpolation.json").write_text(json.dumps({"alpha": best[1], "results": results}, indent=2))
        print("saved", a.save_best)


if __name__ == "__main__":
    main()
