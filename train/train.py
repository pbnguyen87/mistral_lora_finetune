"""Continual pretraining of the vocabulary-extended Mistral (PLAN.md §3).

  --stage 1  new embedding + lm_head rows only; everything else frozen (gradient hook masks old rows).
  --stage 2  LoRA on the attention/MLP projections + the same new rows.

Data: packed blocks from data/filter_and_pack.py, mixed by the `weight` of each source in mix.yaml.
Eval: bits-per-byte per language on data/heldout/<lang>.jsonl every `eval_steps` (forgetting guard).
Saves: `new_rows.safetensors` (embed + head rows >= first_new_id) and, for stage 2, the LoRA adapter.
Both are small; the full model is reassembled by export/export_hf.py.

    python train/train.py --config train/configs/stage1.yaml
    python train/train.py --config train/configs/stage2.yaml --override training.max_steps=2000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlf_common import ROOT, bits_per_byte, iter_jsonl, load_yaml  # noqa: E402


# --------------------------------------------------------------------------- data
def build_train_dataset(cfg: dict, seed: int):
    from datasets import concatenate_datasets, interleave_datasets, load_from_disk

    mix = load_yaml(ROOT / cfg["data"]["mix"])
    packed = (ROOT / mix.get("packed_dir", "data/packed")).resolve()
    parts, weights, names = [], [], []
    for src in mix["sources"]:
        d = packed / src["lang"] / src["name"]
        if not (d / "dataset_info.json").exists():
            print(f"[data] missing packed source {src['name']} ({d}), skipped")
            continue
        ds = load_from_disk(str(d))
        parts.append(ds); weights.append(float(src["weight"])); names.append(f"{src['name']}:{len(ds)}")
    if not parts:
        sys.exit("[data] no packed sources found; run data/filter_and_pack.py")
    tot = sum(weights); probs = [w / tot for w in weights]
    print("[data] sources:", ", ".join(names))
    print("[data] probabilities:", [round(p, 3) for p in probs])
    ds = interleave_datasets(parts, probabilities=probs, seed=seed, stopping_strategy="all_exhausted") if len(parts) > 1 else parts[0]
    return ds.with_format("torch")


def collate(features):
    import torch

    ids = torch.stack([f["input_ids"] for f in features]).long()
    return {"input_ids": ids, "labels": ids.clone()}


# --------------------------------------------------------------------------- model
def set_trainable(model, stage: int, first_new: int, lora_cfg: dict | None):
    import torch

    base = model
    if stage == 2:
        from peft import LoraConfig, get_peft_model

        lc = LoraConfig(r=int(lora_cfg.get("r", 64)), lora_alpha=int(lora_cfg.get("alpha", 128)),
                        lora_dropout=float(lora_cfg.get("dropout", 0.05)), bias="none", task_type="CAUSAL_LM",
                        target_modules=list(lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])))
        model = get_peft_model(model, lc)
        base = model.get_base_model()
    else:
        for p in model.parameters():
            p.requires_grad = False

    emb_w = base.get_input_embeddings().weight
    head_w = base.get_output_embeddings().weight
    emb_w.requires_grad = True

    def mask_old(grad):
        g = grad.clone(); g[:first_new] = 0; return g

    emb_w.register_hook(mask_old)
    if head_w.data_ptr() != emb_w.data_ptr():
        head_w.requires_grad = True
        head_w.register_hook(mask_old)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_eff = n_train - (first_new * emb_w.shape[1]) * (1 if head_w.data_ptr() == emb_w.data_ptr() else 2)
    print(f"[model] stage {stage}: trainable tensors {n_train:,} params; effective after old-row mask: {n_eff:,}")
    return model, base


# --------------------------------------------------------------------------- eval
class BPBEvaluator:
    def __init__(self, heldout_dir: Path, tok, langs, max_docs: int, seq_len: int):
        self.sets = {}
        for lang in langs:
            p = heldout_dir / f"{lang}.jsonl"
            if not p.exists():
                continue
            texts = [r["text"] for _, r in zip(range(max_docs), iter_jsonl(p))]
            if texts:
                self.sets[lang] = texts
        self.tok, self.seq_len = tok, seq_len

    def run(self, model, device) -> dict:
        import torch

        model.eval(); out = {}
        with torch.no_grad():
            for lang, texts in self.sets.items():
                nats, nbytes = 0.0, 0
                for t in texts:
                    ids = self.tok(t, add_special_tokens=False, truncation=True, max_length=self.seq_len)["input_ids"]
                    if len(ids) < 2:
                        continue
                    x = torch.tensor([ids], device=device)
                    loss = model(input_ids=x, labels=x).loss.item()  # mean nats/token over len-1 predictions
                    nats += loss * (len(ids) - 1)
                    nbytes += len(self.tok.decode(ids[1:]).encode("utf-8"))
                out[lang] = round(bits_per_byte(nats, max(nbytes, 1)), 4)
        model.train()
        return out


# --------------------------------------------------------------------------- saving
def save_new_rows(base, first_new: int, out_dir: Path):
    import torch
    from safetensors.torch import save_file

    emb = base.get_input_embeddings().weight.detach()[first_new:].cpu().contiguous()
    head = base.get_output_embeddings().weight.detach()[first_new:].cpu().contiguous()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file({"embed_new": emb, "head_new": head}, str(out_dir / "new_rows.safetensors"), metadata={"first_new_id": str(first_new)})


# --------------------------------------------------------------------------- main
def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    for ov in overrides or []:
        k, v = ov.split("=", 1)
        d = cfg
        for part in k.split(".")[:-1]:
            d = d.setdefault(part, {})
        try:
            v = json.loads(v)
        except Exception:
            pass
        d[k.split(".")[-1]] = v
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", help="dotted.key=value (JSON values ok)")
    a = ap.parse_args()
    cfg = apply_overrides(load_yaml(a.config), a.override)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

    t = cfg["training"]; stage = int(cfg["stage"]); first_new = int(cfg["first_new_id"])
    out_dir = (ROOT / t["output_dir"]).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and t.get("bf16", True)) else torch.float32

    tok = AutoTokenizer.from_pretrained(cfg["model_path"], token=False)
    model = AutoModelForCausalLM.from_pretrained(cfg["model_path"], torch_dtype=dtype, token=False,
                                                 attn_implementation=t.get("attn_implementation", None))
    if t.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    model, base = set_trainable(model, stage, first_new, cfg.get("lora"))

    train_ds = build_train_dataset(cfg, int(t.get("seed", 42)))
    mix = load_yaml(ROOT / cfg["data"]["mix"])
    evaluator = BPBEvaluator((ROOT / mix.get("heldout_dir", "data/heldout")).resolve(), tok, cfg["data"].get("eval_langs", ["vi", "en"]),
                             int(cfg["data"].get("eval_docs_per_lang", 50)), int(mix.get("seq_len", 4096)))
    log_path = out_dir / "bpb_log.jsonl"

    class EvalAndSave(TrainerCallback):
        def on_step_end(self, args, state, control, model=None, **kw):
            if state.global_step % int(t.get("eval_steps", 200)) == 0 and evaluator.sets:
                res = evaluator.run(model, model.device)
                res.update({"step": state.global_step, "time": time.time()})
                print(f"[eval] step {state.global_step}: {res}", flush=True)
                with open(log_path, "a") as f:
                    f.write(json.dumps(res) + "\n")
            if state.global_step % int(t.get("save_steps", 500)) == 0 and state.global_step > 0:
                ck = out_dir / f"step_{state.global_step:07d}"
                save_new_rows(base, first_new, ck)
                if stage == 2:
                    model.save_pretrained(str(ck))  # adapter only
                print(f"[save] {ck}", flush=True)

    args = TrainingArguments(
        output_dir=str(out_dir / "hf_trainer"), per_device_train_batch_size=int(t.get("micro_batch", 1)),
        gradient_accumulation_steps=int(t.get("grad_accum", 16)), learning_rate=float(t["lr"]),
        lr_scheduler_type=t.get("scheduler", "cosine"), warmup_ratio=float(t.get("warmup_ratio", 0.02)),
        max_steps=int(t["max_steps"]), weight_decay=float(t.get("weight_decay", 0.1)), adam_beta1=0.9, adam_beta2=0.95,
        max_grad_norm=float(t.get("grad_clip", 1.0)), bf16=(dtype == torch.bfloat16), logging_steps=int(t.get("log_steps", 10)),
        save_strategy="no", report_to=t.get("report_to", "none"), dataloader_num_workers=int(t.get("num_workers", 2)),
        seed=int(t.get("seed", 42)), remove_unused_columns=False, gradient_checkpointing=False,
    )
    if evaluator.sets:
        res = evaluator.run(model, model.device); res["step"] = 0
        print(f"[eval] step 0: {res}", flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(res) + "\n")
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, data_collator=collate, callbacks=[EvalAndSave()])
    trainer.train()
    final = out_dir / "final"
    save_new_rows(base, first_new, final)
    if stage == 2:
        model.save_pretrained(str(final))
    (final / "train_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"[done] stage {stage} -> {final}")


if __name__ == "__main__":
    main()
