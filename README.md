# mistral_lora_finetune

Vietnamese continual pretraining of Mistral-7B v0.1 with LoRA, producing a Vietnamese-extended
embedding table for `Confucius4-TTS/vistral_finetune/` and a Vietnamese LM as a by-product.
Rationale, data mix, recipe and the anti-forgetting controls are in `PLAN.md`.

Status: scripts complete and smoke-tested end to end on CPU with a tiny Mistral-shaped model
(`./smoke_test.sh`). Nothing has been trained at scale.

```
mistral_lora_finetune/
├── PLAN.md
├── requirements.txt                 (local .venv: torch 2.2.2 → transformers 4.46.3, numpy<2)
├── mlf_common.py                    shared helpers
├── data/
│   ├── mix.yaml                     sources, weights (70 % vi / 20 % en / 10 % fr-de-es-it), caps, held-out sizes
│   ├── download_corpora.py          stream HF sources + copy local files -> data/raw/<lang>/<source>/*.jsonl.gz
│   └── filter_and_pack.py           quality filters, PII mask, optional MinHash dedup, held-out split, pack to seq_len
├── tokenizer/
│   ├── extend_tokenizer.py          +N Vietnamese pieces appended to Mistral's SentencePiece model, ids 0..31999 unchanged
│   └── test_tokenizer.py            acceptance: shared ids, EU text identical, tokens/syllable, English spans intact
├── train/
│   ├── init_new_rows.py             resize to the new vocab, init new embed/head rows from Mistral sub-piece means
│   ├── train.py                     --stage 1 (new rows only) / --stage 2 (LoRA + new rows); per-language bits-per-byte eval
│   └── configs/stage1.yaml, stage2.yaml
├── eval/
│   ├── eval_bpb.py                  bits-per-byte per language for any checkpoint (+adapter, +new rows)
│   ├── drift_vs_base.py             cosine of shared rows vs base, new-row norms
│   └── interpolate.py               base/tuned weight interpolation sweep with BPB per alpha
├── export/export_hf.py              init + new_rows (+ merged LoRA) -> full HF checkpoint with safetensors index
└── smoke_test.sh                    whole pipeline on a tiny model, CPU, a few minutes
```

## Setup

```bash
cd mistral_lora_finetune
uv venv --python 3.11 .venv && source .venv/bin/activate
pip install torch            # CUDA build on the GPU box
pip install -r requirements.txt
export HF_TOKEN=...          # only needed for the gated CulturaX / CulturaY downloads
./smoke_test.sh              # optional: verifies the environment in a few minutes
```

## Run order (GPU box)

```bash
MIS=../Confucius4-TTS/checkpoints                          # Mistral v0.1 tokenizer lives here

# 1. data
python data/download_corpora.py --mix data/mix.yaml

# 2. tokenizer (+8k Vietnamese pieces) and acceptance tests
python tokenizer/extend_tokenizer.py --mistral-tokenizer $MIS \
    --corpus "data/raw/vi/culturax_vi/shard-0000*.jsonl.gz" ../code-switched_datasets/text/uit_visfd/ViSFD.csv \
    --num-new 8000 --out tokenizer/out/mistral_vi
python tokenizer/test_tokenizer.py --mistral-tokenizer $MIS --new tokenizer/out/mistral_vi

# 3. filter, hold out, pack (uses the new tokenizer)
python data/filter_and_pack.py --mix data/mix.yaml --tokenizer tokenizer/out/mistral_vi
python data/filter_and_pack.py --mix data/mix.yaml --tokenizer tokenizer/out/mistral_vi --only spoken_cs_vi --dedup

# 4. base model + new rows
python train/init_new_rows.py --base mistralai/Mistral-7B-v0.1 --base-tokenizer $MIS \
    --new-tokenizer tokenizer/out/mistral_vi --out checkpoints/mistral-7b-vi-init
python eval/eval_bpb.py --model checkpoints/mistral-7b-vi-init --out runs/bpb_init.json      # reference numbers

# 5. stage 1: new rows only
python train/train.py --config train/configs/stage1.yaml
python export/export_hf.py --model checkpoints/mistral-7b-vi-init --new-rows runs/stage1/final/new_rows.safetensors \
    --out checkpoints/mistral-7b-vi-stage1

# 6. stage 2: LoRA + new rows
python train/train.py --config train/configs/stage2.yaml
python export/export_hf.py --model checkpoints/mistral-7b-vi-stage1 --new-rows runs/stage2/final/new_rows.safetensors \
    --adapter runs/stage2/final --out checkpoints/mistral-7b-vi-final

# 7. evaluate, check forgetting, optionally interpolate with the base
python eval/eval_bpb.py --model checkpoints/mistral-7b-vi-final --out runs/bpb_final.json
python eval/drift_vs_base.py --base mistralai/Mistral-7B-v0.1 --tuned checkpoints/mistral-7b-vi-final --tokenizer checkpoints/mistral-7b-vi-final
python eval/interpolate.py --base mistralai/Mistral-7B-v0.1 --tuned checkpoints/mistral-7b-vi-final --alphas 0.5 0.7 0.85 1.0 \
    --save-best checkpoints/mistral-7b-vi-interp

# 8. hand the table to Confucius4-TTS
cd ../Confucius4-TTS && python vistral_finetune/build_hybrid_embedding.py --vistral-repo ../mistral_lora_finetune/checkpoints/mistral-7b-vi-final
```

## Notes

- `train.py` keeps the 32,000 Mistral rows of the embedding and lm_head bit-identical through both stages
  (gradient hook); `eval/drift_vs_base.py` confirms it. Stage 1 must keep `weight_decay: 0`.
- Memory on one A100 80 GB at 4,096 tokens per micro-batch with gradient checkpointing: ~22 GB for stage 2.
  Note the optimizer keeps state for the whole embedding/head tensors (~4 GB for 7B); PEFT's
  `trainable_token_indices` is the alternative if that matters.
- `tokenizer/extend_tokenizer.py` adds only pieces containing Vietnamese letters, so European text is
  segmented exactly as before. `--allow-ascii-terms` can add loanwords as whole pieces, at the cost of
  changing their segmentation in English.
- Training cost at ~40 % MFU: 1 B tokens ≈ 95 GPU-hours (4 days on one A100); 5 B ≈ 470; 20 B ≈ 1,900.
- Scripts set `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` because a stale `~/.cache/huggingface/token` breaks public
  downloads; `HF_TOKEN` in the environment still works for gated datasets.
