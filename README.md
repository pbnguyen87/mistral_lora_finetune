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

## What each step is for

| Step | Purpose | Reads | Writes |
|---|---|---|---|
| 1. `data/download_corpora.py` | Pull only the language slices named in `mix.yaml` (Vietnamese from CulturaX, CulturaY, Wikipedia; English and fr/de/es/it as replay) up to each source's cap, and copy the local code-switched text. Streaming, resumable per source. | Hugging Face, local files | `data/raw/<lang>/<source>/*.jsonl.gz` |
| 2. `tokenizer/extend_tokenizer.py` + `test_tokenizer.py` | Give Mistral a Vietnamese vocabulary: discover frequent Vietnamese pieces on a sample of every Vietnamese register and append them after id 31,999, so the 32,000 pretrained rows stay valid and Vietnamese drops from ~2.75 to ~1.1 tokens per syllable. The test proves European text and English spans inside Vietnamese are segmented exactly as before, which is what keeps the replay data and code-switching meaningful. | step 1 (vi sources), Mistral tokenizer | `tokenizer/out/mistral_vi/` |
| 3. `data/filter_and_pack.py` | Make training blocks: drop short, non-text, repetitive or misfiled documents (plus optional MinHash dedup for the small local sources), mask emails and phone numbers, divert the first documents of each language into a held-out set that training never sees, then tokenize with the new tokenizer, join documents with EOS and cut into 4,096-token blocks, one dataset per source so they can be mixed by weight. | step 1, step 2 | `data/heldout/<lang>.jsonl`, `data/packed/<lang>/<source>/` |
| 4. `train/init_new_rows.py` | Resize Mistral-7B to the new vocabulary and give every new embedding and lm_head row a sensible start: the mean of the Mistral rows of its sub-pieces, which already lives in Mistral's space. `eval_bpb.py` on this checkpoint gives the reference bits-per-byte per language before any training. | Mistral-7B-v0.1, step 2 | `checkpoints/mistral-7b-vi-init/` |
| 5. Stage 1, `train/train.py` + `export_hf.py` | Train only the new rows (body, projector-free here, all frozen; old rows gradient-masked) so the new tokens settle before they can push large gradients through the network. Export merges the rows into a full checkpoint. | step 3, step 4 | `runs/stage1/`, `checkpoints/mistral-7b-vi-stage1/` |
| 6. Stage 2, `train/train.py` + `export_hf.py` | The continual pretraining proper: LoRA on attention and MLP projections plus the new rows, on the 70/20/10 mix, with per-language bits-per-byte logged every `eval_steps` as the forgetting guard. Export merges the adapter. | step 3, step 5 | `runs/stage2/`, `checkpoints/mistral-7b-vi-final/` |
| 7. `eval/eval_bpb.py`, `drift_vs_base.py`, `interpolate.py` | Judge the result: Vietnamese bits-per-byte should fall well below step 4's reference while English and the European languages rise by no more than ~5 %; drift confirms the 32,000 original rows are untouched; interpolation with the base model recovers original ability if stage 2 overshot. | steps 4, 6 | `runs/*.json`, optional `checkpoints/mistral-7b-vi-interp/` |
| 8. `Confucius4-TTS/vistral_finetune/build_hybrid_embedding.py` | Hand the new embedding table to the TTS project in place of Vistral's: the hybrid build keeps Confucius4's Mistral rows and takes only the new Vietnamese rows from this checkpoint. | step 6 or 7 | Confucius4 hybrid T2S checkpoint |

## Run order (GPU box)

```bash
MIS=../Confucius4-TTS/checkpoints                          # Mistral v0.1 tokenizer lives here

# 1. data
python data/download_corpora.py --mix data/mix.yaml

# 2. tokenizer (+8k Vietnamese pieces) and acceptance tests
#    Vietnamese sources only (the extension never adds pieces without Vietnamese letters). A sample of each
#    register is enough: formal web (CulturaX, CulturaY), reference prose (Wikipedia), colloquial /
#    code-switched (spoken_cs_vi = transcripts, ViMedCSS, ViSFD, ViSpamReviews). --max-sentences caps the total.
python tokenizer/extend_tokenizer.py --mistral-tokenizer $MIS \
    --corpus "data/raw/vi/culturax_vi/shard-0000*.jsonl.gz" \
             "data/raw/vi/culturay_vi/shard-0000[0-3]*.jsonl.gz" \
             "data/raw/vi/wiki_vi/shard-0000*.jsonl.gz" \
             "data/raw/vi/spoken_cs_vi/*.jsonl.gz" \
    --num-new 8000 --max-sentences 2000000 --out tokenizer/out/mistral_vi
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
