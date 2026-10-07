# Plan: Vietnamese continual pretraining of Mistral-7B v0.1 with LoRA

Goal: a Vietnamese-extended Mistral-7B whose embedding table can replace Vistral's in
`Confucius4-TTS/vistral_finetune/`, with a usable Vietnamese LM as a by-product. Secondary goal: control
over the tokenizer merges (keep English spans and technology loanwords as whole pieces) and over licensing.

Decision rule before starting: run `vistral_finetune` Stage A/B on the existing Vistral rows first. Only
if its benchmark shows the text front end is the bottleneck is this project worth the GPU time.

## 1. Data

| Source | Vietnamese size (verified) | Role |
|---|---|---|
| CulturaX `vi` (uonlp/CulturaX) | 57.6 M docs, 55.4 B tokens, 0.88 % of corpus | bulk web text, MinHash-deduplicated |
| CulturaY `vi` (ontocord/CulturaY) | 4.5 M docs, 17 GB, 0.38 % of corpus | second web source (HPLT / Internet Archive) |
| Vietnamese Wikipedia | ~1 B tokens | clean reference prose |
| Own podcast transcripts, ViMedCSS text, UIT-ViSFD, ViSpamReviews | tens of M tokens | spoken register + natural code-switching, upsample 3-5x |
| English replay (FineWeb-Edu sample) + CulturaX fr/de/es/it | 30 % of the mix | anti-forgetting for the languages Mistral was built for |

Mix: 70 % Vietnamese, 20 % English, 10 % fr/de/es/it. Processing: second language-ID pass, perplexity
filter with a small Vietnamese LM, cross-source near-dedup, PII scrub, pack to 4,096 tokens. Hold out 50 M
Vietnamese tokens and 20 M tokens per replay language before training. Check CulturaX/CulturaY terms.

## 2. Tokenizer extension

Train SentencePiece BPE on a 3-5 GB Vietnamese sample (include the IT glossary and top ViMedCSS insertions);
keep 6-10 k pieces not already in Mistral's 32,000; append after id 31,999; pad vocab to a multiple of 128.
Acceptance: Vietnamese <= 1.15 tokens/syllable, English unchanged (1.23 tokens/word), English spans inside
Vietnamese sentences still segmented as English pieces. New input-embedding and lm_head rows initialised from
the mean of each piece's Mistral sub-piece rows.

## 3. Training

| Stage | Trainable | Tokens | LR | Purpose |
|---|---|---|---|---|
| 1 | new embedding + head rows only (PEFT `trainable_token_indices`) | 0.5-1 B | 1e-4 const | settle new tokens, body frozen |
| 2 | LoRA r=64 on q,k,v,o,gate,up,down + new rows | 1-5 B (single A100) / 5-20 B (8 GPUs) | 2e-5 peak, cosine to 1e-5, 2 % warmup | continual pretraining |
| 3 (optional) | LoRA | 50-100 k examples | 1e-5 | instruction tuning, only if a chat model is wanted |

Fixed: bf16, seq 4,096 packed, global batch 2-4 M tokens, AdamW (0.9, 0.95), wd 0.1, clip 1.0,
FlashAttention, gradient checkpointing. Original 32,000 embedding and head rows frozen throughout.

Memory (A100 80 GB, 4 k tokens/micro-batch, checkpointing): bf16 base 14.5 GB + LoRA state 2.5 GB +
activations 3.5 GB + overhead 1.5 GB ~= 22 GB; `trainable_token_indices` keeps optimizer state only for the
new rows. Micro-batch of 8 x 4,096 fits.

Compute at ~40 % MFU: 1 B tokens ~95 GPU-h (4 days on 1 A100); 5 B ~470 GPU-h; 20 B ~1,900 GPU-h.

## 4. Keeping the original languages

1. Replay (above) -- mandatory for a language shift this large.
2. Low LR, staged start, original embedding/head rows frozen.
3. Per-language held-out bits-per-byte every 500 steps (en, fr, de, es, it, vi); forgetting budget <= 3-5 %
   rise on any language; select the checkpoint on the combined objective. Add HellaSwag/ARC (en) and Belebele
   at a few checkpoints.
4. Post-hoc interpolation with the base weights, alpha 0.5-0.9 sweep (new rows kept as is); LoRA can instead be
   scaled down.
5. Optional KL distillation from the frozen base on replay batches.

## 5. Evaluation and hand-off

Success: Vietnamese bits-per-byte well below base; English BPB rise <= 5 %; tokens/syllable <= 1.15; optional
VMLU zero-shot for a public comparison with Vistral and Qwen3. Then run
`Confucius4-TTS/vistral_finetune/check_projection.py` on the new table and the shared-row drift measurement.
Save in HF format with a safetensors index so `build_hybrid_embedding.py --vistral-repo <path>` consumes it.

## 6. Tooling candidates (all verified public)

PEFT + TRL (`trainable_token_indices` matches Stage 1 exactly), Unsloth (fastest single-GPU, has a
continued-pretraining recipe for Mistral with embed_tokens/lm_head), Axolotl (YAML, `lora_modules_to_save`),
LLaMA-Factory (pretraining stage + `resize_vocab`), torchtune (`recipes/configs/mistral/7B_lora.yaml`).

## 7. Risks

Forgetting (replay + eval guard); web-text quality (filters, upsampled clean sources); tokenizer merges that
split English spans (acceptance test); data licences (research-oriented); cost vs. return for TTS (decision
rule at the top).
