#!/usr/bin/env bash
# End-to-end smoke test on CPU with a tiny Mistral-shaped model (trl-internal-testing/tiny-MistralForCausalLM-0.2,
# hidden 8, 32,000-piece Mistral tokenizer) and a few MB of local Vietnamese/English text.
# Exercises: tokenizer extension + tests, init of new rows, filter/pack, stage 1, export, stage 2, export,
# eval_bpb, drift, interpolation. Takes a few minutes. Does NOT download CulturaX.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
MIS=${MISTRAL_TOKENIZER:-../Confucius4-TTS/checkpoints}
S=smoke; rm -rf $S; mkdir -p $S/raw/vi/local_vi $S/raw/en/local_en

echo "== tokenizer extension (tiny: 300 new pieces)"
$PY tokenizer/extend_tokenizer.py --mistral-tokenizer $MIS \
  --corpus ../code-switched_datasets/text/uit_visfd/ViSFD.csv "../code-switched_datasets/text/vimedcss_metadata/ViMedCSS-Metadata/*.csv" \
  --num-new 300 --spm-vocab 3000 --max-sentences 60000 --out $S/tok
$PY tokenizer/test_tokenizer.py --mistral-tokenizer $MIS --new $S/tok --max-tps 2.6 --no-strict

echo "== tiny base model + new rows"
$PY train/init_new_rows.py --base trl-internal-testing/tiny-MistralForCausalLM-0.2 --base-tokenizer $MIS \
  --new-tokenizer $S/tok --out $S/init --dtype fp32

echo "== tiny data: raw shards -> heldout + packed (seq 128)"
cat > $S/mix.yaml <<EOF
raw_dir: $S/raw
packed_dir: $S/packed
heldout_dir: $S/heldout
seq_len: 128
heldout_tokens: {vi: 3000, en: 3000}
sources:
  - {name: local_vi, lang: vi, weight: 0.7, local: ["../code-switched_datasets/text/uit_visfd/ViSFD.csv"]}
  - {name: local_en, lang: en, weight: 0.3, local: ["../Confucius4-TTS/README.md", "../ZONOS2/README.md"]}
EOF
$PY data/download_corpora.py --mix $S/mix.yaml
$PY data/filter_and_pack.py --mix $S/mix.yaml --tokenizer $S/tok --min-chars 40 --dedup

echo "== stage 1 (new rows only)"
$PY train/train.py --config train/configs/stage1.yaml --override stage=1 model_path=$S/init data.mix=$S/mix.yaml \
  "data.eval_langs=[\"vi\",\"en\"]" data.eval_docs_per_lang=5 training.output_dir=$S/runs/stage1 training.max_steps=6 \
  training.grad_accum=2 training.eval_steps=3 training.save_steps=6 training.bf16=false training.gradient_checkpointing=false training.num_workers=0
$PY export/export_hf.py --model $S/init --new-rows $S/runs/stage1/final/new_rows.safetensors --out $S/stage1

echo "== stage 2 (LoRA + new rows)"
$PY train/train.py --config train/configs/stage2.yaml --override stage=2 model_path=$S/stage1 data.mix=$S/mix.yaml \
  "data.eval_langs=[\"vi\",\"en\"]" data.eval_docs_per_lang=5 training.output_dir=$S/runs/stage2 training.max_steps=6 \
  training.grad_accum=2 training.eval_steps=3 training.save_steps=6 training.bf16=false training.gradient_checkpointing=false training.num_workers=0 \
  lora.r=4 lora.alpha=8
$PY export/export_hf.py --model $S/stage1 --new-rows $S/runs/stage2/final/new_rows.safetensors --adapter $S/runs/stage2/final --out $S/final

echo "== eval / drift / interpolation"
$PY eval/eval_bpb.py --model $S/final --heldout-dir $S/heldout --langs vi en --max-docs 5 --seq-len 128
$PY eval/drift_vs_base.py --base $S/init --tuned $S/final --tokenizer $S/final
$PY eval/interpolate.py --base $S/init --tuned $S/final --alphas 0.5 1.0 --langs vi en --heldout-dir $S/heldout --max-docs 5 --seq-len 128 --objective vi:1,en:1
echo "SMOKE TEST PASSED"
