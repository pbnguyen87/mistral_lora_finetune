"""Extend Mistral's SentencePiece BPE vocabulary with Vietnamese pieces (PLAN.md §2).

Method (the Chinese-LLaMA / Vistral approach): train a fresh SentencePiece BPE on Vietnamese text, take the
pieces Mistral does not have, append them to Mistral's model proto after id 31,999 so every existing id is
unchanged, pad the vocabulary to a multiple of --pad-to, and write a Hugging Face tokenizer directory.

Only pieces containing a Vietnamese-specific letter are appended by default, so English, French, German,
Spanish and Italian text is segmented exactly as before (verified by test_tokenizer.py). Pass
--allow-ascii-terms FILE to also add ASCII loanwords / technology terms as whole pieces (this *does*
change their segmentation in English text; keep it off unless that is what you want).

    python tokenizer/extend_tokenizer.py \\
        --mistral-tokenizer ../Confucius4-TTS/checkpoints \\
        --corpus data/raw/vi/culturax_vi/shard-0000*.jsonl.gz ../code-switched_datasets/text/uit_visfd/ViSFD.csv \\
        --num-new 8000 --out tokenizer/out/mistral_vi
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlf_common import MISTRAL_VOCAB, has_vietnamese_letter, read_text_column  # noqa: E402


def collect_sentences(patterns: list[str], max_sentences: int, seed: int) -> Path:
    files: list[str] = []
    for pat in patterns:
        files += sorted(glob.glob(pat))
    if not files:
        sys.exit(f"[extend] no corpus files match {patterns}")
    tmp = Path(tempfile.mkdtemp(prefix="spm_")) / "sentences.txt"
    n = 0
    with open(tmp, "w", encoding="utf-8") as out:
        for f in files:
            for text in read_text_column(f):
                for sent in text.replace("\r", "").split("\n"):
                    sent = sent.strip()
                    if len(sent) < 10 or not has_vietnamese_letter(sent):
                        continue
                    out.write(sent[:2000] + "\n"); n += 1
                    if n >= max_sentences:
                        break
                if n >= max_sentences:
                    break
            if n >= max_sentences:
                break
    print(f"[extend] {n:,} Vietnamese sentences from {len(files)} files -> {tmp}")
    return tmp


def train_spm(sentences: Path, vocab_size: int, prefix: Path) -> Path:
    import sentencepiece as spm

    spm.SentencePieceTrainer.train(
        input=str(sentences), model_prefix=str(prefix), model_type="bpe", vocab_size=vocab_size,
        character_coverage=0.9999, split_digits=True, byte_fallback=False, normalization_rule_name="identity",
        add_dummy_prefix=True, remove_extra_whitespaces=False, num_threads=8, input_sentence_size=3_000_000,
        shuffle_input_sentence=True, max_sentence_length=4192, train_extremely_large_corpus=False,
    )
    return prefix.with_suffix(".model")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mistral-tokenizer", required=True, help="dir containing Mistral v0.1 tokenizer.model (+ tokenizer_config.json)")
    ap.add_argument("--corpus", nargs="+", required=True, help="Vietnamese text files/globs (.txt/.jsonl(.gz)/.csv/.parquet)")
    ap.add_argument("--num-new", type=int, default=8000, help="Vietnamese pieces to append (before padding)")
    ap.add_argument("--spm-vocab", type=int, default=24000, help="vocab of the temporary Vietnamese SentencePiece model")
    ap.add_argument("--max-sentences", type=int, default=2_000_000)
    ap.add_argument("--pad-to", type=int, default=128)
    ap.add_argument("--allow-ascii-terms", default=None, help="file with one ASCII term per line to add as whole pieces")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from sentencepiece import sentencepiece_model_pb2 as pb
    import sentencepiece as spm

    src_dir = Path(a.mistral_tokenizer)
    base_model = src_dir / "tokenizer.model"
    if not base_model.is_file():
        sys.exit(f"[extend] {base_model} not found")
    base = pb.ModelProto(); base.ParseFromString(base_model.read_bytes())
    base_pieces = {p.piece for p in base.pieces}
    if len(base.pieces) != MISTRAL_VOCAB:
        print(f"[extend] WARNING base vocab is {len(base.pieces)}, expected {MISTRAL_VOCAB}")

    sentences = collect_sentences(a.corpus, a.max_sentences, a.seed)
    tmp_prefix = sentences.parent / "vi_spm"
    vi_model = train_spm(sentences, a.spm_vocab, tmp_prefix)
    vi = pb.ModelProto(); vi.ParseFromString(vi_model.read_bytes())

    # candidate pieces in score order (SentencePiece BPE scores = merge priority, higher first)
    cands = [p for p in vi.pieces if p.type == pb.ModelProto.SentencePiece.NORMAL and p.piece not in base_pieces]
    cands.sort(key=lambda p: -p.score)
    new_pieces = [p.piece for p in cands if has_vietnamese_letter(p.piece)][: a.num_new]
    ascii_terms = []
    if a.allow_ascii_terms:
        for line in Path(a.allow_ascii_terms).read_text(encoding="utf-8").splitlines():
            t = line.strip()
            if t and ("▁" + t) not in base_pieces and t not in base_pieces:
                ascii_terms.append("▁" + t)
    added = new_pieces + ascii_terms
    n_pad = (-(len(base.pieces) + len(added))) % a.pad_to
    pads = [f"<pad_extra_{i}>" for i in range(n_pad)]

    out = pb.ModelProto(); out.CopyFrom(base)
    for piece in added:
        sp = out.pieces.add(); sp.piece = piece; sp.score = 0.0; sp.type = pb.ModelProto.SentencePiece.NORMAL
    for piece in pads:
        sp = out.pieces.add(); sp.piece = piece; sp.score = 0.0; sp.type = pb.ModelProto.SentencePiece.UNUSED
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tokenizer.model").write_bytes(out.SerializeToString())
    # Build the fast tokenizer explicitly. Converting a bare .model drops the leading "▁" on the first word
    # (shipped Mistral tokenizer.json uses Metaspace prepend_scheme="first"); add_prefix_space=True restores it,
    # which test_tokenizer.py verifies by requiring identical ids on European text.
    from transformers import LlamaTokenizerFast

    src_cfg = json.loads((src_dir / "tokenizer_config.json").read_text(encoding="utf-8")) if (src_dir / "tokenizer_config.json").is_file() else {}
    fast = LlamaTokenizerFast(vocab_file=str(out_dir / "tokenizer.model"), legacy=False, add_prefix_space=True,
                              bos_token=src_cfg.get("bos_token", "<s>"), eos_token=src_cfg.get("eos_token", "</s>"),
                              unk_token=src_cfg.get("unk_token", "<unk>"), add_bos_token=src_cfg.get("add_bos_token", True),
                              add_eos_token=src_cfg.get("add_eos_token", False), clean_up_tokenization_spaces=False)
    fast.save_pretrained(out_dir)

    report = {"base_vocab": len(base.pieces), "vi_spm_vocab": len(vi.pieces), "candidates": len(cands),
              "added_vietnamese_pieces": len(new_pieces), "added_ascii_terms": len(ascii_terms), "padding": n_pad,
              "new_vocab": len(out.pieces), "first_new_id": len(base.pieces),
              "examples": new_pieces[:25]}
    (out_dir / "extension_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    sp_check = spm.SentencePieceProcessor(model_file=str(out_dir / "tokenizer.model"))
    assert sp_check.get_piece_size() == len(out.pieces)
    print(f"[extend] vocab {len(base.pieces)} -> {len(out.pieces)} (+{len(new_pieces)} Vietnamese, +{len(ascii_terms)} ASCII, +{n_pad} pad)")
    print(f"[extend] examples: {new_pieces[:15]}")
    print(f"[extend] wrote {out_dir}  (run tokenizer/test_tokenizer.py next)")


if __name__ == "__main__":
    main()
