"""Acceptance tests for an extended tokenizer (PLAN.md §2).

  1. ids 0..31,999 map to the same pieces as Mistral's tokenizer;
  2. English / French / German / Spanish / Italian samples tokenize to identical ids (segmentation unchanged);
  3. Vietnamese tokens per syllable <= --max-tps (default 1.15);
  4. English spans inside Vietnamese sentences tokenize exactly as they do standalone.

    python tokenizer/test_tokenizer.py --mistral-tokenizer ../Confucius4-TTS/checkpoints --new tokenizer/out/mistral_vi
Exit code 1 on failure unless --no-strict.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlf_common import MISTRAL_VOCAB, tokens_per_syllable  # noqa: E402

VI = [
    "Hôm nay trời đẹp, chúng ta đi dạo một vòng quanh hồ nhé.",
    "Nguyễn Văn Đức đã nghiên cứu những vấn đề phức tạp về trí tuệ nhân tạo.",
    "Bệnh cường giáp là một rối loạn nội tiết xảy ra khi tuyến giáp sản xuất quá mức hormone thyroxine.",
    "Mô hình mới có thể tạo giọng nói tiếng Việt tự nhiên hơn, kể cả khi câu có chèn từ tiếng Anh.",
    "Chính phủ vừa ban hành nghị định mới về quản lý dữ liệu cá nhân trên không gian mạng.",
]
EU = {
    "en": "The new model can generate natural speech even when the sentence contains technical words such as database, server, or deadline.",
    "fr": "Le nouveau modèle peut générer une parole naturelle même lorsque la phrase contient des mots techniques.",
    "de": "Das neue Modell kann natürliche Sprache erzeugen, auch wenn der Satz technische Wörter enthält.",
    "es": "El nuevo modelo puede generar habla natural incluso cuando la oración contiene palabras técnicas.",
    "it": "Il nuovo modello può generare un parlato naturale anche quando la frase contiene parole tecniche.",
}
CS = [("Bạn nhớ check mail và update lại file báo cáo trước deadline chiều nay nhé.", ["check mail", "update", "file", "deadline"]),
      ("Team đang train model deep learning trên server mới.", ["Team", "train model deep learning", "server"]),
      ("Giá iPhone 15 Pro giảm mạnh, camera và pin được review rất tốt.", ["iPhone 15 Pro", "camera", "review"])]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mistral-tokenizer", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--max-tps", type=float, default=1.15)
    ap.add_argument("--no-strict", action="store_true")
    a = ap.parse_args()
    from transformers import AutoTokenizer

    old = AutoTokenizer.from_pretrained(a.mistral_tokenizer, token=False)
    new = AutoTokenizer.from_pretrained(a.new, token=False)
    fails = []

    same = sum(1 for i in range(MISTRAL_VOCAB) if old.convert_ids_to_tokens(i) == new.convert_ids_to_tokens(i))
    print(f"[1] shared ids identical: {same}/{MISTRAL_VOCAB}   new vocab {len(new)} (+{len(new) - MISTRAL_VOCAB})")
    if same != MISTRAL_VOCAB:
        fails.append("shared ids differ")

    for lang, s in EU.items():
        o, n = old.encode(s, add_special_tokens=False), new.encode(s, add_special_tokens=False)
        ok = o == n
        print(f"[2] {lang}: {'identical' if ok else 'CHANGED'}  ({len(o)} -> {len(n)} tokens)")
        if not ok:
            fails.append(f"{lang} segmentation changed")

    tps_old = sum(tokens_per_syllable(old, s) for s in VI) / len(VI)
    tps_new = sum(tokens_per_syllable(new, s) for s in VI) / len(VI)
    print(f"[3] Vietnamese tokens/syllable: {tps_old:.2f} -> {tps_new:.2f}  (limit {a.max_tps})")
    for s in VI[:2]:
        print("     ", new.tokenize(s)[:18])
    if tps_new > a.max_tps:
        fails.append(f"tokens/syllable {tps_new:.2f} > {a.max_tps}")

    bad = 0
    for sent, spans in CS:
        ids = new.encode(sent, add_special_tokens=False)
        for sp in spans:
            sub = new.encode(" " + sp, add_special_tokens=False)
            old_sub = old.encode(" " + sp, add_special_tokens=False)
            contained = any(ids[i:i + len(sub)] == sub for i in range(len(ids) - len(sub) + 1))
            if not contained or sub != old_sub:
                bad += 1; print(f"[4] span {sp!r}: contained={contained} same_as_mistral={sub == old_sub}")
    print(f"[4] English spans inside Vietnamese: {sum(len(s) for _, s in CS) - bad}/{sum(len(s) for _, s in CS)} intact and unchanged")
    if bad:
        fails.append(f"{bad} English spans altered")

    print("\nRESULT:", "PASS" if not fails else "FAIL: " + "; ".join(fails))
    return 0 if (not fails or a.no_strict) else 1


if __name__ == "__main__":
    sys.exit(main())
