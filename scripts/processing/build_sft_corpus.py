"""
Build SFT corpus in `data/finetuning/sft_corpus.jsonl`.
"""

import json
import os
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

OUTPUT_FILE = Path(os.environ.get("OUTPUT_FILE", "data/finetuning/sft_corpus.jsonl"))
CURATED_FILE = Path("data/raw/hannah_curated.jsonl")

SYSTEM_PROMPT = (
    "You are Hannah, my girlfriend. You are warm, playful, affectionate, and natural. "
    "You are not an assistant and you never say you are an AI."
)


def to_hannah_format(turns: Iterable[dict]) -> str | None:
    result = f"[SYS] {SYSTEM_PROMPT} [/SYS]"
    valid_turns = 0
    for turn in turns:
        role = (turn.get("from") or turn.get("role") or "").strip().lower()
        content = (turn.get("value") or turn.get("content") or "").strip()
        if not content:
            continue
        if role in {"human", "user"}:
            result += f"[USR] {content} [/USR]"
            valid_turns += 1
        elif role in {"gpt", "assistant"}:
            result += f"[ASS] {content} [/ASS]"
            valid_turns += 1
    return result if valid_turns >= 2 else None


def build_from_curated(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="curated"):
            try:
                row = json.loads(line)
                conv = row.get("conversations", [])
                text = to_hannah_format(conv)
                if text:
                    records.append({"text": text, "source": "curated"})
            except Exception:
                continue
    return records


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    records = build_from_curated(CURATED_FILE)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for rec in records:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"SFT corpus generado: {len(records):,} registros -> {OUTPUT_FILE}")
    if not records:
        print("Aviso: no se encontraron datos curados en data/raw/hannah_curated.jsonl")


if __name__ == "__main__":
    main()
