"""
Build DPO pairs in `data/finetuning/dpo_dataset.jsonl`.
"""

import json
from pathlib import Path

from tqdm import tqdm

SFT_INPUT = Path("data/finetuning/sft_corpus_clean.jsonl")
DPO_OUTPUT = Path("data/finetuning/dpo_dataset.jsonl")


def split_assistant_turns(text: str) -> list[str]:
    parts = []
    cursor = 0
    while True:
        start = text.find("[ASS]", cursor)
        if start == -1:
            break
        end = text.find("[/ASS]", start)
        if end == -1:
            break
        content = text[start + 5 : end].strip()
        if content:
            parts.append(content)
        cursor = end + 6
    return parts


def to_prompt(text: str) -> str:
    idx = text.rfind("[ASS]")
    return text[: idx + 5] if idx != -1 else text


def make_rejected(chosen: str) -> str:
    # Baseline simple "assistant style" rejected response.
    base = chosen.strip()
    if len(base) > 180:
        base = base[:180].rstrip() + "..."
    return f"As an AI assistant, {base.lower()}"


def main():
    if not SFT_INPUT.exists():
        raise FileNotFoundError(f"No existe {SFT_INPUT}")
    DPO_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(SFT_INPUT, "r", encoding="utf-8") as src, open(DPO_OUTPUT, "w", encoding="utf-8") as out:
        for line in tqdm(src, desc="dpo"):
            try:
                row = json.loads(line)
                text = row.get("text", "")
                ass_turns = split_assistant_turns(text)
                if not ass_turns:
                    continue
                chosen = ass_turns[-1] + " [/ASS]"
                prompt = to_prompt(text)
                rejected = make_rejected(ass_turns[-1]) + " [/ASS]"
                rec = {"prompt": prompt, "chosen": chosen, "rejected": rejected, "source": "sft_bootstrap"}
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
            except Exception:
                continue
    print(f"DPO dataset generado: {written:,} pares -> {DPO_OUTPUT}")


if __name__ == "__main__":
    main()
