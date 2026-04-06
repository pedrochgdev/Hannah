from datasets import load_dataset
import json, pathlib

OUTPUT = pathlib.Path("raw/hf")
OUTPUT.mkdir(parents=True, exist_ok=True)

# IDs verificados contra HuggingFace Hub - abril 2026
# empathetic_dialogues  → Estwld/empathetic_dialogues_llm
#                         conversión Parquet oficial, 24,850 filas, formato LLM-ready
#                         cols: ['conv_id','emotion','situation','dialog']
# dailydialog           → frankdarkluo/DailyDialog
#                         73,554 filas, Parquet nativo
#                         cols: verificar con debug script abajo
# blended_skill_talk    → ParlAI/blended_skill_talk
#                         6,808 filas, Parquet nativo confirmado
#                         cols: ['personas','additional_context','context',
#                                'free_turker_utterance','guided_turker_utterance',
#                                'suggestions']
SOURCES = [
    {
        "name":   "empatheticdialogues",
        "id":     "Estwld/empathetic_dialogues_llm",
        "subset": None,
        "split":  "train",
        "cols":   ["situation", "dialog"],
    },
    {
        "name":   "personachat",
        "id":     "AlekseyKorshuk/persona-chat",
        "subset": None,
        "split":  "train",
        "cols":   ["personality", "utterances"],
    },
    {
        "name":   "soda",
        "id":     "allenai/soda",
        "subset": None,
        "split":  "train",
        "cols":   ["narrative"],
    },
    {
        "name":   "openassistant",
        "id":     "OpenAssistant/oasst1",
        "subset": None,
        "split":  "train",
        "cols":   ["text"], 
    },
    {
        "name":   "tinystories",
        "id":     "roneneldan/TinyStories",
        "subset": None,
        "split":  "train",
        "cols":   ["text"],
    },
    {
        "name":     "bookcorpus",
        "id":       "lucadiliello/bookcorpusopen",
        "subset":   None,
        "split":    "train",
        "cols":     ["text"],
        "max_rows": 8_000_000,
    },
    {
        "name":   "wikipedia",
        "id":     "wikimedia/wikipedia",
        "subset": "20231101.simple",
        "split":  "train",
        "cols":   ["text"],
    },
    {
        "name":     "c4",
        "id":       "allenai/c4",
        "subset":   "en",
        "split":    "train",
        "cols":     ["text"],
        "max_rows": 9_000_000,
    },
]

def flatten_text(sample: dict, cols: list) -> str:
    parts = []
    for col in cols:
        val = sample.get(col, "")
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, str) and item.strip():
                    parts.append(item.strip())
                elif isinstance(item, dict):
                    for v in item.values():
                        if isinstance(v, list):
                            parts.extend(s for s in v
                                         if isinstance(s, str) and s.strip())
                        elif isinstance(v, str) and v.strip():
                            parts.append(v.strip())
    return " ".join(parts)

def download_source(src: dict):
    out_path = OUTPUT / f"{src['name']}.jsonl"
    if out_path.exists():
        print(f"[SKIP] {src['name']} ya descargado")
        return

    print(f"[DOWN] {src['name']} <- {src['id']}")
    try:
        ds = load_dataset(src["id"], src.get("subset"), split=src["split"])
    except Exception as e:
        print(f"[ERR]  {src['name']}: {e}")
        return

    max_rows = src.get("max_rows", len(ds))
    ds = ds.select(range(min(max_rows, len(ds))))

    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for sample in ds:
            text = flatten_text(sample, src["cols"])
            if len(text.strip()) > 5:  # Solo evitar líneas completamente vacías
                f.write(json.dumps({"text": text, "source": src["name"]}) + "\n")
                written += 1

    size_mb = out_path.stat().st_size / 1e6
    print(f"[OK]   {src['name']}: {written:,} muestras -> {size_mb:.1f} MB")

def download_all():
    for src in SOURCES:
        download_source(src)

if __name__ == "__main__":
    download_all()
