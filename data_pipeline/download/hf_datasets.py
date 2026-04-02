#HuggingFace

from datasets import load_dataset
import json, pathlib

OUTPUT = pathlib.Path("raw/hf")
OUTPUT.mkdir(parents=True, exist_ok=True)

# Cada fuente con su subset y columnas de texto relevantes
SOURCES = [
    {
        "name":    "empatheticdialogues",
        "id":      "empathetic_dialogues",
        "subset":  None,
        "split":   "train",
        "cols":    ["prompt", "utterance"],   
    },
    {
        "name":    "personachat",
        "id":      "bavard/personachat_truecased",
        "subset":  None,
        "split":   "train",
        "cols":    ["history", "candidates"],
    },
    {
        "name":    "dailydialog",
        "id":      "daily_dialog",
        "subset":  None,
        "split":   "train",
        "cols":    ["dialog"],              
    },
    {
        "name":    "blendedskill",
        "id":      "blended_skill_talk",
        "subset":  None,
        "split":   "train",
        "cols":    ["previous_utterance", "free_messages", "guided_messages"],
    },
    {
        "name":    "tinystories",
        "id":      "roneneldan/TinyStories",
        "subset":  None,
        "split":   "train",
        "cols":    ["text"],
    },
    {
        "name":    "bookcorpus",
        "id":      "bookcorpus",
        "subset":  None,
        "split":   "train",
        "cols":    ["text"],
        "max_rows": 9_000_000,            
    },
    {
        "name":    "wikipedia",
        "id":      "wikipedia",
        "subset":  "20220301.simple",    
        "split":   "train",
        "cols":    ["text"],
    },
    {
        "name":    "c4",
        "id":      "c4",
        "subset":  "en",
        "split":   "train",
        "cols":    ["text"],
        "max_rows": 9_000_000,        
    },
]

def flatten_text(sample: dict, cols: list) -> str:
    """Extrae y concatena el texto de las columnas relevantes."""
    parts = []
    for col in cols:
        val = sample.get(col, "")
        if isinstance(val, list):
            # Listas de turnos (DailyDialog, etc.) → unir con separador
            parts.append(" ".join(
                str(v) for v in val if isinstance(v, str) and v.strip()
            ))
        elif isinstance(val, str) and val.strip():
            parts.append(val.strip())
    return " ".join(parts)

def download_all():
    for src in SOURCES:
        out_path = OUTPUT / f"{src['name']}.jsonl"
        if out_path.exists():
            print(f"[SKIP] {src['name']} download done")
            continue

        print(f"[DOWN] Descargando {src['name']}...")
        
        # Solo añadir trust_remote_code=True a todas las llamadas
        try:
            ds = load_dataset(
                src["id"],
                src.get("subset"),
                split=src["split"],
                trust_remote_code=True,  # ÚNICO CAMBIO: añadir esta línea
            )
        except Exception as e:
            print(f"[ERROR] Could not load {src['name']}: {e}")
            continue

        max_rows = src.get("max_rows", len(ds))
        ds = ds.select(range(min(max_rows, len(ds))))

        with open(out_path, "w", encoding="utf-8") as f:
            for sample in ds:
                text = flatten_text(sample, src["cols"])
                if text:
                    f.write(json.dumps({"text": text, "source": src["name"]}) + "\n")

        print(f"[OK]   {src['name']}: {len(ds):,} muestras → {out_path}")

if __name__ == "__main__":
    download_all()