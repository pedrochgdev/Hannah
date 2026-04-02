
from datasets import load_dataset

TESTS = [
    ("facebook/empathetic_dialogues", None),
    ("AlekseyKorshuk/persona-chat",   None),
    ("roskoN/daily_dialog",           None),
    ("Facebook/blended_skill_talk",   None),
    ("roneneldan/TinyStories",        None),
    ("lucadiliello/bookcorpusopen",   None),
    ("wikimedia/wikipedia",           "20231101.simple"),
    ("allenai/c4",                    "en"),
]

for dataset_id, subset in TESTS:
    try:
        ds = load_dataset(dataset_id, subset, split="train", streaming=True)
        sample = next(iter(ds))
        cols = list(sample.keys())
        print(f"[OK]  {dataset_id:45} cols: {cols}")
    except Exception as e:
        print(f"[ERR] {dataset_id:45} → {e}")

# Agregar al final de TESTS en debug_datasets.py para confirmar los nuevos IDs
TESTS_NEW = [
    ("bdotloh/empathetic-dialogues-contextual", None),
    ("Vipitis/DailyDialog",                     None),
    ("joujou123/blended_skill_talk",             None),
]

for dataset_id, subset in TESTS_NEW:
    try:
        ds     = load_dataset(dataset_id, subset, split="train", streaming=True)
        sample = next(iter(ds))
        cols   = list(sample.keys())
        print(f"[OK]  {dataset_id:50} cols: {cols}")
    except Exception as e:
        print(f"[ERR] {dataset_id:50} → {e}")


# agregar al final de debug_datasets.py
TESTS_NEW = [
    ("Estwld/empathetic_dialogues_llm", None),
    ("frankdarkluo/DailyDialog",        None),
    ("ParlAI/blended_skill_talk",       None),
]

for dataset_id, subset in TESTS_NEW:
    try:
        ds     = load_dataset(dataset_id, subset, split="train", streaming=True)
        sample = next(iter(ds))
        cols   = list(sample.keys())
        print(f"[OK]  {dataset_id:50} cols: {cols}")
    except Exception as e:
        print(f"[ERR] {dataset_id:50} -> {e}")