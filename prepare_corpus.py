# prepare_corpus.py
import numpy as np
import pathlib
import json
from transformers import AutoTokenizer
from tqdm import tqdm

tok = AutoTokenizer.from_pretrained("tokenizer/hannah_tok")
corpus_path = pathlib.Path("corpus_merged.jsonl")
output_dir  = pathlib.Path("nanoGPT/data/hannah")
output_dir.mkdir(parents=True, exist_ok=True)

all_tokens = []

print("Tokenizando corpus...")
with open(corpus_path, "r", encoding="utf-8") as f:
    for line in tqdm(f):
        try:
            text = json.loads(line).get("text", "")
            if len(text.strip()) > 20:
                ids = tok.encode(text, add_special_tokens=False)
                ids.append(tok.eos_token_id)  # separador entre documentos
                all_tokens.extend(ids)
        except:
            continue

print(f"Total tokens: {len(all_tokens):,}")

# Split 95% train, 5% validation
split = int(len(all_tokens) * 0.95)
train = np.array(all_tokens[:split],  dtype=np.uint16)
val   = np.array(all_tokens[split:],  dtype=np.uint16)

train.tofile(output_dir / "train.bin")
val.tofile(output_dir  / "val.bin")

print(f"Train: {len(train):,} tokens → train.bin")
print(f"Val:   {len(val):,}   tokens → val.bin")
