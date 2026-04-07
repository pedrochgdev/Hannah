# fix_final.py
import pathlib, json, os

output_dir = pathlib.Path("tokenizer/hannah_tok")

# Borrar el tokenizer.json roto
tok_json = output_dir / "tokenizer.json"
if tok_json.exists():
    os.remove(tok_json)
    print("Borrado tokenizer.json roto")

# Config apuntando directo al .model (tokenizador lento pero correcto)
config = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "unk_token": "<unk>",
    "pad_token": "<pad>",
    "additional_special_tokens": [
        "[SYS]","[/SYS]","[USR]","[/USR]",
        "[ASS]","[/ASS]","[MEMORY]","[/MEMORY]"
    ],
    "tokenizer_class": "LlamaTokenizer",
    "model_max_length": 32768,
    "legacy": False,
    "sp_model_kwargs": {}
}
with open(output_dir / "tokenizer_config.json", "w") as f:
    json.dump(config, f, indent=2)

print("Listo. Valida con:")
print("  python tokenizer/validate.py --tok tokenizer/hannah_tok")