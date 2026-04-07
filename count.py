from transformers import LlamaTokenizer

print("1. Rescatando el archivo .model nativo...")
tok = LlamaTokenizer(
    vocab_file="tokenizer/hannah_tok/tokenizer.model",
    bos_token="<bos>",
    eos_token="<eos>",
    unk_token="<unk>",
    pad_token="<pad>",
    legacy=False
)

print(f"   ¡Éxito! Vocabulario recuperado: {len(tok)} tokens")

print("\n2. Añadiendo tokens especiales...")
SPECIAL_TOKENS = ["[SYS]", "[/SYS]", "[USR]", "[/USR]", "[ASS]", "[/ASS]", "[MEMORY]", "[/MEMORY]"]
tok.add_special_tokens({'additional_special_tokens': SPECIAL_TOKENS})

print("\n3. Guardando configuración limpia...")
# Guardamos en una nueva carpeta para no mezclar con lo roto
tok.save_pretrained("tokenizer/hannah_tok_fixed")

print("\n¡Rescate completo! Usa hannah_tok_fixed para validar.")