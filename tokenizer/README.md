# Entrenamiento completo - Ejecutar

python tokenizer/train.py \
 --corpus corpus_final.jsonl \
 --output tokenizer/hannah_tok

# Prueba rápida con 500k oraciones - Ya Testeado

python tokenizer/train.py \
 --corpus corpus_final.jsonl \
 --output tokenizer/hannah_tok_test \
 --max-lines 500000

# Validación - Ejecutar 

python tokenizer/validate.py --tok tokenizer/hannah_tok

# Training Loop

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("tokenizer/hannah_tok")

# Por batch

encoded = tok(
["Hello, how are you?", "I missed you today."],
padding = True,
truncation = True,
max_length = 1024,
return_tensors = "pt",
)
