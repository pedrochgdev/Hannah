# Tokenizer - Tokenización de Texto

Módulo para entrenar, validar y usar el tokenizador Hannah.

## Estructura

```
src/tokenizer/
├── train.py                 # Entrenar tokenizador desde corpus
├── validate.py              # Validar tokenizador
├── hannah_tok/              # ★ Tokenizador preentrenado
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── tokenizer.vocab
├── hannah_tok_fixed/        # Tokenizador alternativo (respaldo)
└── test/
    ├── test_corpus.py       # Crear corpus de prueba
    └── README.md
```

---

## Tokenizador Pretrainado

El proyecto ya incluye un tokenizador entrenado: **hannah_tok**

### Configuración

```
Vocab Size: 32,000
Model Type: BPE (Byte-Pair Encoding - HuggingFace)
Format: HuggingFace `transformers`
```

### Uso

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("src/tokenizer/hannah_tok")

# Encoding
tokens = tokenizer.encode("Hola mundo")
print(tokens)  # [1234, 5678]

# Decoding
text = tokenizer.decode(tokens)
print(text)  # "Hola mundo"

# Con special tokens
input_ids = tokenizer.encode("Hola", add_special_tokens=True)
```

---

## Entrenar Tokenizador Nuevo

Si necesitas entrenar un tokenizador personalizado desde un corpus:

```bash
python src/tokenizer/train.py \
    --corpus data/processed/corpus_final.jsonl \
    --vocab_size 32000 \
    --output src/tokenizer/hannah_tok_new
```

### Hiperparámetros

```python
--corpus TEXT               # Ruta al corpus JSONL
--vocab_size INT            # Tamaño vocabulario (default: 32000)
--output PATH               # Directorio de salida
--min_frequency INT         # Frecuencia mínima (default: 2)
--special_tokens STR        # Tokens especiales
```

### Output esperado

```
src/tokenizer/hannah_tok_new/
├── tokenizer_config.json    # Config JSON
├── tokenizer.json           # Vocabulario BPE
└── tokenizer.vocab          # Vocabulario plano (backup)
```

**Tiempo:** 1-5 minutos (depende corpus size)

---

## Validar Tokenizador

Para validar que el tokenizador funciona correctamente:

```bash
python src/tokenizer/validate.py \
    --tokenizer src/tokenizer/hannah_tok
```

### Output esperado

```
[✓] Tokenizador cargado: hannah_tok
[✓] Vocab size: 32,000
[✓] Encoding/Decoding test passed
```

---

## Tests del Tokenizador

```bash
python src/tokenizer/test/test_corpus.py
```

---

## Tokens Especiales

| Token   | Uso                       |
| ------- | ------------------------- |
| `<s>`   | Inicio de secuencia (BOS) |
| `</s>`  | Fin de secuencia (EOS)    |
| `<unk>` | Token desconocido         |
| `[SYS]` | System prompt             |
| `[USR]` | User input                |
| `[ASS]` | Assistant response        |

---

**Última actualización:** Abril 2026
