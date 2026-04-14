# Processing - Construcción de Datasets Específicos

Scripts para construir datasets SFT (Supervised Fine-Tuning) y DPO (Direct Preference Optimization).

## Estructura

```
scripts/processing/
├── build_sft_corpus.py    # Construir corpus SFT desde datos curados
└── build_dpo_corpus.py    # Construir dataset DPO con pares de preferencias
```

## SFT Corpus

### build_sft_corpus.py

Construye un corpus de conversaciones en formato conversacional (sistema, usuario, asistente).

```bash
python scripts/processing/build_sft_corpus.py
```

**Input requerido:**

- `data/raw/hannah_curated.jsonl` - Conversaciones curadas

**Formato esperado:**

```json
{
  "conversations": [
    { "from": "human", "value": "¿Cómo estás?" },
    { "from": "gpt", "value": "¡Bien, gracias! ¿Y tú?" },
    { "from": "human", "value": "Excelente." },
    { "from": "gpt", "value": "¿En qué puedo ayudarte?" }
  ]
}
```

**Output:**

- `data/finetuning/sft_corpus.jsonl`

**Formato de salida:**

```json
{
  "text": "[SYS] You are Hannah... [/SYS][USR] ¿Cómo estás? [/USR][ASS] ¡Bien! [/ASS]..."
}
```

### System Prompt

El corpus SFT usa un system prompt curado:

```
You are Hannah, my girlfriend. You are warm, playful, affectionate, and natural.
You are not an assistant and you never say you are an AI.
```

Edítalo en `build_sft_corpus.py` si necesitas cambiar el comportamiento.

---

## DPO Dataset

### build_dpo_corpus.py

Construye dataset de DPO con pares de respuestas preferidas vs rechazadas.

```bash
python scripts/processing/build_dpo_corpus.py
```

**Input:**

- `data/finetuning/sft_corpus_clean.jsonl` (conversaciones limpias)

**Lógica:**

1. Divide cada conversación en turnos de asistente
2. Para cada turno, elige respuesta "buena" vs "mala"
3. Genera pares DPO: (contexto, respuesta_preferida, respuesta_rechazada)

**Output:**

- `data/finetuning/dpo_dataset.jsonl`

**Formato:**

```json
{
  "prompt": "[SYS] ... [/SYS][USR] ¿Cómo estás? [/USR]",
  "chosen": "[ASS] ¡Hola! Estoy muy bien. ¿Y tú? [/ASS]",
  "rejected": "[ASS] soy un asistente de IA [/ASS]"
}
```

---

## Flujo Típico

1. **build_sft_corpus.py** (datos curados → corpus conversacional)

   ```bash
   python scripts/processing/build_sft_corpus.py
   ```

2. **clean_sft_corpus.py** (limpieza adicional)

   ```bash
   python scripts/data_pipeline/clean_sft_corpus.py
   ```

3. **prepare_sft_corpus.py** (tokenización → bins binarios)

   ```bash
   python scripts/data_pipeline/prepare_sft_corpus.py
   ```

4. **build_dpo_corpus.py** (dataset DPO)
   ```bash
   python scripts/processing/build_dpo_corpus.py
   ```

**Outputs finales:**

- `data/finetuning/sft/train.bin` + `val.bin` (para entrenar SFT)
- `data/finetuning/dpo_dataset.jsonl` (para entrenar DPO)

---

## Personalización

### Cambiar System Prompt

En `build_sft_corpus.py`:

```python
SYSTEM_PROMPT = (
    "Tu nuevo prompt aquí. "
    "Puede tener múltiples líneas."
)
```

### Cambiar Proporción Train/Val

En `prepare_sft_corpus.py`, modifica:

```python
if idx % 20 == 0:  # 5% para validación
    val_ids.extend(ids)
else:
    train_ids.extend(ids)  # 95% para entrenamiento
```

### Filtros Adicionales

En `clean_sft_corpus.py`, agrega regexes:

```python
MINOR_HARD_FILTERS = [
    r"\b(menores|niños)\b",
    # Agregar más patrones aquí...
]
```

---

## Estadísticas Esperadas

**SFT Corpus típico:**

- 2,000 - 10,000 conversaciones
- ~50 MB - 200 MB gzipped

**DPO Dataset típico:**

- 1,000 - 5,000 pares
- ~20 MB - 50 MB

---

## Troubleshooting

| Error                             | Causa                        | Solución                         |
| --------------------------------- | ---------------------------- | -------------------------------- |
| "CURATED_FILE not found"          | Falta `hannah_curated.jsonl` | Coloca el archivo en `data/raw/` |
| "No se encontraron datos curados" | Formato incorrecto           | Valida JSON en cada línea        |
| "IndexError in DPO"               | Conversaciones muy cortas    | Filtra con `min_turns >= 2`      |

---

## Formato de Entrada Esperado

### hannah_curated.jsonl

Cada línea debe ser un JSON válido:

```json
{
  "conversations": [
    { "from": "human", "value": "Hola" },
    { "from": "gpt", "value": "¡Hola!" }
  ]
}
```

O formato alternativo:

```json
{"role": "user", "content": "Hola"}
{"role": "assistant", "content": "¡Hola!"}
```

---

**Última actualización:** Abril 2026
