# Data Pipeline - Preparación de Datos

Pipeline de limpieza, combinación y preparación de corpus para entrenamiento.

## Estructura

```
scripts/data_pipeline/
├── build_corpus.py        # Combinador principal: merge + clean + dedup
├── debug_datasets.py      # Verificar acceso a datasets de HuggingFace
├── prepare_corpus.py      # Tokenizar corpus (genera pretrain bins)
├── prepare_sft_corpus.py  # Tokenizar SFT corpus (genera SFT bins)
├── clean_sft_corpus.py    # Limpiar corpus SFT (elimina contenido inapropiado)
├── download/              # Descarga de fuentes externas
│   ├── gutenberg.py       # Project Gutenberg (clásicos de literatura)
│   ├── hf_datasets.py     # HuggingFace Datasets
│   └── extend_corpus.py   # Extensión de corpus adicionales
└── clean/                 # Módulos de limpieza
    ├── filters.py         # Filtros de limpieza de texto
    ├── dedup.py           # MinHash deduplication
    ├── pipeline.py        # Pipeline de limpieza
    ├── stats.py           # Estadísticas del corpus
    └── validate.py        # Validación de calidad
```

## Flujo Completo

### 1️⃣ Fase 1: Descargar Datos (Opcional)

Si necesitas expandir el corpus:

```bash
# Descargar de HuggingFace Datasets
python scripts/data_pipeline/download/hf_datasets.py

# Descargar de Project Gutenberg (romance/clásicos)
python scripts/data_pipeline/download/gutenberg.py

# Validar acceso a datasets
python scripts/data_pipeline/debug_datasets.py
```

**Outputs:** Se guardan en `data/raw/{hf,gutenberg}/`

---

### 2️⃣ Fase 2: Limpiar y Combinar Corpus

El script principal que hace todo:

```bash
cd scripts/data_pipeline
python build_corpus.py
```

**Qué hace:**

1. Lee todos los `.jsonl` en `data/raw/`
2. Aplica filtros de limpieza (HTML, URLs, spam, etc.)
3. Deduplica con MinHash (Jaccard > 85%)
4. Escribe resultado en `data/processed/corpus_final.jsonl`

**⚠️ Requisitos previos:**

- Archivos JSON/JSONL en `data/raw/`
- Al menos 100 MB de datos

**Outputs:**

- `data/processed/corpus_final.jsonl` (corpus limpio)

---

### 3️⃣ Fase 3: Tokenizar para Pretraining

Convierte el corpus en binarios para entrenamiento:

```bash
python scripts/data_pipeline/prepare_corpus.py
```

**Qué hace:**

1. Lee `data/processed/corpus_final.jsonl`
2. Tokeniza cada línea con `src/tokenizer/hannah_tok`
3. Splittea en train (90%) / val (10%)
4. Descarga en binarios numpy memmap

**Outputs:**

- `data/finetuning/pretrain/train.bin` (~80% de datos)
- `data/finetuning/pretrain/val.bin` (~20% de datos)

---

## Scripts de SFT

### Preparar Corpus SFT

```bash
python scripts/processing/build_sft_corpus.py
```

Requiere `data/raw/hannah_curated.jsonl` con conversaciones curadas.

---

### Limpiar Corpus SFT

```bash
python scripts/data_pipeline/clean_sft_corpus.py
```

Elimina:

- Contenido con menores
- Frases genéricas de AI
- Textos demasiado cortos/rotos
- HTML y caracteres de control

**Output:** `data/finetuning/sft_corpus_clean.jsonl`

---

### Tokenizar SFT

```bash
python scripts/data_pipeline/prepare_sft_corpus.py
```

Genera binarios para SFT:

- `data/finetuning/sft/train.bin`
- `data/finetuning/sft/val.bin`

---

## Configuración de Filtros

En `scripts/data_pipeline/clean/filters.py`:

```python
@dataclasses.dataclass
class FilterConfig:
    min_chars:       int   = 80         # Mínimo de caracteres
    max_chars:       int   = 5_000_000  # Máximo de caracteres
    max_digit_ratio: float = 0.15       # Máx % de dígitos
    max_upper_ratio: float = 0.40       # Máx % de MAYÚSCULAS
    min_avg_word:    float = 3.5        # Longitud promedio mínima
    max_alpha_ratio: float = 0.60       # Mínimo % de caracteres alfabéticos
```

Edita estos valores si necesitas ajustar la limpieza.

---

## Variables de Entorno

```bash
# Salida customizada
export OUTPUT_FILE="data/finetuning/sft_corpus.jsonl"

# Paralelismo (desactivar si hay issues):
export TOKENIZERS_PARALLELISM=false
```

---

## 🐛 Troubleshooting

| Problema                          | Solución                                     |
| --------------------------------- | -------------------------------------------- |
| "No existe: data/raw/"            | Crea la carpeta y agrega archivos JSONL      |
| "No se encontraron datos curados" | Coloca `hannah_curated.jsonl` en `data/raw/` |
| Memoria insuficiente              | Reduce chunk_size en `prepare_corpus.py`     |
| Tokenizador no encontrado         | Verifica `src/tokenizer/hannah_tok/`         |

---

## ℹ️ Notas

- El tokenizador usa `AutoTokenizer.from_pretrained()` (HuggingFace format)
- La deduplicación usa MinHash con threshold de Jaccard > 85%
- Los bins son formato NumPy memmap (eficiente en RAM)
- Soporta multiprocessing para acelerar tokenización

**Última actualización:** Abril 2026
