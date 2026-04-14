# Hannah NLP - Language Model Training Pipeline

> Un pipeline completo de entrenamiento de un modelo de lenguaje personalizado (Hannah 360M) con pretraining, SFT (Supervised Fine-Tuning) y DPO (Direct Preference Optimization).

## Estructura del Proyecto

```
/HannahNLP
├── data/                          # Gestión de datos
│   ├── raw/                       # Datos originales sin procesar (descargados/curados)
│   ├── processed/                 # Corpus limpios (corpus_final.jsonl)
│   └── finetuning/                # Datasets listos para entrenamiento
│       ├── pretrain/              # Binarios para pretraining (train.bin, val.bin)
│       ├── sft/                   # Binarios para SFT (train.bin, val.bin)
│       └── dpo_dataset.jsonl      # Dataset para DPO
├── src/                           # Código fuente principal
│   ├── tokenizer/                 # Tokenizador Hannah
│   │   ├── train.py               # Entrenar tokenizador desde scratch
│   │   ├── validate.py            # Validar tokenizador
│   │   ├── hannah_tok/            # Tokenizador entrenado (directorio)
│   │   ├── hannah_tok_fixed/      # Tokenizador alternativo
│   │   └── test/                  # Tests del tokenizador
│   ├── model/                     # Arquitectura del modelo (OLMo)
│   └── training/                  # Scripts de entrenamiento
│       ├── train_hannah.py        # Pretraining del modelo base
│       ├── train_sft_hannah.py    # SFT con datos curados
│       └── train_dpo_hannah.py    # DPO con prefer. ajustadas
├── scripts/                       # Utilidades y procesamiento
│   ├── data_pipeline/             # Pipeline de preparación de datos
│   │   ├── build_corpus.py        # Combinar, limpiar y deduplicar raw data
│   │   ├── debug_datasets.py      # Verificar datasets de HF
│   │   ├── prepare_corpus.py      # Tokenizar corpus (generar pretrain bins)
│   │   ├── prepare_sft_corpus.py  # Tokenizar SFT corpus (generar SFT bins)
│   │   ├── clean_sft_corpus.py    # Limpiar corpus SFT
│   │   ├── download/              # Descargar datos de fuentes externas
│   │   │   ├── gutenberg.py       # Descargar de Project Gutenberg
│   │   │   ├── hf_datasets.py     # Descargar de HuggingFace
│   │   │   └── extend_corpus.py   # Extender corpus con más datos
│   │   └── clean/                 # Limpieza y deduplicación
│   │       ├── filters.py         # Filtros de limpieza
│   │       ├── dedup.py           # MinHash deduplication
│   │       ├── pipeline.py        # Pipeline de limpieza
│   │       ├── stats.py           # Estadísticas
│   │       └── validate.py        # Validación
│   ├── processing/                # Construcción de datasets específicos
│   │   ├── build_sft_corpus.py    # Construir corpus SFT de datos curados
│   │   └── build_dpo_corpus.py    # Construir dataset DPO
│   └── tests/                     # Tests y validación
│       ├── test_hannah.py         # Test del modelo base entrenado
│       ├── test_sft_hannah.py     # Test del modelo SFT
│       └── tokenizer/             # Tests del tokenizador
├── checkpoints/                   # Pesos del modelo guardados
│   ├── hannah_360m/               # Modelo pretrainado (hannah_final.pt)
│   ├── hannah_sft/                # Modelo SFT (hannah_sft_final.pt)
│   └── hannah_dpo/                # Modelo DPO (hannah_dpo_final.pt)
├── configs/                       # Configuraciones
│   └── hana_360m.yaml             # Hiperparámetros del modelo
├── requirements.txt               # Dependencias
├── .gitignore                     # Ignorar carpetas grandes (data/, checkpoints/)
└── README.md                      # Este archivo
```

## Instalación

```bash
pip install -r requirements.txt
```

**Dependencias principales:**

- `numpy`, `torch`, `transformers`
- `tqdm`, `datasets`, `langdetect`
- `datasketch` (para deduplicación MinHash)
- `olmo_core` (arquitectura OLMo para el modelo)

---

##  Pipeline de Entrenamiento Completo

### **Fase 0: Preparar Tokenizador** 

El tokenizador ya viene entrenado en `src/tokenizer/hannah_tok/`

```bash
# Entrenar tokenizador desde un corpus
python src/tokenizer/train.py --corpus data/processed/corpus_final.jsonl \
   --vocab_size 32000 --output src/tokenizer/hannah_tok_new

# Validar tokenizador
python src/tokenizer/validate.py --tokenizer src/tokenizer/hannah_tok
```

---

### **Fase 1: Construir Corpus Base** 

Combina, limpia y deduplica datos crudos.

```bash
# 1. Descargar datos opcionales
python scripts/data_pipeline/download/hf_datasets.py
python scripts/data_pipeline/download/gutenberg.py

# 2. Limpiar, combinar y deduplicar
cd scripts/data_pipeline && python build_corpus.py
# Output: data/processed/corpus_final.jsonl

# 3. Tokenizar para pretraining
python scripts/data_pipeline/prepare_corpus.py
# Outputs:
#   - data/finetuning/pretrain/train.bin
#   - data/finetuning/pretrain/val.bin
```

---

### **Fase 2: Pretraining**

Entrenar el modelo base (Hannah 360M).

```bash
python src/training/train_hannah.py
# Output: checkpoints/hannah_360m/hannah_final.pt
```

---

### **Fase 3: Construir Corpus SFT**

Preparar datos de SFT y DPO.

```bash
# 1. Construir SFT corpus
python scripts/processing/build_sft_corpus.py
# Output: data/finetuning/sft_corpus.jsonl

# 2. Limpiar corpus SFT
python scripts/data_pipeline/clean_sft_corpus.py
# Output: data/finetuning/sft_corpus_clean.jsonl

# 3. Tokenizar para SFT
python scripts/data_pipeline/prepare_sft_corpus.py
# Outputs:
#   - data/finetuning/sft/train.bin
#   - data/finetuning/sft/val.bin

# 4. Construir dataset DPO
python scripts/processing/build_dpo_corpus.py
# Output: data/finetuning/dpo_dataset.jsonl
```

---

### **Fase 4: SFT (Supervised Fine-Tuning)**

```bash
python src/training/train_sft_hannah.py
# Output: checkpoints/hannah_sft/hannah_sft_final.pt
```

---

### **Fase 5: DPO (Direct Preference Optimization)**

```bash
python src/training/train_dpo_hannah.py
# Output: checkpoints/hannah_dpo/hannah_dpo_final.pt  ← MODELO FINAL
```

---

##  Testing y Validación

```bash
# Test del modelo base
python scripts/tests/test_hannah.py

# Test del modelo DPO
python scripts/tests/test_sft_hannah.py
```

---

##  Estructura de Datos

Antes de ejecutar el pipeline:

```
data/raw/
├── hannah_curated.jsonl       # Datos curados (2-3k ejemplos)
├── gutenberg/                 # Clásicos de literatura
├── hf/                        # Datasets de HuggingFace
└── c4/                        # Common Crawl
```

Salidas del pipeline:

| Fase          | Archivo                                      | Tamaño     |
| ------------- | -------------------------------------------- | ---------- |
| Corpus        | `data/processed/corpus_final.jsonl`          | 10-50 GB   |
| Pretrain Bins | `data/finetuning/pretrain/{train,val}.bin`   | 20-100 GB  |
| SFT Bins      | `data/finetuning/sft/{train,val}.bin`        | 100-500 MB |
| DPO Dataset   | `data/finetuning/dpo_dataset.jsonl`          | 50-200 MB  |
| Model Final   | `checkpoints/hannah_dpo/hannah_dpo_final.pt` | ~1.5 GB    |

---

##  Configuração

### Hiperparámetros

Edita los valores en `src/training/train_*.py`:

**Pretraining:**

- `BATCH_SIZE = 4` (ajusta según VRAM)
- `GRAD_ACCUM_STEPS = 16`
- `LEARNING_RATE = 3e-4`
- `MAX_STEPS = 80_000`
- `SEQ_LEN = 1024`

**SFT/DPO:**

- Valores similares, menos steps

### CUDA Out of Memory

Si se queda sin memoria:

1. Reduce `BATCH_SIZE`
2. Aumenta `GRAD_ACCUM_STEPS`
3. Reduce `SEQ_LEN` a 512

---

##  Troubleshooting

| Error                                           | Solución                           |
| ----------------------------------------------- | ---------------------------------- |
| "No existe: data/finetuning/pretrain/train.bin" | Ejecuta `prepare_corpus.py`        |
| CUDA OOM                                        | Reduce BATCH_SIZE o SEQ_LEN        |
| Tokenizer not found                             | Verifica rutas en `src/tokenizer/` |

---

##  Limpiar Espacio

```bash
# Ver tamaño
du -sh data/ checkpoints/

# Limpiar intermedios
rm -rf data/processed/corpus_merged.jsonl
rm -rf data/processed/corpus_deduped.jsonl
```

---

**Última actualización:** Abril 2026  
