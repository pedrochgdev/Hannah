# Training - Scripts de Entrenamiento del Modelo

Scripts para entrenar el modelo Hannah en 3 fases: Pretraining, SFT (Supervised Fine-Tuning), y DPO (Direct Preference Optimization).

## 📋 Estructura

```
src/training/
├── train_hannah.py         # Pretraining del modelo base (fase 1)
├── train_sft_hannah.py     # SFT fine-tuning (fase 2)
└── train_dpo_hannah.py     # DPO optimization (fase 3)
```

---

## 🎯 Fase 1: Pretraining (train_hannah.py)

Entrena el modelo base desde cero usando memmap binarios.

```bash
python src/training/train_hannah.py
```

### Arquitectura del Modelo

- **Modelo:** OLMo Architecture
- **Vocab Size:** 32,000
- **D Model:** 1,024 (embedding dimension)
- **N Heads:** 16 (attention heads)
- **N Layers:** 24 (transformer blocks)
- **Seq Length:** 1,024

### Hiperparámetros

```python
BATCH_SIZE = 4                  # Ajusta según VRAM disponible
GRAD_ACCUM_STEPS = 16           # Acumulación de gradientes
LEARNING_RATE = 3e-4            # Learning rate
MAX_STEPS = 80_000              # Pasos totales de entrenamiento
WARMUP_STEPS = 800              # Warmup
EVAL_INTERVAL = 500             # Evaluar cada N pasos
SAVE_INTERVAL = 2000            # Guardar checkpoint cada N pasos
LOG_INTERVAL = 10               # Loggear cada N pasos
```

### Inputs Requeridos

```
data/finetuning/pretrain/
├── train.bin                # Binario con tokens de entrenamiento
└── val.bin                  # Binario con tokens de validación
```

Genéralos con: `python scripts/data_pipeline/prepare_corpus.py`

### Outputs

```
checkpoints/hannah_360m/
└── hannah_final.pt          # Modelo pretrainado (~1.5 GB)
```

### Estimaciones de Tiempo

- **GPU:** 1x A100 40GB → 1-4 semanas (depende de corpus size)
- **CPU:** No recomendado (muy lento)

---

## 💬 Fase 2: SFT (train_sft_hannah.py)

Fine-tuning supervisado con conversaciones curadas.

```bash
python src/training/train_sft_hannah.py
```

### Arquitectura

Usa el modelo pretrainado como punto de partida.

### Hiperparámetros Clave

```python
TRAIN_BIN_PATH = "data/finetuning/sft/train.bin"
VAL_BIN_PATH = "data/finetuning/sft/val.bin"
CHECKPOINT_BASE = "checkpoints/hannah_360m/hannah_final.pt"
OUT_DIR = "checkpoints/hannah_sft"

BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 1e-4            # Learning rate menor que pretraining
MAX_STEPS = 5000                # Menos steps que pretraining
```

### Inputs Requeridos

```
data/finetuning/sft/
├── train.bin                # Conversaciones curadas (90%)
└── val.bin                  # Validación (10%)

checkpoints/hannah_360m/
└── hannah_final.pt          # Modelo pretrainado (del paso anterior)
```

### Outputs

```
checkpoints/hannah_sft/
└── hannah_sft_final.pt      # Modelo SFT fine-tuned (~1.5 GB)
```

### Estimaciones de Tiempo

- **GPU:** 1x A100 40GB → 2-6 horas

---

## 🎲 Fase 3: DPO (train_dpo_hannah.py)

Direct Preference Optimization con pares de preferencias.

```bash
python src/training/train_dpo_hannah.py
```

### Concepto

DPO optimiza el modelo para preferir respuestas "buenas" sobre "malas" sin usar reward model.

### Hiperparámetros Clave

```python
DPO_DATA_PATH = "data/finetuning/dpo_dataset.jsonl"
CHECKPOINT_SFT = "checkpoints/hannah_sft/hannah_sft_final.pt"
OUT_DIR = "checkpoints/hannah_dpo"
TOK_PATH = "src/tokenizer/hannah_tok"

BATCH_SIZE = 4
LEARNING_RATE = 5e-5            # Learning rate muy baja para DPO
BETA = 0.1                       # DPO beta parameter (controla fuerza)
MAX_STEPS = 3000
```

### Inputs Requeridos

```
data/finetuning/
└── dpo_dataset.jsonl        # Pares (prompt, chosen, rejected)

checkpoints/hannah_sft/
└── hannah_sft_final.pt      # Modelo SFT (del paso anterior)
```

### Outputs

```
checkpoints/hannah_dpo/
└── hannah_dpo_final.pt      # ★ MODELO FINAL (~1.5 GB)
```

### Estimaciones de Tiempo

- **GPU:** 1x A100 40GB → 1-3 horas

---

## 🔧 Ajuste de Hiperparámetros

### Para aumentar Batch Size

Si tienes más memoria VRAM:

```python
# Opción 1: Aumentar directo
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 8  # Mantener batch efectivo similar

# Opción 2: Reducir SEC_LEN
SEQ_LEN = 512  # De 1024 a 512 (reduce VRAM ~50%)
```

### Para reducir tiempo

```python
# Menos steps
MAX_STEPS = 40_000  # De 80_000 (pretraining)

# Evaluar menos frecuentemente
EVAL_INTERVAL = 1000
SAVE_INTERVAL = 4000
```

### Para mejor convergencia

```python
# Learning rate schedule más agresivo
LEARNING_RATE = 5e-4
WARMUP_STEPS = 2000  # Aumentar warmup

# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 💾 Checkpoints y Reanudación

El entrenamiento guarda checkpoints periódicamente.

Para reanudar un entrenamiento interrumpido:

1. El script auto-detecta el último checkpoint
2. Solo ejecuta de nuevo el script:

```bash
python src/training/train_hannah.py  # Reanuda automáticamente
```

---

## 📊 Monitoreo

### TensorBoard (si implementado)

```bash
tensorboard --logdir=checkpoints/hannah_360m/logs
```

### Manual

Los scripts imprimen loss cada `LOG_INTERVAL` pasos:

```
Step 100/80000 | Train Loss: 4.215 | Val Loss: 4.312
Step 110/80000 | Train Loss: 4.189 | Val Loss: 4.298
```

---

## 🐛 Troubleshooting

| Error                                           | Solución                            |
| ----------------------------------------------- | ----------------------------------- |
| CUDA Out of Memory                              | Reduce BATCH_SIZE o SEQ_LEN         |
| "No existe: data/finetuning/pretrain/train.bin" | Ejecuta `prepare_corpus.py` primero |
| "Checkpoint not found"                          | Verifica path es correcto           |
| NaN loss                                        | Reduce LEARNING_RATE                |
| Modelo no aprende                               | Aumenta GRAD_ACCUM_STEPS o warmup   |

---

## 🔄 Workflow Típico Completo

```bash
# 1. Preparar datos
python scripts/data_pipeline/prepare_corpus.py

# 2. Entrenar base (ESPERAR 1-4 semanas)
python src/training/train_hannah.py

# 3. Validar base
python scripts/tests/test_hannah.py

# 4. Preparar SFT data
python scripts/data_pipeline/prepare_sft_corpus.py
python scripts/processing/build_dpo_corpus.py

# 5. SFT (ESPERAR 2-6 horas)
python src/training/train_sft_hannah.py

# 6. DPO (ESPERAR 1-3 horas)
python src/training/train_dpo_hannah.py

# 7. Validar modelo final
python scripts/tests/test_sft_hannah.py
```

---

## 📝 Notas

- El modelo usa `torch.cuda.matmul.allow_tf32 = True` para optimizar A100
- Checkpointing de gradientes activo para reducir VRAM
- Multiprocessing para DataLoader
- Gradient accumulation para simular batch sizes mayores

---

**Última actualización:** Abril 2026
