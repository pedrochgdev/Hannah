# Configs - Configuración del Modelo

Archivos YAML/JSON con hiperparámetros y configuración del modelo.

## Estructura

```
configs/
└── hana_360m.yaml            # ★ Configuración principal del modelo Hannah 360M
```

## hana_360m.yaml

Configuración completa del modelo OLMo para Hannah.

### Contenido Típico

```yaml
# Arquitectura del modelo
model:
  d_model: 1024 # Embedding dimension
  n_layers: 24 # Number of transformer layers
  n_heads: 16 # Number of attention heads
  vocab_size: 32000 # Vocabulary size
  seq_length: 1024 # Sequence length

# Hiperparámetros de Entrenamiento
training:
  batch_size: 4
  grad_accum_steps: 16
  learning_rate: 3e-4
  warmup_steps: 800
  max_steps: 80000
  eval_interval: 500
  save_interval: 2000

# Optimizador
optimizer:
  name: adamw
  betas: [0.9, 0.95]
  weight_decay: 0.1

# Data
data:
  train_path: "data/finetuning/pretrain/train.bin"
  val_path: "data/finetuning/pretrain/val.bin"
```

### Uso en Scripts

Los scripts de entrenamiento pueden cargar esta configuración:

```python
import yaml

with open("configs/hana_360m.yaml") as f:
    config = yaml.safe_load(f)

# Usar valores
BATCH_SIZE = config['training']['batch_size']
LEARNING_RATE = config['training']['learning_rate']
```

---

## Personalización

### Para un modelo más pequeño

Crea `configs/hannah_small.yaml`:

```yaml
model:
  d_model: 512
  n_layers: 12
  n_heads: 8
  vocab_size: 32000
  seq_length: 512
```

### Para un modelo más grande

Crea `configs/hannah_large.yaml`:

```yaml
model:
  d_model: 2048
  n_layers: 32
  n_heads: 32
  vocab_size: 32000
  seq_length: 2048
```

### Para entrenamiento rápido (test)

```yaml
training:
  max_steps: 1000 # De 80000
  eval_interval: 100
  save_interval: 500
```

---

## Parámetros Clave

### Model Architecture

| Parámetro  | Rango típico    | Hannah 360M |
| ---------- | --------------- | ----------- |
| d_model    | 256-4096        | 1024        |
| n_layers   | 12-80           | 24          |
| n_heads    | 8-64            | 16          |
| head_dim   | d_model/n_heads | 64          |
| vocab_size | 8k-128k         | 32k         |
| seq_length | 256-8192        | 1024        |

### Training Hyperparameters

| Parámetro          | Recomendado           | Hannah       |
| ------------------ | --------------------- | ------------ |
| bsz (batch size)   | 1-128                 | 4            |
| lr (learning rate) | 1e-5 to 1e-3          | 3e-4         |
| warmup             | 1%-10% de máxml_steps | 800/80k (1%) |
| decay              | linear/cosine         | linear       |
| weight_decay       | 0.01-0.1              | 0.1          |

---

## Configurar por Proyecto

Si tienes múltiples proyectos:

```
configs/
├── hana_360m.yaml           # Configuración base
├── hannah_sft.yaml          # Overrides para SFT
├── hannah_dpo.yaml          # Overrides para DPO
└── experiments/
    ├── exp_001_baseline.yaml
    ├── exp_002_larger.yaml
    └── exp_003_longer_seq.yaml
```

### Combinar configs (patrón recomendado)

```python
import yaml

# Config base
with open("configs/hana_360m.yaml") as f:
    base_config = yaml.safe_load(f)

# Overrides para SFT
with open("configs/hannah_sft.yaml") as f:
    sft_overrides = yaml.safe_load(f)

# Combinar
config = {**base_config, **sft_overrides}
```

---

## Best Practices

1. **Versionar configs:** Guarda historico de configs usadas
2. **Documentar cambios:** Agregar comentarios en YAML
3. **Usar defaults:** Tener un `hana_360m.yaml` como referencia
4. **Separar concerns:** Config modelo ≠ config training ≠ config data
5. **Mantener sincronía:** Config debe coincidir con scripts

---

## Troubleshooting

| Error                                  | Causa                 | Solución                  |
| -------------------------------------- | --------------------- | ------------------------- |
| "No such file: configs/hana_360m.yaml" | Archivo falta         | Verifica ruta es correcta |
| "yaml.YAMLError"                       | Formato YAML inválido | Usa validador YAML online |
| "KeyError: 'learning_rate'"            | Clave no existe       | Verifica estructura YAML  |

---

## Referencia Rápida

### Cambios Comunes

```yaml
# Aumentar velocidad (menos precisión)
batch_size: 8
grad_accum_steps: 8
learning_rate: 5e-4

# Mejor convergencia (más lento)
batch_size: 2
grad_accum_steps: 32
learning_rate: 1e-4
warmup_steps: 2000

# Para GPU pequeña (8GB)
d_model: 512
n_layers: 12
seq_length: 512
batch_size: 1

# Para testing rápido
max_steps: 100
eval_interval: 10
```

---

**Última actualización:** Abril 2026
