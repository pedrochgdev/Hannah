# Model - Arquitectura del Modelo

Definición de la arquitectura OLMo utilizada para Hannah.

## Estructura

```
src/model/
└── README.md                # Este archivo
```

---

## Arquitectura: OLMo

Hannah utiliza la arquitectura **OLMo** (Open Language Model from AllenAI).

### Especificaciones

```
Arquitectura:       OLMo Transformer
Vocab Size:         32,000 tokens
Model Dim:          1,024 (d_model)
MLP Dim:            ~2,730 (8/3 * d_model)
Num Heads:          16 (attention)
Num KV Heads:       16 (grouped query attention)
Num Layers:         24 transformer blocks
Sequence Length:    1,024 tokens
Activation:         SwiGLU
Normalization:      RMSNorm (layer norm)
Dropout:            0.0 (dropout rate)
```

### Tamaño del Modelo

- **Parámetros:** ~360M (360 millones)
- **FP32:** ~1.5 GB
- **FP16/BF16:** ~750 MB
- **Checkpoint:** ~1.5 GB (con optimizer state)

---

## Configuración

La configuración del modelo está en `configs/hana_360m.yaml`:

```yaml
# OLMo configuration for Hannah 360M
d_model: 1024
n_layers: 24
n_heads: 16
vocab_size: 32000
sequence_length: 1024
# ... más parámetros
```

---

## Imports

En los scripts de entrenamiento, la arquitectura se importa así:

```python
from olmo_core.nn.transformer import TransformerConfig, TransformerBlock

# Cargar config
config = TransformerConfig.olmo3_7B(vocab_size=32000)
config.d_model = 1024
config.n_layers = 24

# Construir modelo
model = config.build()
```

---

## Características

### Attention Mechanism

- **Multi-Head Attention:** 16 heads
- **Group Query Attention (GQA):** Reduce parámetros de KV cache
- **Backend:** Torch attention (optimizada para GPU)

### Feed-Forward Network

```
Linear(d_model → 8/3 * d_model)
→ SwiGLU activation
→ Linear(8/3 * d_model → d_model)
```

### Normalization

- **LayerNorm:** RMSNorm (más estable que LayerNorm)
- **Pre-normalization:** Norm antes de cada sub-layer

### Training Optimizations

- **Gradient Checkpointing:** Reduce VRAM (~40% savings)
- **bfloat16:** Mixed precision para velocidad
- **Activation Functions:** SwiGLU (mejor que ReLU)

---

## Comparación con Otros Modelos

| Modelo               | Parámetros | Layers | Heads | Seq Len |
| -------------------- | ---------- | ------ | ----- | ------- |
| Hannah (Hannah 360M) | 360M       | 24     | 16    | 1024    |
| GPT-2                | 1.5B       | 12     | 12    | 1024    |
| OPT-1.3B             | 1.3B       | 24     | 16    | 2048    |
| Llama 3B             | 3B         | 26     | 8     | 8192    |

---

## Extensión / Customización

### Cambiar el Tamaño del Modelo

Para un modelo más pequeño:

```python
config.d_model = 512
config.n_layers = 12
config.n_heads = 8
```

Para un modelo más grande:

```python
config.d_model = 2048
config.n_layers = 32
config.n_heads = 32
```

### Usar Otra Arquitectura

Si quieres cambiar de OLMo a otra (Llama, Qwen, etc.):

1. Instala la librería correspondiente
2. Modifica `src/training/train_hannah.py`
3. Cambia imports y config

---

## Referencias

- OLMo Paper: https://arxiv.org/abs/2404.01657
- AllenAI Research: https://allenai.org
- OLMo Core Github: https://github.com/allenai/OLMo

---

## Notas

- La arquitectura está fija por compatibilidad con checkpoints existentes
- Para cambios mayores, requiere reentrenamiento
- El tokenizador (32k vocab) es específico de esta arquitectura

---

**Última actualización:** Abril 2026
