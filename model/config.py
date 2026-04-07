from dataclasses import dataclass

@dataclass
class HannahConfig:
    # ── Arquitectura ──────────────────────────────────────────────────────────
    vocab_size:   int = 32_000
    context_len:  int = 1_024
    d_model:      int = 768       # dimensión de embedding
    n_layers:     int = 16        # 16 capas → ~197M params totales
    n_heads:      int = 16        # heads de atención
    d_ff:         int = 3_072     # FFN inner dim = 4 × d_model
    dropout:      float = 0.1

    # ── Derivados (calculados automáticamente) ────────────────────────────────
    @property
    def d_head(self) -> int:
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) debe ser divisible por n_heads ({self.n_heads})"
        return self.d_model // self.n_heads

    # ── Entrenamiento ─────────────────────────────────────────────────────────
    lr:           float = 3e-4
    weight_decay: float = 0.1
    beta1:        float = 0.9
    beta2:        float = 0.95
    grad_clip:    float = 1.0

    # Warmup + cosine decay
    warmup_steps: int   = 2_000
    max_steps:    int   = 100_000

    # Batch en tokens (no en secuencias)
    # 524_288 tokens/step = 512 × 1024 tokens
    # En RTX 5070: batch_size=4 secuencias × grad_accum=128 pasos
    batch_tokens:     int = 524_288
    batch_size:       int = 4          # secuencias por forward pass
    grad_accum_steps: int = 128        # 4 × 128 × 1024 = 524,288 tokens/step

    # Checkpointing
    checkpoint_every: int = 2_000      # pasos
    eval_every:       int = 500

    # Precisión
    dtype: str = "bfloat16"

# Instancia global — importar desde aquí en todos los módulos
CFG = HannahConfig()