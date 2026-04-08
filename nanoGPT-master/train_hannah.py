# train_hannah.py
import os
from train import *  # importa todo de nanoGPT

# ── Datos ──────────────────────────────────────────────
dataset      = 'hannah'
data_dir     = 'data/hannah'

# ── Arquitectura 360M ──────────────────────────────────
n_layer      = 28
n_head       = 16
n_embd       = 1024
block_size   = 1024       # contexto de 1024 tokens
dropout      = 0.0
bias         = False

# ── Tokenizador ────────────────────────────────────────
vocab_size   = 32000

# ── Entrenamiento ──────────────────────────────────────
batch_size       = 8
gradient_accumulation_steps = 8   # batch efectivo = 64
max_iters        = 600_000
learning_rate    = 3e-4
lr_decay_iters   = 600_000
min_lr           = 3e-5
warmup_iters     = 2000
weight_decay     = 0.1
grad_clip        = 1.0

# ── Optimizador ────────────────────────────────────────
optimizer_name  = 'adamw'
beta1           = 0.9
beta2           = 0.95

# ── GPU (5070 Ti 16GB) ─────────────────────────────────
device          = 'cuda'
dtype           = 'bfloat16'   # más estable que float16
compile         = True         # torch.compile, +30% velocidad

# ── Checkpoints ────────────────────────────────────────
out_dir         = 'out/hannah_360m'
eval_interval   = 500
save_checkpoint = True
eval_iters      = 100
log_interval    = 10
