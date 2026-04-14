"""
SFT training script from base checkpoint.
"""

import math
import os
import time
import types
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.modules["bettermap"] = types.ModuleType("bettermap")
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerBlock, TransformerConfig

TRAIN_BIN_PATH = "data/finetuning/sft/train.bin"
VAL_BIN_PATH = "data/finetuning/sft/val.bin"
CHECKPOINT_BASE = "checkpoints/hannah_360m/hannah_final.pt"
OUT_DIR = "checkpoints/hannah_sft"
TOK_PATH = "src/tokenizer/hannah_tok"
os.makedirs(OUT_DIR, exist_ok=True)

VOCAB_SIZE = 32000
D_MODEL = 1024
N_HEADS = 16
N_LAYERS = 24
SEQ_LEN = 1024
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 16
LEARNING_RATE = 5e-5
MAX_STEPS = 15_000
WARMUP_STEPS = 150
EVAL_INTERVAL = 250
SAVE_INTERVAL = 1000
EVAL_ITERS = 50
LOG_INTERVAL = 10

ASS_ID = 8
EASS_ID = 9


class MemmapDataset(Dataset):
    def __init__(self, bin_path, seq_len):
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"No existe: {bin_path}")
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.num_samples = (len(self.data) // (self.seq_len + 1)) - 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        i = torch.randint(0, len(self.data) - self.seq_len - 1, (1,)).item()
        chunk = torch.from_numpy(self.data[i : i + self.seq_len + 1].astype(np.int64))
        return chunk[:-1], chunk[1:]


def masked_loss(logits, targets, input_ids):
    _, _, vocab = logits.shape
    is_ass = input_ids == ASS_ID
    is_eass = input_ids == EASS_ID
    mask = (is_ass.float().cumsum(dim=1) - is_eass.float().cumsum(dim=1)) > 0
    mask = mask & ~is_ass
    if mask.sum() == 0:
        return torch.nn.functional.cross_entropy(logits.view(-1, vocab), targets.view(-1))
    return torch.nn.functional.cross_entropy(logits[mask], targets[mask])


def get_lr(step: int) -> float:
    if step < WARMUP_STEPS:
        return LEARNING_RATE * step / max(1, WARMUP_STEPS)
    if step >= MAX_STEPS:
        return LEARNING_RATE * 0.1
    progress = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    return LEARNING_RATE * 0.1 + 0.5 * (LEARNING_RATE - LEARNING_RATE * 0.1) * (1 + math.cos(math.pi * progress))


def main():
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_loader = DataLoader(MemmapDataset(TRAIN_BIN_PATH, SEQ_LEN), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(MemmapDataset(VAL_BIN_PATH, SEQ_LEN), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    config = TransformerConfig.olmo3_7B(vocab_size=VOCAB_SIZE, attn_backend=AttentionBackendName.torch)
    config.d_model = D_MODEL
    config.n_layers = N_LAYERS
    config.block.sequence_mixer.d_model = D_MODEL
    config.block.sequence_mixer.n_heads = N_HEADS
    config.block.sequence_mixer.n_kv_heads = N_HEADS
    config.block.feed_forward.hidden_size = int(D_MODEL * 8 / 3)
    model = config.build()
    for module in model.modules():
        if isinstance(module, TransformerBlock):
            module.use_checkpointing = True

    ckpt = torch.load(CHECKPOINT_BASE, map_location="cpu", weights_only=False)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state_dict)

    model.to(device)
    model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1, betas=(0.9, 0.95), fused=True)

    @torch.inference_mode()
    def estimate_loss():
        model.eval()
        val_loss = 0.0
        it = iter(val_loader)
        for _ in range(EVAL_ITERS):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(val_loader)
                x, y = next(it)
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = masked_loss(logits, y, x)
            val_loss += loss.item()
        model.train()
        return val_loss / EVAL_ITERS

    step = 0
    t0 = time.time()
    train_it = iter(train_loader)
    while step < MAX_STEPS:
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(GRAD_ACCUM_STEPS):
            try:
                x, y = next(train_it)
            except StopIteration:
                train_it = iter(train_loader)
                x, y = next(train_it)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = masked_loss(logits, y, x) / GRAD_ACCUM_STEPS
            loss.backward()
            accum_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % LOG_INTERVAL == 0:
            dt = time.time() - t0
            tok_sec = (BATCH_SIZE * GRAD_ACCUM_STEPS * SEQ_LEN * LOG_INTERVAL) / max(dt, 1e-6)
            print(f"Step {step}/{MAX_STEPS} | Loss {accum_loss:.4f} | LR {lr:.2e} | {tok_sec/1000:.1f}k tok/s")
            t0 = time.time()
        if step > 0 and step % EVAL_INTERVAL == 0:
            v_loss = estimate_loss()
            print(f"[EVAL] step={step} val_loss={v_loss:.4f} ppl={math.exp(min(v_loss, 20)):.1f}")
        if step > 0 and step % SAVE_INTERVAL == 0:
            path = os.path.join(OUT_DIR, f"hannah_sft_step_{step}.pt")
            torch.save({"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
            print(f"Checkpoint: {path}")
        step += 1

    final_path = os.path.join(OUT_DIR, "hannah_sft_final.pt")
    torch.save({"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, final_path)
    print(f"SFT completado -> {final_path}")


if __name__ == "__main__":
    main()
