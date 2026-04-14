"""
DPO training from SFT checkpoint.
"""

import json
import math
import os
import time
import types
import sys

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

sys.modules["bettermap"] = types.ModuleType("bettermap")
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerBlock, TransformerConfig

DPO_DATA_PATH = "data/finetuning/dpo_dataset.jsonl"
CHECKPOINT_SFT = "checkpoints/hannah_sft/hannah_sft_final.pt"
OUT_DIR = "checkpoints/hannah_dpo"
TOK_PATH = "src/tokenizer/hannah_tok"
os.makedirs(OUT_DIR, exist_ok=True)

VOCAB_SIZE = 32000
D_MODEL = 1024
N_HEADS = 16
N_LAYERS = 24
SEQ_LEN = 512
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 2e-6
BETA = 0.3
MAX_STEPS = 1_500
WARMUP_STEPS = 100
LOG_INTERVAL = 10
SAVE_INTERVAL = 500


class DPODataset(Dataset):
    def __init__(self, path, tok, seq_len):
        self.tok = tok
        self.seq_len = seq_len
        self.data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if all(k in row for k in ("prompt", "chosen", "rejected")):
                        self.data.append(row)
                except Exception:
                    continue
        print(f"DPO dataset: {len(self.data):,} pares")

    def __len__(self):
        return len(self.data)

    def tokenize(self, text):
        ids = self.tok.encode(text, add_special_tokens=False)[: self.seq_len]
        pad = [self.tok.pad_token_id] * (self.seq_len - len(ids))
        mask = [1] * len(ids) + [0] * len(pad)
        ids = ids + pad
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long)

    def __getitem__(self, idx):
        row = self.data[idx]
        prompt = row["prompt"]
        chosen = prompt + row["chosen"]
        rejected = prompt + row["rejected"]
        chosen_ids, chosen_mask = self.tokenize(chosen)
        rejected_ids, rejected_mask = self.tokenize(rejected)
        return {"chosen_ids": chosen_ids, "chosen_mask": chosen_mask, "rejected_ids": rejected_ids, "rejected_mask": rejected_mask}


def get_log_probs(model, input_ids, attention_mask):
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(input_ids)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    labels = input_ids[:, 1:].clone()
    log_probs = log_probs[:, :-1, :]
    mask = attention_mask[:, 1:].float()
    token_log_probs = log_probs.gather(2, labels.unsqueeze(2)).squeeze(2)
    return (token_log_probs * mask).sum(dim=1)


def dpo_loss(policy_chosen_lp, policy_rejected_lp, ref_chosen_lp, ref_rejected_lp, beta=0.1):
    chosen_reward = beta * (policy_chosen_lp - ref_chosen_lp)
    rejected_reward = beta * (policy_rejected_lp - ref_rejected_lp)
    loss = -torch.nn.functional.logsigmoid(chosen_reward - rejected_reward).mean()
    accuracy = (chosen_reward > rejected_reward).float().mean()
    margin = (chosen_reward - rejected_reward).mean()
    return loss, accuracy, margin


def build_model(device, ckpt_path):
    config = TransformerConfig.olmo3_7B(vocab_size=VOCAB_SIZE, attn_backend=AttentionBackendName.torch)
    config.d_model = D_MODEL
    config.n_layers = N_LAYERS
    config.block.sequence_mixer.d_model = D_MODEL
    config.block.sequence_mixer.n_heads = N_HEADS
    config.block.sequence_mixer.n_kv_heads = N_HEADS
    config.block.feed_forward.hidden_size = int(D_MODEL * 8 / 3)
    model = config.build()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state_dict)
    return model


def get_lr(step):
    if step < WARMUP_STEPS:
        return LEARNING_RATE * step / max(1, WARMUP_STEPS)
    progress = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    return LEARNING_RATE * 0.1 + 0.5 * (LEARNING_RATE - LEARNING_RATE * 0.1) * (1 + math.cos(math.pi * progress))


def main():
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    tok = AutoTokenizer.from_pretrained(TOK_PATH)

    policy = build_model(device, CHECKPOINT_SFT)
    for m in policy.modules():
        if isinstance(m, TransformerBlock):
            m.use_checkpointing = True
    policy.to(device)
    policy.train()

    reference = build_model(device, CHECKPOINT_SFT)
    reference.to(device)
    reference.eval()
    for p in reference.parameters():
        p.requires_grad = False

    ds = DPODataset(DPO_DATA_PATH, tok, SEQ_LEN)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=LEARNING_RATE, weight_decay=0.01, betas=(0.9, 0.999), fused=True)

    step = 0
    t0 = time.time()
    data_iter = iter(loader)
    while step < MAX_STEPS:
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        accum_loss = accum_acc = accum_margin = 0.0

        for _ in range(GRAD_ACCUM_STEPS):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
            chosen_ids = batch["chosen_ids"].to(device)
            chosen_mask = batch["chosen_mask"].to(device)
            rejected_ids = batch["rejected_ids"].to(device)
            rejected_mask = batch["rejected_mask"].to(device)

            policy_chosen_lp = get_log_probs(policy, chosen_ids, chosen_mask)
            policy_rejected_lp = get_log_probs(policy, rejected_ids, rejected_mask)
            with torch.no_grad():
                ref_chosen_lp = get_log_probs(reference, chosen_ids, chosen_mask)
                ref_rejected_lp = get_log_probs(reference, rejected_ids, rejected_mask)

            loss, acc, margin = dpo_loss(policy_chosen_lp, policy_rejected_lp, ref_chosen_lp, ref_rejected_lp, beta=BETA)
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()
            accum_loss += loss.item()
            accum_acc += acc.item() / GRAD_ACCUM_STEPS
            accum_margin += margin.item() / GRAD_ACCUM_STEPS

        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        if step % LOG_INTERVAL == 0:
            eta_h = ((MAX_STEPS - step) * (time.time() - t0) / LOG_INTERVAL) / 3600
            print(f"Step {step}/{MAX_STEPS} | Loss {accum_loss:.4f} | Acc {accum_acc:.3f} | Margin {accum_margin:.3f} | LR {lr:.2e} | ETA {eta_h:.1f}h")
            t0 = time.time()
        if step > 0 and step % SAVE_INTERVAL == 0:
            path = os.path.join(OUT_DIR, f"hannah_dpo_step_{step}.pt")
            torch.save({"step": step, "model": policy.state_dict(), "optimizer": optimizer.state_dict()}, path)
            print(f"Checkpoint: {path}")
        step += 1

    final_path = os.path.join(OUT_DIR, "hannah_dpo_final.pt")
    torch.save({"step": step, "model": policy.state_dict(), "optimizer": optimizer.state_dict()}, final_path)
    print(f"DPO completado -> {final_path}")


if __name__ == "__main__":
    main()
