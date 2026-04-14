import math
import sys
import types

import torch
from transformers import AutoTokenizer

sys.modules["bettermap"] = types.ModuleType("bettermap")
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig

VOCAB_SIZE = 32000
D_MODEL = 1024
N_HEADS = 16
N_LAYERS = 24
CHECKPOINT = "checkpoints/hannah_360m/hannah_final.pt"
TOK_PATH = "src/tokenizer/hannah_tok"

device = torch.device("cuda")
tok = AutoTokenizer.from_pretrained(TOK_PATH)
config = TransformerConfig.olmo3_7B(vocab_size=VOCAB_SIZE, attn_backend=AttentionBackendName.torch)
config.d_model = D_MODEL
config.n_layers = N_LAYERS
config.block.sequence_mixer.d_model = D_MODEL
config.block.sequence_mixer.n_heads = N_HEADS
config.block.sequence_mixer.n_kv_heads = N_HEADS
config.block.feed_forward.hidden_size = int(D_MODEL * 8 / 3)
model = config.build()
ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
model.load_state_dict(state_dict)
model.to(device)
model.eval()


@torch.inference_mode()
def generate(prompt, max_new_tokens=150, temperature=0.8, top_k=50):
    ids = tok.encode(prompt, return_tensors="pt").to(device)
    for _ in range(max_new_tokens):
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(ids)
        logits = logits[:, -1, :] / temperature
        top_vals, top_idx = torch.topk(logits, top_k)
        probs = torch.softmax(top_vals, dim=-1)
        chosen = torch.multinomial(probs[0], 1)
        next_tok = top_idx[0][chosen]
        ids = torch.cat([ids, next_tok.view(1, 1)], dim=1)
        if next_tok.item() == tok.eos_token_id:
            break
    return tok.decode(ids[0], skip_special_tokens=True)


@torch.inference_mode()
def perplexity(text):
    ids = tok.encode(text, return_tensors="pt").to(device)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(ids)
    loss = torch.nn.functional.cross_entropy(logits[0, :-1], ids[0, 1:])
    return math.exp(loss.item())


if __name__ == "__main__":
    print("Modelo base cargado.")
    sample = "Once upon a time"
    print(f"Prompt: {sample}")
    print(f"Gen: {generate(sample)}")
    print(f"PPL: {perplexity('The cat sat on the mat.'):.2f}")
