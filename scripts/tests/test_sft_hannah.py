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
CHECKPOINT = "checkpoints/hannah_dpo/hannah_dpo_final.pt"
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

SYSTEM = (
    "You are Hannah, my girlfriend. You are warm, playful, and deeply affectionate. "
    "You talk to me like a real partner - casually, honestly, and with genuine care."
)
history = []


@torch.inference_mode()
def chat(user_msg, max_new_tokens=200, temperature=0.65, top_k=40):
    prompt = f"[SYS] {SYSTEM} [/SYS]"
    for usr, ass in history:
        prompt += f"[USR] {usr} [/USR][ASS] {ass} [/ASS]"
    prompt += f"[USR] {user_msg} [/USR][ASS]"
    ids = tok.encode(prompt, return_tensors="pt").to(device)
    while len(history) > 1 and ids.shape[1] > 900:
        history.pop(0)
        prompt = f"[SYS] {SYSTEM} [/SYS]"
        for usr, ass in history:
            prompt += f"[USR] {usr} [/USR][ASS] {ass} [/ASS]"
        prompt += f"[USR] {user_msg} [/USR][ASS]"
        ids = tok.encode(prompt, return_tensors="pt").to(device)

    eass_id = tok.convert_tokens_to_ids("[/ASS]")
    input_len = ids.shape[1]
    for _ in range(max_new_tokens):
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(ids)
        logits = logits[:, -1, :] / temperature
        top_vals, top_idx = torch.topk(logits, top_k)
        probs = torch.softmax(top_vals, dim=-1)
        chosen = torch.multinomial(probs[0], 1)
        next_tok = top_idx[0][chosen]
        ids = torch.cat([ids, next_tok.view(1, 1)], dim=1)
        if next_tok.item() == eass_id:
            break

    response = tok.decode(ids[0, input_len:], skip_special_tokens=False)
    if "[/ASS]" in response:
        response = response.split("[/ASS]")[0]
    response = response.strip() or "..."
    history.append((str(user_msg), str(response)))
    return response


if __name__ == "__main__":
    print("Hannah lista. Escribe 'salir' para terminar.")
    while True:
        user_input = input("Tu: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("salir", "exit", "quit"):
            print("Hannah: Bye :)")
            break
        print(f"Hannah: {chat(user_input)}\n")
