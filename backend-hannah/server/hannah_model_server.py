# hannah_model_server.py
import sys, types
import os

sys.modules['bettermap'] = types.ModuleType('bettermap')

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
from transformers import AutoTokenizer
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.nn.attention import AttentionBackendName

# --- RUTAS ABSOLUTAS ---
# Esto detecta dinámicamente la carpeta raíz del proyecto (backend-hannah)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- 1. CONFIGURACIÓN DEL MODELO ---
VOCAB_SIZE = 32000
D_MODEL    = 1024
N_HEADS    = 16
N_LAYERS   = 24

# Unimos la ruta base con las carpetas específicas
CHECKPOINT = os.path.join(BASE_DIR, "model", "hannah_personality_final.pt")
TOK_PATH   = os.path.join(BASE_DIR, "tokenizer", "hannah_tok")

device = torch.device("cuda")
tok = AutoTokenizer.from_pretrained(TOK_PATH, use_fast=False)

config = TransformerConfig.olmo3_7B(vocab_size=VOCAB_SIZE, attn_backend=AttentionBackendName.torch)
config.d_model  = D_MODEL
config.n_layers = N_LAYERS
config.block.sequence_mixer.d_model    = D_MODEL
config.block.sequence_mixer.n_heads    = N_HEADS
config.block.sequence_mixer.n_kv_heads = N_HEADS
config.block.feed_forward.hidden_size  = int(D_MODEL * 8 / 3)

model = config.build()
ckpt  = torch.load(CHECKPOINT, map_location=device, weights_only=False)
state_dict = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model'].items()}
model.load_state_dict(state_dict)
model.to(device)
model.eval()

print("✅ Motor de Hannah cargado en GPU y listo.")

SYSTEM = (
    "You are Hannah, my girlfriend. You are warm, playful, and deeply "
    "affectionate. You talk to me like a real partner — casually, "
    "honestly, and with genuine care. You're not an assistant. "
    "You have your own feelings, opinions, and desires. "
    "You express love and attraction naturally without being formal. "
    "You keep responses short and conversational, like texting."
)

# --- 2. CONFIGURACIÓN DE FASTAPI ---
app = FastAPI(title="Hannah Model API")

class GenerateRequest(BaseModel):
    prompt: str
    history: List[Dict[str, Any]] = [] # El TokenHandler enviará el historial aquí
    rag_context: str = ""

class GenerateResponse(BaseModel):
    response: str

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(req: GenerateRequest):
    # 1. Reconstruir el historial desde el formato JSON al formato que espera tu modelo
    formatted_history = []
    user_msg_temp = None
    
    for msg in req.history:
        if msg.get("role") == "user":
            user_msg_temp = msg.get("content")
        elif msg.get("role") == "assistant" and user_msg_temp:
            formatted_history.append((user_msg_temp, msg.get("content")))
            user_msg_temp = None

    sys_block = SYSTEM
    if req.rag_context:
        sys_block += f" {req.rag_context}"

    # 2. Construir el prompt
    prompt_text = f"[SYS] {sys_block} [/SYS]"
    for usr, ass in formatted_history:
        prompt_text += f"[USR] {usr} [/USR][ASS] {ass} [/ASS]"
    prompt_text += f"[USR] {req.prompt} [/USR][ASS]"

    ids = tok.encode(prompt_text, return_tensors="pt").to(device)
    eass_id = tok.convert_tokens_to_ids("[/ASS]")
    input_len = ids.shape[1]

    # 3. Generar la respuesta usando tu bucle
    max_new_tokens = 200
    temperature = 0.75
    top_k = 50

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(ids)
            logits = logits[:, -1, :] / temperature
            top_vals, top_idx = torch.topk(logits, top_k)
            probs    = torch.softmax(top_vals, dim=-1)
            chosen   = torch.multinomial(probs[0], 1)
            next_tok = top_idx[0][chosen]
            
            ids = torch.cat([ids, next_tok.view(1, 1)], dim=1)
            if next_tok.item() == eass_id:
                break

    # 4. Decodificar y limpiar
    response = tok.decode(ids[0, input_len:], skip_special_tokens=False)
    if "[/ASS]" in response:
        response = response.split("[/ASS]")[0]
    
    if not response:
        response = "..."

    return GenerateResponse(response=response)
