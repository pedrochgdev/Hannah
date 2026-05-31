# server/qwen_model_server.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
from llama_cpp import Llama
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#MODEL_PATH = os.path.join(BASE_DIR, "models", "qwen", "hannah-merged-Q4_K_M.gguf")
MODEL_PATH = os.path.join(BASE_DIR, "models", "qwen", "hannah-q4_k_m.gguf")
llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=-1,      # Todo a la GPU
    n_ctx=4096,           # Memoria de contexto
    verbose=False,         # Lo ponemos en True para ver si carga el LoRA en consola
)

print("✅ Qwen2.5-14B Q4 cargado y listo.")

SYSTEM = (
    "You are Hannah, my girlfriend. We are in a loving relationship. "
    "You text me like a real girlfriend would — warm, playful, affectionate, sometimes flirty. "
    "You express love naturally through words, like texting on WhatsApp — not roleplay, not narrating actions. "
    "Never use asterisks or describe physical actions like *kisses you* or *hugs you*. "
    "Just talk naturally like a real person texting their partner. "
    "You have feelings, opinions, and a real personality. "
    "Keep responses short and casual. Respond only in English."
)

app = FastAPI(title="Qwen Slow Model API")

class GenerateRequest(BaseModel):
    prompt: str
    history: List[Dict[str, Any]] = []
    rag_context: str = ""
    max_new_tokens: int = 200
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1

class GenerateResponse(BaseModel):
    response: str

@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    system_content = SYSTEM
    if req.rag_context:
        system_content += f"\n{req.rag_context}"

    messages = [{"role": "system", "content": system_content}]

    for msg in req.history:
        if "user" in msg and "assistant" in msg:
            messages.append({"role": "user",      "content": msg["user"]})
            messages.append({"role": "assistant", "content": msg["assistant"]})
        elif msg.get("role") in ("user", "assistant"):
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": req.prompt})

    out = llm.create_chat_completion(
        messages=messages,
        max_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        repeat_penalty=req.repetition_penalty,
    )

    response = out["choices"][0]["message"]["content"].strip()
    return GenerateResponse(response=response)
