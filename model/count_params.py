"""
Correr esto antes de iniciar el preentrenamiento.
    python model/count_params.py

Muestra el conteo exacto de parámetros por componente
y la estimación de VRAM para el informe académico.
"""

import torch
from model.architecture import HannahLM
from model.config       import CFG

def count_params(model: HannahLM) -> dict:
    groups = {
        "tok_emb":  model.transformer["tok_emb"],
        "pos_emb":  model.transformer["pos_emb"],
        "blocks":   model.transformer["blocks"],
        "ln_f":     model.transformer["ln_f"],
    }
    result = {}
    for name, module in groups.items():
        n = sum(p.numel() for p in module.parameters())
        result[name] = n

    # lm_head comparte pesos con tok_emb (weight tying) → no suma
    result["lm_head"] = 0   # tied
    result["TOTAL"]   = sum(p.numel() for p in model.parameters())

    # Con weight tying, el total real es menor
    result["TOTAL_UNIQUE"] = result["TOTAL"] - CFG.vocab_size * CFG.d_model
    return result

def estimate_vram(n_params: int, cfg) -> dict:
    bytes_bf16 = n_params * 2
    return {
        "model_bf16_GB":  bytes_bf16 / 1e9,
        "gradients_GB":   bytes_bf16 / 1e9,
        "adamw_states_GB": n_params * 4 * 2 / 1e9,  # m + v en FP32
        "activations_GB": (cfg.batch_size * cfg.context_len *
                           cfg.d_model * cfg.n_layers * 2) / 1e9,
    }

def main():
    print("Inicializando modelo...")
    model = HannahLM(CFG)

    counts = count_params(model)
    vram   = estimate_vram(counts["TOTAL_UNIQUE"], CFG)

    print(f"\n{'═'*55}")
    print(f"  HANNAH — Conteo de parámetros")
    print(f"{'═'*55}")
    print(f"  Configuración: {CFG.n_layers}L / {CFG.n_heads}H / {CFG.d_model}d")
    print(f"  Vocab: {CFG.vocab_size:,}  |  Context: {CFG.context_len} tokens")
    print(f"{'─'*55}")

    for name, n in counts.items():
        if name == "TOTAL_UNIQUE":
            continue
        pct  = n / counts["TOTAL"] * 100 if counts["TOTAL"] > 0 else 0
        bar  = "█" * int(pct / 2)
        print(f"  {name:<12} {n:>12,}   {pct:5.1f}%  {bar}")

    print(f"{'─'*55}")
    print(f"  TOTAL (con tied)  {counts['TOTAL']:>12,}")
    print(f"  TOTAL (único)     {counts['TOTAL_UNIQUE']:>12,}  "
          f"({counts['TOTAL_UNIQUE']/1e6:.1f}M params)")

    print(f"\n  VRAM estimada en RTX 5070:")
    total_vram = sum(vram.values())
    for k, v in vram.items():
        print(f"    {k:<22} {v:.2f} GB")
    print(f"    {'TOTAL':<22} {total_vram:.2f} GB  "
          f"({'OK' if total_vram < 10 else 'AJUSTADO'} para 12 GB)")

    print(f"{'═'*55}\n")

if __name__ == "__main__":
    main()