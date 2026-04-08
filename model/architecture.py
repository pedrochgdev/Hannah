import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.config import HannahConfig

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.
    'Causal' = cada token solo ve tokens anteriores (máscara triangular inferior).
    """
    def __init__(self, cfg: HannahConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head  = cfg.d_head
        self.d_model = cfg.d_model
        self.dropout = cfg.dropout

        # Q, K, V en una sola proyección para eficiencia
        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj  = nn.Linear(cfg.d_model, cfg.d_model,     bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop= nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape   # batch, tokens, d_model

        # Proyectar y dividir en Q, K, V
        qkv = self.qkv_proj(x)                          # (B, T, 3C)
        q, k, v = qkv.split(self.d_model, dim=2)        # cada uno (B, T, C)

        # Reshape para multi-head: (B, n_heads, T, d_head)
        def reshape(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)

        # Flash Attention si está disponible (PyTorch >= 2.0)
        # Más rápido y usa menos VRAM que la atención naive
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p  = self.dropout if self.training else 0.0,
            is_causal  = True,   # aplica la máscara causal automáticamente
        )

        # Reunir heads y proyectar de vuelta a d_model
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(y))


class FeedForward(nn.Module):
    """
    FFN con activación GELU.
    Arquitectura: Linear(d_model → d_ff) → GELU → Linear(d_ff → d_model)
    d_ff = 4 × d_model es el estándar GPT.
    """
    def __init__(self, cfg: HannahConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff, bias=False),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model, bias=False),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Un bloque transformer con Pre-LayerNorm (más estable que Post-LN).
    Orden: LN → Atención → residual → LN → FFN → residual
    """
    def __init__(self, cfg: HannahConfig):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = nn.LayerNorm(cfg.d_model)
        self.ffn  = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))   # residual connection + atención
        x = x + self.ffn(self.ln2(x))    # residual connection + FFN
        return x


class HannahLM(nn.Module):
    """
    Modelo completo: embedding → N bloques transformer → LM head.
    """
    def __init__(self, cfg: HannahConfig):
        super().__init__()
        self.cfg = cfg

        self.transformer = nn.ModuleDict({
            # Token embeddings: vocab_size × d_model
            "tok_emb": nn.Embedding(cfg.vocab_size, cfg.d_model),

            # Positional embeddings aprendidos (no sinusoidales)
            "pos_emb": nn.Embedding(cfg.context_len, cfg.d_model),

            "drop": nn.Dropout(cfg.dropout),

            # Stack de N bloques transformer
            "blocks": nn.ModuleList([
                TransformerBlock(cfg) for _ in range(cfg.n_layers)
            ]),

            # LayerNorm final antes del LM head
            "ln_f": nn.LayerNorm(cfg.d_model),
        })

        # LM head: proyecta d_model → vocab_size
        # Sin bias, y con weight tying: comparte pesos con tok_emb
        # Esto ahorra ~25M parámetros y mejora la calidad
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.transformer["tok_emb"].weight   # weight tying

        # Inicializar pesos
        self.apply(self._init_weights)

        # Inicialización especial para proyecciones residuales
        # Escala por 1/sqrt(2 × n_layers) para estabilizar el training
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("net.2.weight"):
                nn.init.normal_(p, mean=0.0,
                                std=0.02 / math.sqrt(2 * cfg.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self,
                input_ids: torch.Tensor,
                labels:    torch.Tensor = None
                ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        input_ids: (B, T) — secuencias de tokens
        labels:    (B, T) — targets para el cálculo de loss (input_ids shifteado 1)
                            Si es None, solo devuelve los logits (para inferencia)
        """
        B, T = input_ids.shape
        assert T <= self.cfg.context_len, \
            f"Secuencia de longitud {T} excede context_len {self.cfg.context_len}"

        device = input_ids.device

        # Embeddings: token + posición
        tok = self.transformer["tok_emb"](input_ids)                 # (B, T, d_model)
        pos = self.transformer["pos_emb"](
            torch.arange(T, device=device).unsqueeze(0)              # (1, T)
        )
        x = self.transformer["drop"](tok + pos)

        # Pasar por todos los bloques transformer
        for block in self.transformer["blocks"]:
            x = block(x)

        # LayerNorm final
        x = self.transformer["ln_f"](x)

        if labels is not None:
            # Training: calcular loss en todos los tokens excepto el último
            logits = self.lm_head(x)                                  # (B, T, V)
            loss   = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # (B×T, V)
                labels.view(-1),                   # (B×T,)
                ignore_index = -100,               # ignorar tokens de padding
            )
            return logits, loss
        else:
            # Inferencia: solo necesitamos el último token
            logits = self.lm_head(x[:, [-1], :])                     # (B, 1, V)
            return logits, None

    @torch.no_grad()
    def generate(self,
                 input_ids:   torch.Tensor,
                 max_new:     int   = 256,
                 temperature: float = 0.8,
                 top_p:       float = 0.9,
                 eos_id:      int   = 3) -> torch.Tensor:
        """
        Generación autoregresiva con top-p (nucleus) sampling.
        """
        self.eval()
        for _ in range(max_new):
            # Truncar al context window si es necesario
            ctx = input_ids[:, -self.cfg.context_len:]

            logits, _ = self(ctx)                    # (B, 1, V)
            logits     = logits[:, -1, :] / temperature

            # Top-p (nucleus) sampling
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)

            # Eliminar tokens que sumen más del top_p
            sorted_probs[cumulative - sorted_probs > top_p] = 0.0
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

            # Samplear y deshacer el sort
            sampled_idx = torch.multinomial(sorted_probs, num_samples=1)
            next_token  = sorted_idx.gather(1, sampled_idx)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            if next_token.item() == eos_id:
                break

        return input_ids