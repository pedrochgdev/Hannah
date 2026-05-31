"""
core/model_selector_v2.py
Improved model selector using semantic embeddings + hand-crafted features.

Architecture:
    prompt -> SentenceTransformer (384 dim) + 9 hand-crafted features + 1 prev_signal (= 394 dim)
           -> StandardScaler -> LogisticRegression (with class_weight='balanced')
           -> (signal, confidence)

Drop-in replacement for ModelSelector v1.

Training: see train_selector_v2.py.
Saved model: data/model_selector_v2.joblib (contains scaler + LR + class labels).
The embedder (all-MiniLM-L6-v2) is loaded lazily and reused (same instance the RAG uses).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from config import settings
from core.model_selector import ModelSignal  # reuse enum from v1


# ── Hand-crafted feature keywords (same as v1, kept for compatibility) ──

COMPLEXITY_KEYWORDS = [
    "insomnia", "anxiety", "depression", "stress", "panic", "trauma",
    "grief", "suicidal", "self-harm", "hallucination", "paranoid",
    "dissociat", "concentration", "memory", "fatigue", "burnout",
    "addiction", "relapse", "chronic", "disorder", "medication",
    "therapy", "psychiatrist", "psychologist",
    "ansiedad", "depresion", "insomnio", "estres", "panico",
]

TEMPORAL_MARKERS = [
    r"\bsince\b", r"\bfor weeks\b", r"\bfor months\b", r"\bfor years\b",
    r"\blately\b", r"\brecently\b", r"\bover time\b", r"\bmore and more\b",
    r"\bworse\b", r"\bgets worse\b",
    r"\bdesde hace\b", r"\bultimamente\b", r"\bcada vez\b",
]

MULTI_TOPIC_PATTERNS = [
    r"\band also\b", r"\bin addition\b", r"\bbesides\b", r"\bmoreover\b",
    r"\bon top of that\b", r"\bat the same time\b", r"\bnot only\b",
    r"\bademas\b", r"\btambien\b", r"\bpor otro lado\b",
]


# ── Embedder singleton (shared with RAG when possible) ─────────────────

_embedder = None


def _get_embedder():
    """Lazy-load the sentence embedder (all-MiniLM-L6-v2)."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return _embedder


# ── Hand-crafted feature extraction (9 features) ───────────────────────

def extract_hand_features(prompt: str, context: dict) -> np.ndarray:
    """Compute the 9 hand-crafted features from v1."""
    lower = prompt.lower()
    prompt_token_len = len(prompt) / settings.chars_per_token
    complexity_kw_count = sum(1 for kw in COMPLEXITY_KEYWORDS if kw in lower)
    sentence_count = max(1, len(re.split(r"[.!?]+", prompt.strip())))
    question_count = prompt.count("?")
    temporal_marker_count = sum(1 for p in TEMPORAL_MARKERS if re.search(p, lower))
    multi_topic_count = sum(1 for p in MULTI_TOPIC_PATTERNS if re.search(p, lower))
    history = context.get("history", [])
    history_turns = len(history)
    if history_turns > 0:
        avg_assistant_len = float(np.mean([len(t.get("assistant", "")) for t in history]))
    else:
        avg_assistant_len = 0.0
    prompt_char_len = len(prompt)
    return np.array([
        prompt_token_len, complexity_kw_count, sentence_count,
        question_count, temporal_marker_count, multi_topic_count,
        history_turns, avg_assistant_len, prompt_char_len,
    ], dtype=float)


def extract_embedding(prompt: str) -> np.ndarray:
    """Compute the 384-dim sentence embedding."""
    embedder = _get_embedder()
    return embedder.encode(prompt, normalize_embeddings=True).astype(float)


def extract_features_v2(prompt: str, context: dict) -> np.ndarray:
    """Combined feature vector: 384 (embedding) + 9 (hand) = 393 dims."""
    emb = extract_embedding(prompt)
    hand = extract_hand_features(prompt, context)
    prev_slow = np.array([float(context.get("prev_signal") == "slow")])
    return np.concatenate([emb, hand, prev_slow])


# ── Main selector class ────────────────────────────────────────────────

class ModelSelectorV2:
    """
    Improved selector using semantic embeddings + hand-crafted features.

    Same public interface as ModelSelector v1:
        select(prompt: str, context: dict) -> (ModelSignal, float)
        is_trained() -> bool
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        self._model_path = Path(model_path or settings.selector_v2_model_path)
        self._classifier: Optional[Pipeline] = None
        self._load_classifier()

    # ── Public API ────────────────────────────────────────────────────

    def select(self, prompt: str, context: dict) -> Tuple[ModelSignal, float]:
        features = extract_features_v2(prompt, context)

        if self._classifier is not None:
            return self._classify(features)
        else:
            return self._rule_based(prompt, context)

    def is_trained(self) -> bool:
        return self._classifier is not None

    # ── Classification backends ───────────────────────────────────────

    def _classify(self, features: np.ndarray) -> Tuple[ModelSignal, float]:
        x = features.reshape(1, -1)
        # LogisticRegression has predict_proba natively
        proba = self._classifier.predict_proba(x)[0]
        classes = list(self._classifier.classes_)

        slow_idx = classes.index("slow") if "slow" in classes else 1
        fast_idx = classes.index("fast") if "fast" in classes else 0
        slow_prob = float(proba[slow_idx])
        fast_prob = float(proba[fast_idx])

        if slow_prob > fast_prob:
            signal = ModelSignal.SLOW
            confidence = slow_prob
        else:
            signal = ModelSignal.FAST
            confidence = fast_prob

        # Apply confidence threshold (default to slow if uncertain)
        if confidence < settings.selector_confidence_threshold:
            return ModelSignal.SLOW, confidence

        return signal, confidence

    @staticmethod
    def _rule_based(prompt: str, context: dict) -> Tuple[ModelSignal, float]:
        """Fallback when no trained model is available."""
        feats = extract_hand_features(prompt, context)
        prompt_token_len, complexity_kw_count, _, question_count, \
            temporal_marker_count, multi_topic_count, *_ = feats

        is_slow = (
            complexity_kw_count >= 2
            or temporal_marker_count >= 1
            or multi_topic_count >= 1
            or (prompt_token_len > 80 and question_count >= 2)
        )
        return (ModelSignal.SLOW if is_slow else ModelSignal.FAST), 0.75

    # ── Model loading ─────────────────────────────────────────────────

    def _load_classifier(self) -> None:
        if self._model_path.exists():
            self._classifier = joblib.load(self._model_path)
        else:
            self._classifier = None
