"""
core/model_selector.py
----------------------
Model Selector — decides whether a query should go to the Fast or Slow model.

Classifier:
  - Uses an SVM (LinearSVC) over hand-crafted features extracted from the
    prompt and its conversation context.
  - A Naive Bayes variant is also available and can be swapped in via the
    CLASSIFIER env var if the deployment machine is memory-constrained.
  - When no trained model is found on disk, a rule-based fallback is used
    so the system is always functional.

Feature engineering (see _extract_features):
  - Prompt length in tokens (longer = more complex)
  - Number of distinct symptom / emotional keywords
  - Number of sentences in the prompt
  - Number of question marks (multiple questions = reasoning required)
  - Temporal markers ("since", "for weeks", "lately"...)
  - Presence of multi-topic conjunctions ("and also", "besides", "moreover")
  - Context depth: how many turns of history are available
  - Average length of assistant turns in history (longer past = complex thread)
"""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from config import settings


# ── Signal enum ───────────────────────────────────────────────────────

class ModelSignal(str, Enum):
    FAST = "fast"
    SLOW = "slow"


# ── Feature keywords ──────────────────────────────────────────────────

# Symptoms / emotional distress markers that suggest complexity
COMPLEXITY_KEYWORDS = [
    "insomnia", "anxiety", "depression", "stress", "panic", "trauma",
    "grief", "suicidal", "self-harm", "hallucination", "paranoid",
    "dissociat", "concentration", "memory", "fatigue", "burnout",
    "addiction", "relapse", "chronic", "disorder", "medication",
    "therapy", "psychiatrist", "psychologist",
    # Spanish equivalents (corpus may contain Spanish)
    "ansiedad", "depresion", "insomnio", "estres", "panico",
]

# Temporal markers indicating ongoing or multi-period symptoms
TEMPORAL_MARKERS = [
    r"\bsince\b", r"\bfor weeks\b", r"\bfor months\b", r"\bfor years\b",
    r"\blately\b", r"\brecently\b", r"\bover time\b", r"\bmore and more\b",
    r"\bworse\b", r"\bgets worse\b",
    r"\bdesde hace\b", r"\bultimamente\b", r"\bcada vez\b",
]

# Conjunctions linking multiple distinct topics
MULTI_TOPIC_PATTERNS = [
    r"\band also\b", r"\bin addition\b", r"\bbesides\b", r"\bmoreover\b",
    r"\bon top of that\b", r"\bat the same time\b", r"\bnot only\b",
    r"\badem[aá]s\b", r"\btambi[eé]n\b", r"\bpor otro lado\b",
]


# ── Model Selector ────────────────────────────────────────────────────

class ModelSelector:
    """
    Wraps the trained sklearn classifier and exposes a single
    select(prompt, context) -> ModelSignal method.

    If no model file is found, falls back to rule-based logic so the
    pipeline never breaks during development.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self._model_path = Path(model_path or settings.selector_model_path)
        self._classifier: Pipeline | None = None
        self._load_classifier()

    # ── Public API ────────────────────────────────────────────────────

    def select(self, prompt: str, context: dict) -> tuple[ModelSignal, float]:
        """
        Decide which model should handle this request.

        Parameters
        ----------
        prompt:
            The current user message.
        context:
            The dict returned by TokenHandler.build_context():
            {"prompt", "history", "token_count", "turns_used"}

        Returns
        -------
        (signal, confidence)
            signal     : ModelSignal.FAST or ModelSignal.SLOW
            confidence : float in [0, 1] — how confident the selector is
        """
        features = self._extract_features(prompt, context)

        if self._classifier is not None:
            return self._classify(features)
        else:
            return self._rule_based(features)

    def is_trained(self) -> bool:
        return self._classifier is not None

    # ── Feature extraction ────────────────────────────────────────────

    def _extract_features(self, prompt: str, context: dict) -> np.ndarray:
        """
        Convert (prompt, context) into a fixed-length numeric feature vector.

        Feature index map:
          0  prompt_token_len       — length of current prompt in est. tokens
          1  complexity_kw_count    — number of distinct complexity keywords hit
          2  sentence_count         — number of sentences in the prompt
          3  question_count         — number of '?' characters
          4  temporal_marker_count  — number of temporal marker patterns matched
          5  multi_topic_count      — number of multi-topic conjunctions matched
          6  history_turns          — turns of history available
          7  avg_assistant_len      — mean length of assistant turns in history
          8  prompt_char_len        — raw character count of prompt
        """
        lower = prompt.lower()

        # 0 — estimated token length
        prompt_token_len = len(prompt) / settings.chars_per_token

        # 1 — complexity keyword hits
        complexity_kw_count = sum(
            1 for kw in COMPLEXITY_KEYWORDS if kw in lower
        )

        # 2 — sentence count
        sentence_count = max(1, len(re.split(r"[.!?]+", prompt.strip())))

        # 3 — question count
        question_count = prompt.count("?")

        # 4 — temporal markers
        temporal_marker_count = sum(
            1 for p in TEMPORAL_MARKERS if re.search(p, lower)
        )

        # 5 — multi-topic conjunctions
        multi_topic_count = sum(
            1 for p in MULTI_TOPIC_PATTERNS if re.search(p, lower)
        )

        # 6 & 7 — history stats
        history = context.get("history", [])
        history_turns = len(history)
        if history_turns > 0:
            avg_assistant_len = np.mean(
                [len(t.get("assistant", "")) for t in history]
            )
        else:
            avg_assistant_len = 0.0

        # 8 — raw character count
        prompt_char_len = len(prompt)

        return np.array([
            prompt_token_len,
            complexity_kw_count,
            sentence_count,
            question_count,
            temporal_marker_count,
            multi_topic_count,
            history_turns,
            avg_assistant_len,
            prompt_char_len,
        ], dtype=float)

    # ── Classification backends ───────────────────────────────────────

    def _classify(self, features: np.ndarray) -> tuple[ModelSignal, float]:
        """Use the trained sklearn pipeline."""
        x = features.reshape(1, -1)
        label = self._classifier.predict(x)[0]

        # Probability estimate if the underlying estimator supports it
        if hasattr(self._classifier, "predict_proba"):
            proba = self._classifier.predict_proba(x)[0]
            confidence = float(np.max(proba))
        else:
            # LinearSVC uses decision_function instead of predict_proba
            decision = self._classifier.decision_function(x)[0]
            # Map decision distance to a rough confidence in [0.5, 1.0]
            confidence = float(min(1.0, 0.5 + abs(decision) * 0.1))

        # If confidence is too low, default to slow (safer)
        if confidence < settings.selector_confidence_threshold:
            return ModelSignal.SLOW, confidence

        signal = ModelSignal.FAST if label == "fast" else ModelSignal.SLOW
        return signal, confidence

    @staticmethod
    def _rule_based(features: np.ndarray) -> tuple[ModelSignal, float]:
        """
        Deterministic fallback used when no trained model is available.

        Rule summary:
          SLOW if any of:
            - complexity keywords >= 2
            - temporal markers >= 1
            - multi-topic conjunctions >= 1
            - prompt length > 80 tokens AND question count >= 2
          FAST otherwise.
        """
        (
            prompt_token_len,
            complexity_kw_count,
            _sentence_count,
            question_count,
            temporal_marker_count,
            multi_topic_count,
            _history_turns,
            _avg_assistant_len,
            _prompt_char_len,
        ) = features

        is_slow = (
            complexity_kw_count >= 2
            or temporal_marker_count >= 1
            or multi_topic_count >= 1
            or (prompt_token_len > 80 and question_count >= 2)
        )

        signal = ModelSignal.SLOW if is_slow else ModelSignal.FAST
        # Rule-based confidence is fixed — no probability estimate available
        confidence = 0.75
        return signal, confidence

    # ── Model loading ─────────────────────────────────────────────────

    def _load_classifier(self) -> None:
        if self._model_path.exists():
            self._classifier = joblib.load(self._model_path)
        else:
            self._classifier = None
