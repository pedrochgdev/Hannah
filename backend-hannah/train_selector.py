"""
train_selector.py
-----------------
Entrena y persiste el clasificador Model Selector para Hannah.

Arquitectura: usa ModelSelector del backend directamente para garantizar
que las features de entrenamiento e inferencia sean 100% idénticas.
Datos: ejemplos reales del proyecto Hannah (chat casual, romántico, RAG,
inglés conversacional) — no ejemplos de salud mental.

Uso:
    cd backend-hannah
    python train_selector.py

Resultado:
    data/model_selector.joblib — listo para inferencia en producción.

Para extender con ejemplos reales de producción:
    Editar data/selector_training_data.json y volver a correr.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from core.model_selector import ModelSelector
from config import settings

DATA_PATH  = Path("data/selector_training_data.json")
MODEL_PATH = Path(settings.selector_model_path)

# ====================================================================
# SEED EXAMPLES — datos reales del proyecto Hannah
# Etiquetas:
#   "fast" → Hannah 360M  (chat casual, respuesta corta, baja latencia)
#   "slow" → Qwen2.5-14B  (razonamiento, texto largo, multi-tema)
#
# Regla general:
#   fast = mensajes que un novio mandaría por WhatsApp
#   slow = preguntas que requieren pensar o generar texto largo
# ====================================================================

SEED_EXAMPLES: list[dict] = [

    # ── FAST: Saludos y mensajes casuales ────────────────────────────
    {"prompt": "hi",                        "history": [], "label": "fast"},
    {"prompt": "hey Hannah",                "history": [], "label": "fast"},
    {"prompt": "hello!",                    "history": [], "label": "fast"},
    {"prompt": "good morning",              "history": [], "label": "fast"},
    {"prompt": "good night babe",           "history": [], "label": "fast"},
    {"prompt": "sup",                       "history": [], "label": "fast"},
    {"prompt": "what's up",                 "history": [], "label": "fast"},
    {"prompt": "how are you?",              "history": [], "label": "fast"},
    {"prompt": "how's your day going?",     "history": [], "label": "fast"},
    {"prompt": "I missed you",              "history": [], "label": "fast"},
    {"prompt": "love you",                  "history": [], "label": "fast"},
    {"prompt": "you're so cute",            "history": [], "label": "fast"},
    {"prompt": "thanks!",                   "history": [], "label": "fast"},
    {"prompt": "ok",                        "history": [], "label": "fast"},
    {"prompt": "yeah",                      "history": [], "label": "fast"},
    {"prompt": "lol",                       "history": [], "label": "fast"},
    {"prompt": "haha that's funny",         "history": [], "label": "fast"},
    {"prompt": "bye see you later",         "history": [], "label": "fast"},
    {"prompt": "talk to you tomorrow",      "history": [], "label": "fast"},
    {"prompt": "I'm bored",                 "history": [], "label": "fast"},

    # ── FAST: Conversación casual / emocional simple ─────────────────
    {"prompt": "I had a good day today",            "history": [], "label": "fast"},
    {"prompt": "I'm feeling happy right now",       "history": [], "label": "fast"},
    {"prompt": "I'm a bit tired",                   "history": [], "label": "fast"},
    {"prompt": "I just ate lunch",                  "history": [], "label": "fast"},
    {"prompt": "tell me a joke",                    "history": [], "label": "fast"},
    {"prompt": "what are you doing?",               "history": [], "label": "fast"},
    {"prompt": "do you like music?",                "history": [], "label": "fast"},
    {"prompt": "what kind of food do you like?",    "history": [], "label": "fast"},
    {"prompt": "do you have any hobbies?",          "history": [], "label": "fast"},
    {"prompt": "I like pizza",                      "history": [], "label": "fast"},
    {"prompt": "it's raining here",                 "history": [], "label": "fast"},
    {"prompt": "I went to the movies yesterday",    "history": [], "label": "fast"},
    {"prompt": "can you sing?",                     "history": [], "label": "fast"},
    {"prompt": "you're funny",                      "history": [], "label": "fast"},
    {"prompt": "what's your name?",                 "history": [], "label": "fast"},

    # ── FAST: Preguntas simples sobre Hannah ─────────────────────────
    {"prompt": "how old are you?",              "history": [], "label": "fast"},
    {"prompt": "when is your birthday?",        "history": [], "label": "fast"},
    {"prompt": "do you have a pet?",            "history": [], "label": "fast"},
    {"prompt": "what's your favorite movie?",   "history": [], "label": "fast"},
    {"prompt": "what music do you listen to?",  "history": [], "label": "fast"},
    {"prompt": "where are you from?",           "history": [], "label": "fast"},
    {"prompt": "what's your favorite color?",   "history": [], "label": "fast"},
    {"prompt": "do you like anime?",            "history": [], "label": "fast"},

    # ── FAST: Inglés simple ──────────────────────────────────────────
    {"prompt": "how do you say 'gato' in English?",         "history": [], "label": "fast"},
    {"prompt": "is 'runned' correct?",                      "history": [], "label": "fast"},
    {"prompt": "what does 'chill' mean?",                   "history": [], "label": "fast"},
    {"prompt": "can you correct my sentence: I goed to school", "history": [], "label": "fast"},
    {"prompt": "what's the past tense of 'go'?",            "history": [], "label": "fast"},
    {"prompt": "is it 'a' or 'an' before 'hour'?",          "history": [], "label": "fast"},

    # ── FAST: Respuestas cortas en conversación ──────────────────────
    {"prompt": "yeah I think so",
     "history": [{"user": "I like ramen", "assistant": "Me too! Have you tried tonkotsu?"}],
     "label": "fast"},
    {"prompt": "that sounds fun",
     "history": [{"user": "what are you doing", "assistant": "Just listening to lo-fi beats"}],
     "label": "fast"},
    {"prompt": "really? tell me more",
     "history": [{"user": "do you have a pet", "assistant": "Yes! I have a cat named Mochi"}],
     "label": "fast"},
    {"prompt": "no I haven't",
     "history": [{"user": "do you like anime", "assistant": "Yes! Have you seen Spirited Away?"}],
     "label": "fast"},
    {"prompt": "haha nice",
     "history": [{"user": "tell me something fun", "assistant": "Mochi once slept on my keyboard and sent a random email"}],
     "label": "fast"},

    # ── FAST: Romántico / flirty ─────────────────────────────────────
    {"prompt": "I love you so much",                "history": [], "label": "fast"},
    {"prompt": "you make me happy",                 "history": [], "label": "fast"},
    {"prompt": "I miss you",                        "history": [], "label": "fast"},
    {"prompt": "you're beautiful",                  "history": [], "label": "fast"},
    {"prompt": "can't stop thinking about you",     "history": [], "label": "fast"},
    {"prompt": "you're trouble you know that",      "history": [], "label": "fast"},
    {"prompt": "stop being so cute",                "history": [], "label": "fast"},
    {"prompt": "I want a hug",                      "history": [], "label": "fast"},
    {"prompt": "goodnight, dream of me",            "history": [], "label": "fast"},
    {"prompt": "you owe me a kiss",                 "history": [], "label": "fast"},

    # ── FAST: Días malos / emocional simple ──────────────────────────
    {"prompt": "today was rough",                   "history": [], "label": "fast"},
    {"prompt": "I'm exhausted",                     "history": [], "label": "fast"},
    {"prompt": "everything went wrong today",       "history": [], "label": "fast"},
    {"prompt": "I need a hug",                      "history": [], "label": "fast"},
    {"prompt": "feeling a bit lonely",              "history": [], "label": "fast"},
    {"prompt": "not myself lately",                 "history": [], "label": "fast"},

    # ── SLOW: Preguntas factuales complejas ──────────────────────────
    {"prompt": "explain how you were created and what architecture you use",
     "history": [], "label": "slow"},
    {"prompt": "what is the difference between SFT and DPO training?",
     "history": [], "label": "slow"},
    {"prompt": "tell me everything about your training process in detail",
     "history": [], "label": "slow"},
    {"prompt": "how does natural language processing work? can you explain it step by step?",
     "history": [], "label": "slow"},
    {"prompt": "what is a transformer model and why is it important for AI?",
     "history": [], "label": "slow"},
    {"prompt": "can you describe how your memory system works and how you retrieve information?",
     "history": [], "label": "slow"},
    {"prompt": "what are the differences between your fast model and slow model?",
     "history": [], "label": "slow"},
    {"prompt": "explain the concept of embeddings and how they represent meaning in vector space",
     "history": [], "label": "slow"},

    # ── SLOW: Razonamiento / cálculo ─────────────────────────────────
    {"prompt": "what is 5 times 5?",                "history": [], "label": "slow"},
    {"prompt": "can you solve this: if a train travels 60 km/h for 2 hours, how far does it go?",
     "history": [], "label": "slow"},
    {"prompt": "what's the capital of France and tell me some history about it?",
     "history": [], "label": "slow"},
    {"prompt": "compare the advantages and disadvantages of learning English online versus in person",
     "history": [], "label": "slow"},
    {"prompt": "why do some people find it harder to learn a second language as adults?",
     "history": [], "label": "slow"},
    {"prompt": "what would happen if humans could only communicate through written text?",
     "history": [], "label": "slow"},

    # ── SLOW: Explicaciones largas / multi-tema ───────────────────────
    {"prompt": "I want to improve my English pronunciation and also learn more vocabulary. Besides that, I struggle with grammar. Can you help me with all of that?",
     "history": [], "label": "slow"},
    {"prompt": "tell me about the history of artificial intelligence, who invented it and how has it evolved over time?",
     "history": [], "label": "slow"},
    {"prompt": "I've been trying to learn English for months but I feel like I'm not improving. My reading is okay but my speaking is terrible and I can't understand native speakers when they talk fast. What should I do differently?",
     "history": [], "label": "slow"},
    {"prompt": "what are the main differences between British English and American English? Include pronunciation, spelling, and vocabulary differences",
     "history": [], "label": "slow"},
    {"prompt": "can you explain the difference between present perfect and past simple? I always confuse them. Give me examples of when to use each one",
     "history": [], "label": "slow"},

    # ── SLOW: Conversación compleja con historial ────────────────────
    {"prompt": "so based on everything we've discussed, what do you think I should focus on first?",
     "history": [
         {"user": "I want to learn English",     "assistant": "That's great! What's your level?"},
         {"user": "intermediate I think",        "assistant": "Nice! What do you struggle with most?"},
         {"user": "grammar and speaking",        "assistant": "Let's work on both. Grammar first?"},
         {"user": "yes and also listening",      "assistant": "We can add listening exercises too."},
     ],
     "label": "slow"},
    {"prompt": "can you summarize what we talked about and give me a study plan?",
     "history": [
         {"user": "I need to pass TOEFL",        "assistant": "When is your test?"},
         {"user": "in 3 months",                 "assistant": "That's enough time. What's your weakest section?"},
         {"user": "writing and speaking",        "assistant": "Let's focus on those. For writing..."},
     ],
     "label": "slow"},

    # ── SLOW: Generación de texto largo / creativo ───────────────────
    {"prompt": "write me a short story about a cat who goes on an adventure",
     "history": [], "label": "slow"},
    {"prompt": "can you write a paragraph about climate change so I can practice reading comprehension?",
     "history": [], "label": "slow"},
    {"prompt": "help me write an email to my professor explaining why I missed class",
     "history": [], "label": "slow"},
    {"prompt": "can you roleplay a job interview in English? You be the interviewer and ask me difficult questions about my experience and skills",
     "history": [], "label": "slow"},
    {"prompt": "create a dialogue between two people at a restaurant ordering food, and include at least 10 exchanges",
     "history": [], "label": "slow"},
]


# ====================================================================
# FEATURE EXTRACTION
# ====================================================================

def examples_to_xy(examples: list[dict]) -> tuple[np.ndarray, list[str]]:
    """
    Convierte ejemplos a (feature_matrix, labels) usando ModelSelector
    del backend — garantiza compatibilidad 100% con inferencia.
    """
    selector = ModelSelector.__new__(ModelSelector)
    selector._classifier = None

    X, y = [], []
    for ex in examples:
        context = {
            "history":    ex.get("history", []),
            "turns_used": len(ex.get("history", [])),
        }
        features = selector._extract_features(ex["prompt"], context)
        X.append(features)
        y.append(ex["label"])

    return np.array(X), y


# ====================================================================
# MAIN
# ====================================================================

def main() -> None:
    print("=" * 60)
    print("  ENTRENAMIENTO — Model Selector Hannah (Fast/Slow)")
    print("=" * 60)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Cargar datos — archivo externo tiene prioridad sobre seeds
    if DATA_PATH.exists():
        with open(DATA_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        # Filtrar entradas de comentario (tienen key "_comment")
        examples = [ex for ex in raw if "_comment" not in ex]
        print(f"  Ejemplos cargados desde {DATA_PATH}: {len(examples)}")
    else:
        examples = SEED_EXAMPLES
        print(f"  Usando seed examples internos: {len(examples)}")
        # Persistir para poder extender desde producción
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)
        print(f"  Seeds guardados en {DATA_PATH}")

    X, y = examples_to_xy(examples)
    n_fast = y.count("fast")
    n_slow = y.count("slow")

    print(f"\n  Feature matrix : {X.shape}")
    print(f"  Distribución   : fast={n_fast}, slow={n_slow}")
    print(f"  Ratio fast:slow: {n_fast/n_slow:.1f}:1")

    if n_fast < 2 or n_slow < 2:
        print("  ERROR: Necesitas al menos 2 ejemplos de cada clase.")
        return

    # ── Pipelines candidatos ─────────────────────────────────────────
    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LinearSVC(C=1.0, max_iter=2000, random_state=42)),
    ])
    nb_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    GaussianNB()),
    ])

    n_splits = min(5, min(n_fast, n_slow))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    print(f"\n  Cross-validation ({n_splits}-fold estratificado):")
    svm_scores = cross_val_score(svm_pipeline, X, y, cv=cv, scoring="f1_macro")
    nb_scores  = cross_val_score(nb_pipeline,  X, y, cv=cv, scoring="f1_macro")

    print(f"    SVM  F1 macro: {svm_scores.mean():.3f} (+/- {svm_scores.std():.3f})")
    print(f"    NB   F1 macro: {nb_scores.mean():.3f}  (+/- {nb_scores.std():.3f})")

    if svm_scores.mean() >= nb_scores.mean():
        best_pipeline = svm_pipeline
        chosen = "SVM (LinearSVC)"
    else:
        best_pipeline = nb_pipeline
        chosen = "Naive Bayes (GaussianNB)"

    print(f"\n  Seleccionado: {chosen}")

    # Entrenar con todos los datos
    best_pipeline.fit(X, y)

    y_pred = best_pipeline.predict(X)
    print("\n  Classification report (training set):")
    print(classification_report(y, y_pred, target_names=["fast", "slow"]))

    # Persistir
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"  Modelo guardado en: {MODEL_PATH}")

    # ── Test rápido ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TEST RÁPIDO")
    print("=" * 60)

    test_messages = [
        # Esperado: FAST
        ("hi",                                              []),
        ("love you",                                        []),
        ("good morning babe",                               []),
        ("I'm tired today",                                 []),
        ("what's your favorite movie?",                     []),
        ("stop being so cute",                              []),
        ("haha nice", [{"user": "joke?", "assistant": "Why did the chicken..."}]),
        # Esperado: SLOW
        ("explain how transformers work step by step",      []),
        ("what is the difference between SFT and DPO?",    []),
        ("write me a paragraph about climate change",       []),
        ("what is 5 times 5?",                             []),
        ("I've been trying to learn English for months but I'm not improving. My grammar and pronunciation are terrible. What should I do?", []),
        ("can you summarize everything and give me a study plan?",
         [{"user": "help", "assistant": "Sure!"},
          {"user": "grammar too", "assistant": "Let's start with that."},
          {"user": "and speaking", "assistant": "We can add that."}]),
    ]

    selector = ModelSelector.__new__(ModelSelector)
    selector._classifier = None

    for msg, hist in test_messages:
        context  = {"history": hist, "turns_used": len(hist)}
        features = selector._extract_features(msg, context)
        pred     = best_pipeline.predict(features.reshape(1, -1))[0]

        if hasattr(best_pipeline, "decision_function"):
            decision = best_pipeline.decision_function(features.reshape(1, -1))[0]
            conf = float(min(1.0, 0.5 + abs(decision) * 0.1))
        elif hasattr(best_pipeline, "predict_proba"):
            conf = float(max(best_pipeline.predict_proba(features.reshape(1, -1))[0]))
        else:
            conf = 0.0

        label   = "FAST" if pred == "fast" else "SLOW"
        display = msg[:65] + "..." if len(msg) > 65 else msg
        print(f"  [{label}] (conf={conf:.2f}) \"{display}\"")

    print("\n" + "=" * 60)
    print("  COMPLETADO")
    print(f"  Modelo listo en: {MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
