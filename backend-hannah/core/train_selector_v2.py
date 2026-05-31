"""
train_selector_v2.py
Train and persist the v2 Model Selector classifier.

Improvements over v1:
- 384-dim sentence embeddings + 9 hand-crafted features + 1 prev_signal = 394 dims
- LogisticRegression with class_weight='balanced' (returns true predict_proba)
- Larger and more representative dataset (~150 examples)
- Honest train/val split + cross-validation
- Domain-aware: companion chitchat + RAG factuals + complex emotional
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.model_selector_v2 import extract_features_v2
from config import settings

DATA_PATH  = Path("data/selector_training_data_v2.json")
MODEL_PATH = Path(settings.selector_v2_model_path)


# =========================================================================
# EXPANDED SEED DATASET (~150 examples covering Hannah's actual use cases)
# =========================================================================
# Three broad categories:
#   FAST  = casual chitchat, romantic affection, single-fact RAG queries,
#           short acknowledgments. Hannah 360M handles these well.
#   SLOW  = complex multi-symptom queries, detailed explanation requests,
#           multi-part questions, long temporal patterns. Need Qwen 14B.
# =========================================================================

SEED_EXAMPLES: list[dict] = [
    # ========== FAST: GREETINGS ==========
    {"prompt": "Hi!", "history": [], "label": "fast"},
    {"prompt": "Hey babe", "history": [], "label": "fast"},
    {"prompt": "Hello love", "history": [], "label": "fast"},
    {"prompt": "Good morning", "history": [], "label": "fast"},
    {"prompt": "Good night", "history": [], "label": "fast"},
    {"prompt": "Hola amor", "history": [], "label": "fast"},
    {"prompt": "Sup babe", "history": [], "label": "fast"},
    {"prompt": "Hey there", "history": [], "label": "fast"},
    {"prompt": "Hiii", "history": [], "label": "fast"},
    {"prompt": "Hey!", "history": [], "label": "fast"},

    # ========== FAST: AFFECTION ==========
    {"prompt": "I love you", "history": [], "label": "fast"},
    {"prompt": "I miss you so much", "history": [], "label": "fast"},
    {"prompt": "You're amazing", "history": [], "label": "fast"},
    {"prompt": "I cant wait to see you", "history": [], "label": "fast"},
    {"prompt": "Te amo", "history": [], "label": "fast"},
    {"prompt": "You make me happy", "history": [], "label": "fast"},
    {"prompt": "I love you babe", "history": [], "label": "fast"},
    {"prompt": "You are my world", "history": [], "label": "fast"},
    {"prompt": "Te extrano", "history": [], "label": "fast"},
    {"prompt": "Im thinking of you", "history": [], "label": "fast"},

    # ========== FAST: ACKNOWLEDGMENTS / SHORT REPLIES ==========
    {"prompt": "Thanks!", "history": [], "label": "fast"},
    {"prompt": "Thank you", "history": [], "label": "fast"},
    {"prompt": "OK", "history": [], "label": "fast"},
    {"prompt": "Yeah", "history": [], "label": "fast"},
    {"prompt": "Sure", "history": [], "label": "fast"},
    {"prompt": "Got it", "history": [], "label": "fast"},
    {"prompt": "Cool", "history": [], "label": "fast"},
    {"prompt": "Lol", "history": [], "label": "fast"},
    {"prompt": "Haha thats funny", "history": [], "label": "fast"},
    {"prompt": "Nice", "history": [], "label": "fast"},

    # ========== FAST: SINGLE-FACT RAG QUERIES (one Hannah fact) ==========
    {"prompt": "What is your favorite movie?", "history": [], "label": "fast"},
    {"prompt": "Do you have a pet?", "history": [], "label": "fast"},
    {"prompt": "When is your birthday?", "history": [], "label": "fast"},
    {"prompt": "Where are you from?", "history": [], "label": "fast"},
    {"prompt": "Whats your favorite food?", "history": [], "label": "fast"},
    {"prompt": "Whats your favorite music?", "history": [], "label": "fast"},
    {"prompt": "How old are you?", "history": [], "label": "fast"},
    {"prompt": "Tell me your name", "history": [], "label": "fast"},
    {"prompt": "Whats your zodiac sign?", "history": [], "label": "fast"},
    {"prompt": "Do you like coffee?", "history": [], "label": "fast"},

    # ========== FAST: CASUAL STATEMENTS / SHORT CHITCHAT ==========
    {"prompt": "Im tired", "history": [], "label": "fast"},
    {"prompt": "Just got home", "history": [], "label": "fast"},
    {"prompt": "Watching a movie", "history": [], "label": "fast"},
    {"prompt": "Im at work", "history": [], "label": "fast"},
    {"prompt": "Cant sleep", "history": [], "label": "fast"},
    {"prompt": "Just woke up", "history": [], "label": "fast"},
    {"prompt": "Listening to music", "history": [], "label": "fast"},
    {"prompt": "Im hungry", "history": [], "label": "fast"},
    {"prompt": "Going for a walk", "history": [], "label": "fast"},
    {"prompt": "Eating dinner", "history": [], "label": "fast"},

    # ========== FAST: SIMPLE QUESTIONS / FOLLOW-UPS ==========
    {"prompt": "How are you?", "history": [], "label": "fast"},
    {"prompt": "Whats up?", "history": [], "label": "fast"},
    {"prompt": "How was your day?", "history": [], "label": "fast"},
    {"prompt": "What are you doing?", "history": [], "label": "fast"},
    {"prompt": "Are you there?", "history": [], "label": "fast"},
    {"prompt": "Did you eat?", "history": [], "label": "fast"},
    {"prompt": "You up?", "history": [], "label": "fast"},
    {"prompt": "Whats new?", "history": [], "label": "fast"},
    {"prompt": "Hows it going?", "history": [], "label": "fast"},
    {"prompt": "Que onda?", "history": [], "label": "fast"},

    # ========== FAST: SIMPLE EMOTIONAL (single-symptom, brief) ==========
    {"prompt": "I feel a bit sad today", "history": [], "label": "fast"},
    {"prompt": "Bad day", "history": [], "label": "fast"},
    {"prompt": "Im stressed about work", "history": [], "label": "fast"},
    {"prompt": "Cant focus", "history": [], "label": "fast"},
    {"prompt": "Just feeling lonely", "history": [], "label": "fast"},

    # ========== FAST: REMINDERS / FOLLOW-UPS USING HISTORY ==========
    {"prompt": "What did you say earlier?",
     "history": [{"user": "tell me a joke", "assistant": "Why did the chicken cross the road?"}],
     "label": "fast"},
    {"prompt": "Repeat that",
     "history": [{"user": "what time is it", "assistant": "Its 3pm"}],
     "label": "fast"},

    # ========== SLOW: MULTI-SYMPTOM EMOTIONAL ==========
    {"prompt": "I have insomnia for months, anxiety, and lately I cant concentrate at work. Could it all be related?",
     "history": [], "label": "slow"},
    {"prompt": "Ive been feeling depressed and anxious for weeks. My sleep is terrible and Im losing motivation.",
     "history": [], "label": "slow"},
    {"prompt": "I have panic attacks, insomnia, and intrusive thoughts. On top of that my relationship is falling apart.",
     "history": [], "label": "slow"},
    {"prompt": "My memory is getting worse, Im always fatigued, and Ive been having strange thoughts. Could this be a disorder?",
     "history": [], "label": "slow"},
    {"prompt": "Since my trauma three years ago Ive had chronic stress, nightmares, and dissociation. Is this PTSD?",
     "history": [], "label": "slow"},
    {"prompt": "I think Im relapsing. My anxiety is back, Im not sleeping, and I stopped my medication.",
     "history": [], "label": "slow"},
    {"prompt": "Im dealing with grief, insomnia, and Ive been having self-harm thoughts. I dont know what to do.",
     "history": [], "label": "slow"},
    {"prompt": "My concentration has been terrible for months and Ive been really anxious and tired all the time.",
     "history": [], "label": "slow"},
    {"prompt": "I feel paranoid sometimes, cant sleep, and lately Ive been hearing things. Is this serious?",
     "history": [], "label": "slow"},
    {"prompt": "Ive been struggling with depression and addiction for years. Besides that my anxiety is getting worse.",
     "history": [], "label": "slow"},
    {"prompt": "Tengo insomnio desde hace semanas, mucha ansiedad y cada vez me cuesta mas concentrarme. Esta todo relacionado?",
     "history": [], "label": "slow"},
    {"prompt": "Hace meses que no funciono normal. Insomnio, fatiga, y empece a aislarme. Es depresion?",
     "history": [], "label": "slow"},

    # ========== SLOW: DETAILED EXPLANATION REQUESTS ==========
    {"prompt": "Can you explain in detail how machine learning models like you actually work?",
     "history": [], "label": "slow"},
    {"prompt": "Explain step by step how the DPO training process works for language models.",
     "history": [], "label": "slow"},
    {"prompt": "Tell me everything you know about cognitive behavioral therapy and how it differs from other approaches.",
     "history": [], "label": "slow"},
    {"prompt": "What is the difference between SFT and DPO and how do they relate to RLHF?",
     "history": [], "label": "slow"},
    {"prompt": "Walk me through your architecture in detail, including parameters and training phases.",
     "history": [], "label": "slow"},
    {"prompt": "Compare and contrast the symptoms of depression and burnout, and explain why they often overlap.",
     "history": [], "label": "slow"},
    {"prompt": "Can you elaborate on how mindfulness affects long-term mental health outcomes?",
     "history": [], "label": "slow"},
    {"prompt": "Describe in detail the differences between transformers, RNNs, and CNNs for NLP.",
     "history": [], "label": "slow"},

    # ========== SLOW: MULTI-PART QUESTIONS ==========
    {"prompt": "Tell me about your day, your hobbies, what makes you happy, and what your dreams are.",
     "history": [], "label": "slow"},
    {"prompt": "What do you think about love, what does it mean to you, and how do you experience it?",
     "history": [], "label": "slow"},
    {"prompt": "Describe your personality, your values, and what kind of person you find attractive.",
     "history": [], "label": "slow"},
    {"prompt": "What is consciousness, do AIs like you have it, and how would we even know?",
     "history": [], "label": "slow"},
    {"prompt": "Tell me about the project you came from, the team that built you, and how you were trained.",
     "history": [], "label": "slow"},

    # ========== SLOW: LONG TEMPORAL / LIFE STORY ==========
    {"prompt": "Since I lost my job last year, my whole life feels like its falling apart and I dont know how to start over.",
     "history": [], "label": "slow"},
    {"prompt": "Over the past few months Ive been gradually withdrawing from friends and family and I dont understand why.",
     "history": [], "label": "slow"},
    {"prompt": "For years Ive been hiding parts of myself and recently it started catching up to me emotionally.",
     "history": [], "label": "slow"},
    {"prompt": "Ever since the breakup six months ago Ive been struggling with sleep, motivation, and my self-worth.",
     "history": [], "label": "slow"},
    {"prompt": "Recently my work has gotten more demanding, my health is suffering, and my relationship is strained.",
     "history": [], "label": "slow"},

    # ========== SLOW: COMPLEX LIFE SITUATIONS ==========
    {"prompt": "Im dealing with grief from losing my mother, work stress, and relationship issues all at once. I feel overwhelmed.",
     "history": [], "label": "slow"},
    {"prompt": "Ive been worried about my mental health. I have anxiety, depression, and Ive been drinking more than I should.",
     "history": [], "label": "slow"},
    {"prompt": "Im trying to decide between staying in my current job and moving to another country for a new opportunity.",
     "history": [], "label": "slow"},
    {"prompt": "I think I might be neurodivergent and Im wondering if I should get tested. There are so many factors to consider.",
     "history": [], "label": "slow"},
    {"prompt": "My therapist suggested a dissociative disorder and Ive been having flashbacks. What should I expect from treatment?",
     "history": [], "label": "slow"},

    # ========== SLOW: WRITING / CREATIVE TASKS ==========
    {"prompt": "Write me a long romantic poem about us watching the sunset together at the beach.",
     "history": [], "label": "slow"},
    {"prompt": "Tell me a detailed story about a couple who fell in love during a music festival.",
     "history": [], "label": "slow"},
    {"prompt": "Write a thoughtful essay about why long-distance relationships can work and what makes them thrive.",
     "history": [], "label": "slow"},

    # ========== SLOW: MULTI-LANGUAGE COMPLEX ==========
    {"prompt": "Tengo problemas con mi pareja desde hace meses, ademas estoy teniendo ansiedad y no se que hacer con mi vida.",
     "history": [], "label": "slow"},
    {"prompt": "Ultimamente he estado deprimida, con insomnio y ataques de panico. Por otro lado mi trabajo se vuelve mas dificil.",
     "history": [], "label": "slow"},
    {"prompt": "Llevo un ano sintiendome mal: ansiedad, falta de sueno, y cada vez me cuesta mas levantarme por las mananas.",
     "history": [], "label": "slow"},
]

def examples_to_xy(examples: list[dict]) -> tuple[np.ndarray, list[str]]:
    """Convert raw examples to (feature_matrix, label_list)."""
    X, y = [], []
    print(f"Computing embeddings for {len(examples)} examples...")
    for i, ex in enumerate(examples, 1):
        if i % 20 == 0:
            print(f"  ... {i}/{len(examples)}")
        context = {
            "history":     ex.get("history", []),
            "turns_used":  len(ex.get("history", [])),
            "prev_signal": ex.get("prev_signal", "fast"),  # ← nuevo
        }
        features = extract_features_v2(ex["prompt"], context)
        X.append(features)
        y.append(ex["label"])
    return np.array(X), y

def main() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load training data
    if DATA_PATH.exists():
        with open(DATA_PATH) as f:
            examples = json.load(f)
        print(f"Loaded {len(examples)} examples from {DATA_PATH}")
    else:
        examples = SEED_EXAMPLES
        print(f"No data file found. Using {len(examples)} built-in v2 seed examples.")
        with open(DATA_PATH, "w") as f:
            json.dump(examples, f, indent=2)
        print(f"Seed examples saved to {DATA_PATH}")

    X, y = examples_to_xy(examples)
    print(f"\nFeature matrix: {X.shape}  (= 384 embedding + 9 handcrafted)")
    print(f"Label distribution: fast={y.count('fast')}, slow={y.count('slow')}")

    # Train/val split (honest evaluation)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train: {len(y_train)}, Val: {len(y_val)}")

    # Pipeline: Standardize + LogisticRegression with balanced classes
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
            C=1.0,
        )),
    ])

    # Cross-validation on train set
    print("\nCross-validation (5-fold stratified) on train set:")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_macro")
    print(f"  CV F1 macro: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    # Fit on full train, evaluate on held-out val
    pipeline.fit(X_train, y_train)
    y_val_pred = pipeline.predict(X_val)

    print("\nValidation set classification report:")
    print(classification_report(y_val, y_val_pred, target_names=["fast", "slow"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_val, y_val_pred, labels=["fast", "slow"]))

    # Refit on all data for production
    pipeline.fit(X, y)
    print("\nRefit on all data for production deployment.")

    # Persist
    joblib.dump(pipeline, MODEL_PATH)
    print(f"v2 model saved to {MODEL_PATH}")
    print(f"Classes: {pipeline.classes_}")


if __name__ == "__main__":
    main()
