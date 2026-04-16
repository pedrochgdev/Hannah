"""
train_selector.py
-----------------
Train and persist the Model Selector classifier.

Run once (or whenever you have new labelled examples):
    python train_selector.py

The script:
  1. Loads the labelled dataset from data/selector_training_data.json
     (or uses the built-in seed examples if the file doesn't exist).
  2. Extracts features using the same logic as ModelSelector._extract_features.
  3. Trains both an SVM and a Naive Bayes model, cross-validates both,
     and saves the better one to data/model_selector.joblib.
  4. Prints a classification report so you can verify quality.
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

# ── Seed training examples ────────────────────────────────────────────
# Each entry: {"prompt": "...", "history": [...], "label": "fast"|"slow"}
# Extend this list with real examples from production logs over time.

SEED_EXAMPLES: list[dict] = [
    # FAST — simple, single-topic, short
    {"prompt": "How long should I meditate each day?",
     "history": [], "label": "fast"},
    {"prompt": "What are some good breathing exercises?",
     "history": [], "label": "fast"},
    {"prompt": "Can you remind me what you said about sleep?",
     "history": [{"user": "I can't sleep", "assistant": "Try a bedtime routine."}],
     "label": "fast"},
    {"prompt": "How many hours of sleep do adults need?",
     "history": [], "label": "fast"},
    {"prompt": "I feel a bit sad today.",
     "history": [], "label": "fast"},
    {"prompt": "What can I do when I feel anxious?",
     "history": [{"user": "I'm feeling anxious", "assistant": "Take a breath."}],
     "label": "fast"},
    {"prompt": "Thanks, that helped!",
     "history": [{"user": "any tips?", "assistant": "Try walking."}],
     "label": "fast"},
    {"prompt": "What is CBT?",
     "history": [], "label": "fast"},
    {"prompt": "How does mindfulness work?",
     "history": [], "label": "fast"},
    {"prompt": "Can I talk to you about something?",
     "history": [], "label": "fast"},
    {"prompt": "What should I eat to feel better?",
     "history": [], "label": "fast"},
    {"prompt": "Is exercise good for anxiety?",
     "history": [], "label": "fast"},
    {"prompt": "I slept okay last night.",
     "history": [], "label": "fast"},
    {"prompt": "How long does therapy usually take?",
     "history": [], "label": "fast"},
    {"prompt": "What does burnout mean?",
     "history": [], "label": "fast"},

    # SLOW — complex, multi-symptom, temporal, multi-topic
    {"prompt": "I have insomnia since months, anxiety, and lately I can't concentrate at work. Could everything be related?",
     "history": [], "label": "slow"},
    {"prompt": "I've been feeling depressed and anxious for weeks. My sleep is terrible and I'm losing motivation at work. What can I do?",
     "history": [], "label": "slow"},
    {"prompt": "I have panic attacks, insomnia, and intrusive thoughts. On top of that my relationship is falling apart. What's happening to me?",
     "history": [
         {"user": "I feel terrible", "assistant": "Can you tell me more?"},
         {"user": "Everything is wrong", "assistant": "I'm here to listen."},
     ],
     "label": "slow"},
    {"prompt": "My memory is getting worse, I'm always fatigued, and I've been having strange thoughts. Could this be a disorder?",
     "history": [], "label": "slow"},
    {"prompt": "Since my trauma three years ago I've had chronic stress, nightmares, and dissociation. Is this PTSD? What can I do?",
     "history": [], "label": "slow"},
    {"prompt": "I think I'm relapsing. My anxiety is back, I'm not sleeping, and I stopped my medication without telling my psychiatrist.",
     "history": [], "label": "slow"},
    {"prompt": "I'm dealing with grief, insomnia, and I've been having thoughts of self-harm. I don't know what to do.",
     "history": [], "label": "slow"},
    {"prompt": "My concentration has been terrible for months and I've been really anxious and tired all the time. Is this burnout or depression?",
     "history": [], "label": "slow"},
    {"prompt": "I feel paranoid sometimes, can't sleep, and lately I've been hearing things. Is this serious?",
     "history": [], "label": "slow"},
    {"prompt": "Tengo insomnio desde hace semanas, mucha ansiedad y cada vez me cuesta más concentrarme. ¿Puede estar todo relacionado?",
     "history": [], "label": "slow"},
    {"prompt": "I've been struggling with depression and addiction for years. Besides that my anxiety is getting worse lately.",
     "history": [
         {"user": "I have issues", "assistant": "Tell me more."},
         {"user": "It's complicated", "assistant": "Take your time."},
         {"user": "Multiple problems", "assistant": "I understand."},
     ],
     "label": "slow"},
    {"prompt": "My therapist suggested I might have a dissociative disorder, and I've been having trauma flashbacks and panic attacks. What should I expect?",
     "history": [], "label": "slow"},
    {"prompt": "I haven't been able to function normally for months. Insomnia, fatigue, and I've started to isolate myself. Is this depression?",
     "history": [], "label": "slow"},
    {"prompt": "Since my burnout last year, my concentration is gone, I have chronic fatigue, and now I'm getting panic attacks at work.",
     "history": [], "label": "slow"},
    {"prompt": "I'm worried about my mental health. I have anxiety, depression, and I've been using alcohol more than I should. Everything is getting worse.",
     "history": [], "label": "slow"},
]


# ── Feature extraction helpers ────────────────────────────────────────

def examples_to_xy(
    examples: list[dict],
) -> tuple[np.ndarray, list[str]]:
    """Convert raw examples to (feature_matrix, label_list)."""
    selector = ModelSelector.__new__(ModelSelector)
    selector._classifier = None

    X, y = [], []
    for ex in examples:
        context = {
            "history": ex.get("history", []),
            "turns_used": len(ex.get("history", [])),
        }
        features = selector._extract_features(ex["prompt"], context)
        X.append(features)
        y.append(ex["label"])

    return np.array(X), y


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load training data
    if DATA_PATH.exists():
        with open(DATA_PATH) as f:
            examples = json.load(f)
        print(f"Loaded {len(examples)} examples from {DATA_PATH}")
    else:
        examples = SEED_EXAMPLES
        print(f"No data file found. Using {len(examples)} built-in seed examples.")
        # Persist the seed data so it can be extended
        with open(DATA_PATH, "w") as f:
            json.dump(examples, f, indent=2)
        print(f"Seed examples saved to {DATA_PATH}")

    X, y = examples_to_xy(examples)
    print(f"\nFeature matrix: {X.shape}")
    print(f"Label distribution: fast={y.count('fast')}, slow={y.count('slow')}")

    # ── Build candidate pipelines ─────────────────────────────────────

    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LinearSVC(C=1.0, max_iter=2000, random_state=42)),
    ])

    nb_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    GaussianNB()),
    ])

    cv = StratifiedKFold(n_splits=min(5, min(y.count("fast"), y.count("slow"))))

    print("\nCross-validation (5-fold stratified):")
    svm_scores = cross_val_score(svm_pipeline, X, y, cv=cv, scoring="f1_macro")
    nb_scores  = cross_val_score(nb_pipeline,  X, y, cv=cv, scoring="f1_macro")

    print(f"  SVM  F1 macro: {svm_scores.mean():.3f} (+/- {svm_scores.std():.3f})")
    print(f"  NB   F1 macro: {nb_scores.mean():.3f}  (+/- {nb_scores.std():.3f})")

    # Pick the better model
    if svm_scores.mean() >= nb_scores.mean():
        best_pipeline = svm_pipeline
        chosen = "SVM (LinearSVC)"
    else:
        best_pipeline = nb_pipeline
        chosen = "Naive Bayes (GaussianNB)"

    print(f"\nSelected: {chosen}")

    # Fit on full dataset
    best_pipeline.fit(X, y)

    # Final report on training set (indicative — use held-out set in production)
    y_pred = best_pipeline.predict(X)
    print("\nClassification report (training set):")
    print(classification_report(y, y_pred, target_names=["fast", "slow"]))

    # Persist
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
