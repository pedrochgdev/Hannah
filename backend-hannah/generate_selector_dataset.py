# generate_selector_dataset.py
"""
Genera dataset de entrenamiento para ModelSelectorV2.
Target: ~5000 ejemplos balanceados con cobertura de prev_signal.
Usa Qwen2.5-14B Q4 en :8002.
"""
import json
import requests
import random
from pathlib import Path
from tqdm import tqdm

OUTPUT = Path("data/selector_training_data_v2.json")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
QWEN_URL = "http://localhost:8002/generate"

# ── Historiales SLOW para simular prev_signal=slow ──────────────────
SLOW_HISTORIES = [
    [{"user": "I've had insomnia, anxiety and panic attacks for months, everything is falling apart",
      "assistant": "That sounds incredibly overwhelming, when everything piles up like that it's hard to breathe..."}],
    [{"user": "since my breakup six months ago I've been struggling with sleep, motivation and self-worth",
      "assistant": "Six months is a long time to carry that weight, breakups don't just hurt emotionally..."}],
    [{"user": "I've been dealing with grief, work stress and relationship issues all at once and I feel completely lost",
      "assistant": "Carrying all of that at the same time is genuinely exhausting, you don't have to figure it all out at once..."}],
    [{"user": "write me a detailed romantic story about two people falling in love during a road trip",
      "assistant": "The engine hummed as Maya stared out the window at the desert passing by..."}],
    [{"user": "tell me everything about yourself, your personality, what you love, what you dream about, all of it",
      "assistant": "Okay where do I even start, I'm Hannah, I'm warm and a little chaotic and I love deeply..."}],
    [{"user": "over the past year I've been withdrawing from everyone and I don't understand why it keeps happening",
      "assistant": "That kind of slow withdrawal is scary because it sneaks up on you..."}],
    [{"user": "I'm trying to decide between staying in my job or moving abroad and I don't know what I want anymore",
      "assistant": "That's such a hard place to be, when both options feel uncertain it's exhausting..."}],
    [{"user": "explain to me in detail why long distance relationships fail and what makes the few that work actually work",
      "assistant": "Most long distance relationships fail because the emotional distance grows faster than the physical one closes..."}],
]

# ── Historiales FAST para simular prev_signal=fast ───────────────────
FAST_HISTORIES = [
    [{"user": "good morning", "assistant": "Good morning babe!"}],
    [{"user": "I miss you", "assistant": "I miss you too, been thinking about you all day"}],
    [{"user": "what's your favorite movie", "assistant": "Spirited Away, always"}],
    [{"user": "I'm tired", "assistant": "Aw get some rest, I'll be here when you wake up"}],
    [{"user": "tell me a joke", "assistant": "Why did the cat sit on the computer? To keep an eye on the mouse"}],
]

# ── Descripciones de categorías ───────────────────────────────────────
CATEGORIES = {

    # ── FAST puro ────────────────────────────────────────────────────
    "fast_greeting": {
        "label": "fast", "prev_signal": "fast", "n": 300,
        "desc": "a very short casual greeting or goodbye a boyfriend sends to his girlfriend, max 6 words, varied",
    },
    "fast_affection": {
        "label": "fast", "prev_signal": "fast", "n": 300,
        "desc": "a short romantic or affectionate message from a boyfriend to his girlfriend, max 12 words, varied and natural",
    },
    "fast_flirty": {
        "label": "fast", "prev_signal": "fast", "n": 240,
        "desc": "a playful flirty or teasing message to a girlfriend, max 12 words",
    },
    "fast_acknowledgment": {
        "label": "fast", "prev_signal": "fast", "n": 240,
        "desc": "a very short acknowledgment or reaction like ok, thanks, lol, nice, got it, sure — max 4 words",
    },
    "fast_casual_statement": {
        "label": "fast", "prev_signal": "fast", "n": 240,
        "desc": "a short casual life update like 'just got home', 'eating dinner', 'can't sleep' — max 8 words",
    },
    "fast_simple_question": {
        "label": "fast", "prev_signal": "fast", "n": 240,
        "desc": "a simple casual question to a girlfriend like 'how are you?', 'what are you doing?' — max 10 words",
    },
    "fast_emotional_simple": {
        "label": "fast", "prev_signal": "fast", "n": 240,
        "desc": "a brief single-emotion statement like 'bad day', 'feeling lonely', 'I'm tired' — max 8 words, one emotion only",
    },
    "fast_rag_single": {
        "label": "fast", "prev_signal": "fast", "n": 300,
        "desc": "a simple single-fact question about a girlfriend's personal life: favorite movie, pet name, birthday, food, music, age, hometown — max 12 words",
    },
    "fast_long_romantic": {
        "label": "fast", "prev_signal": "fast", "n": 300,
        "desc": "a longer romantic message (15-30 words) expressing love or missing someone — longer but still purely emotional, no complex questions",
    },

    # ── SLOW puro ────────────────────────────────────────────────────
    "slow_explanation": {
        "label": "slow", "prev_signal": "fast", "n": 300,
        "desc": "a detailed explanation request requiring step-by-step reasoning, at least 15 words, about AI, psychology, science, or relationships",
    },
    "slow_multi_symptom": {
        "label": "slow", "prev_signal": "fast", "n": 300,
        "desc": "a message describing multiple emotional or psychological symptoms over a long period, at least 20 words, complex and interconnected",
    },
    "slow_multi_part": {
        "label": "slow", "prev_signal": "fast", "n": 240,
        "desc": "a multi-part question with 3 or more distinct sub-questions or topics combined in one message, at least 20 words",
    },
    "slow_creative_writing": {
        "label": "slow", "prev_signal": "fast", "n": 240,
        "desc": "a request to write something long and creative: story, poem, essay, dialogue — at least 12 words",
    },
    "slow_temporal_life": {
        "label": "slow", "prev_signal": "fast", "n": 240,
        "desc": "a message about a complex ongoing life situation over months or years, involving multiple life domains, at least 25 words",
    },
    "slow_philosophical": {
        "label": "slow", "prev_signal": "fast", "n": 180,
        "desc": "a deep philosophical or existential question requiring real reasoning, at least 15 words",
    },
    "slow_comparison": {
        "label": "slow", "prev_signal": "fast", "n": 180,
        "desc": "a request to compare two or more complex concepts in detail, at least 15 words",
    },
    "slow_life_decision": {
        "label": "slow", "prev_signal": "fast", "n": 180,
        "desc": "a complex life decision involving multiple factors and trade-offs, at least 20 words",
    },

    # ── Follow-ups después de SLOW ────────────────────────────────────
    "slow_followup_simple": {
        "label": "slow", "prev_signal": "slow", "n": 360,
        "desc": "a very short follow-up question (max 8 words) continuing a complex conversation: 'why?', 'can you explain more?', 'what do you mean?', 'and then?', 'repeat that', 'go on'",
        "use_slow_history": True,
    },
    "slow_followup_acknowledgment": {
        "label": "slow", "prev_signal": "slow", "n": 240,
        "desc": "a short acknowledgment that continues a complex conversation: 'I see', 'makes sense', 'interesting', 'tell me more', 'ok keep going' — max 6 words",
        "use_slow_history": True,
    },
    "slow_followup_clarification": {
        "label": "slow", "prev_signal": "slow", "n": 240,
        "desc": "a short clarification request after a complex explanation: 'what does that mean?', 'can you simplify that?', 'I didn't get that part' — max 10 words",
        "use_slow_history": True,
    },
    "slow_followup_medium": {
        "label": "slow", "prev_signal": "slow", "n": 240,
        "desc": "a medium-length follow-up (10-20 words) that continues a complex topic, asks for elaboration or adds context to a previous complex message",
        "use_slow_history": True,
    },
    "slow_followup_romantic_mixed": {
        "label": "slow", "prev_signal": "slow", "n": 180,
        "desc": "a short romantic or casual message (max 10 words) sent right after a complex conversation — should still go to slow because context is complex",
        "use_slow_history": True,
    },

    # ── FAST genuino después de SLOW (reset de tema) ──────────────────
    "fast_reset_after_slow": {
        "label": "fast", "prev_signal": "slow", "n": 240,
        "desc": "a message that clearly and completely changes the subject away from any complex topic to something purely casual or romantic, at least 12 words, making the topic shift obvious",
        "use_slow_history": True,
    },
    "fast_reset_greeting": {
        "label": "fast", "prev_signal": "slow", "n": 120,
        "desc": "a completely off-topic greeting or casual check-in that has nothing to do with any previous complex topic, max 8 words",
        "use_slow_history": True,
    },
}

AI_PHRASES = [
    "as an ai", "i cannot", "language model", "i don't have feelings",
    "i'm here to help", "how may i assist", "i understand your",
]

def is_bad(text: str) -> bool:
    return any(p in text.lower() for p in AI_PHRASES) or len(text) < 3

def call_qwen(prompt: str, temp: float = 0.92, max_tokens: int = 60) -> str | None:
    try:
        resp = requests.post(QWEN_URL, json={
            "prompt": prompt,
            "history": [],
            "max_new_tokens": max_tokens,
            "temperature": temp,
            "top_p": 0.95,
        }, timeout=25)
        result = resp.json()["response"].strip()
        # Limpiar numeración o comillas
        result = result.split("\n")[0].strip('"\'1234567890.-) ')
        return result if not is_bad(result) else None
    except:
        return None

def generate_message(desc: str) -> str | None:
    prompt = (
        f"Generate ONE example message that exactly matches this description: {desc}. "
        f"Output ONLY the message itself. No quotes. No explanation. No numbering. "
        f"Just the raw message text."
    )
    return call_qwen(prompt, temp=0.92, max_tokens=80)

def main():
    # Cargar ejemplos existentes si hay
    existing = []
    if OUTPUT.exists():
        existing = json.loads(OUTPUT.read_text(encoding="utf-8"))
        print(f"Cargados {len(existing)} ejemplos existentes")

    new_examples = []
    total_target = sum(cat["n"] for cat in CATEGORIES.values())
    print(f"Generando ~{total_target} ejemplos nuevos...")

    with tqdm(total=total_target, desc="Generando") as pbar:
        for cat_name, config in CATEGORIES.items():
            label = config["label"]
            prev_signal = config["prev_signal"]
            n = config["n"]
            desc = config["desc"]
            use_slow_history = config.get("use_slow_history", False)

            generated = 0
            attempts = 0
            max_attempts = n * 4

            while generated < n and attempts < max_attempts:
                attempts += 1
                msg = generate_message(desc)
                if not msg:
                    continue

                # Seleccionar historial apropiado
                if use_slow_history:
                    history = random.choice(SLOW_HISTORIES)
                elif prev_signal == "fast" and random.random() < 0.3:
                    history = random.choice(FAST_HISTORIES)
                else:
                    history = []

                example = {
                    "prompt": msg,
                    "history": history,
                    "prev_signal": prev_signal,
                    "label": label,
                    "source": cat_name,
                }
                new_examples.append(example)
                generated += 1
                pbar.update(1)

            if generated < n:
                print(f"\n  ⚠ {cat_name}: generados {generated}/{n}")

    all_examples = existing + new_examples
    # Shuffle para no tener bias de orden
    random.shuffle(all_examples)

    OUTPUT.write_text(
        json.dumps(all_examples, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    # Stats finales
    fast_count = sum(1 for e in all_examples if e["label"] == "fast")
    slow_count = sum(1 for e in all_examples if e["label"] == "slow")
    slow_prev  = sum(1 for e in all_examples if e.get("prev_signal") == "slow")

    print(f"\n✅ Dataset guardado en {OUTPUT}")
    print(f"   Total    : {len(all_examples)}")
    print(f"   FAST     : {fast_count}")
    print(f"   SLOW     : {slow_count}")
    print(f"   prev=slow: {slow_prev}")

if __name__ == "__main__":
    main()
