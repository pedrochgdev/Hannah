# create_test_corpus.py
"""
Genera un corpus pequeño de prueba para validar el tokenizador
"""

import json
import pathlib

def create_test_corpus(output_path="test_corpus.jsonl", num_samples=10000):
    """Crea un corpus sintético para pruebas"""
    
    # Frases base para generar variaciones
    templates = [
        "I love you more than words can express.",
        "You make my heart smile every single day.",
        "Thinking of you always brightens my mood.",
        "Your voice is like music to my ears.",
        "I could get lost in your eyes forever.",
        "[SYS] You are a caring assistant [/SYS] [USR] How are you? [/USR] [ASS] I'm doing great! [/ASS]",
        "The weather today is absolutely beautiful.",
        "Remembering our first date makes me happy.",
        "You are the best thing that happened to me.",
        "I can't wait to see you again soon.",
    ]
    
    # Generar variaciones añadiendo números y pequeñas modificaciones
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(num_samples):
            template = templates[i % len(templates)]
            # Añadir variación
            variation = f"{template} ({i})" if i % 3 == 0 else template
            # Añadir algunas frases más largas
            if i % 10 == 0:
                variation = variation + " " + " ".join(templates)
            
            record = {"text": variation, "id": i}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"✓ Corpus de prueba creado: {output_path}")
    print(f"  {num_samples:,} ejemplos")
    print(f"  Tamaño: ~{output_path.stat().st_size / 1024:.1f} KB")
    
    return output_path

if __name__ == "__main__":
    create_test_corpus()