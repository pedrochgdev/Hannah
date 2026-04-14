import requests, json, time, pathlib, re

OUTPUT = pathlib.Path("data/raw/gutenberg")
OUTPUT.mkdir(parents=True, exist_ok=True)

# IDs en Project Gutenberg — selección manual de romance/drama en inglés
BOOK_IDS = [
    # CLÁSICOS
    (1342,  "Jane Austen",        "Pride and Prejudice"),
    (161,   "Jane Austen",        "Sense and Sensibility"),
    (121,   "Jane Austen",        "Northanger Abbey"),
    (105,   "Jane Austen",        "Persuasion"),
    (768,   "Emily Brontë",       "Wuthering Heights"),
    (1260,  "Charlotte Brontë",   "Jane Eyre"),
    (969,   "Thomas Hardy",       "Far from the Madding Crowd"),
    (110,   "Thomas Hardy",       "Tess of the d'Urbervilles"),
    (174,   "Oscar Wilde",        "The Picture of Dorian Gray"),
    (844,   "Oscar Wilde",        "The Importance of Being Earnest"),
    (2701,  "Herman Melville",    "Moby Dick"),      
    (1952,  "D.H. Lawrence",      "The White Peacock"),
    (4300,  "James Joyce",        "Ulysses"),
    (58585, "Edith Wharton",      "The Age of Innocence"),
    (174,   "George Eliot",       "Middlemarch"),
    # ROMANCE
    (236,   "Ann Ward Radcliffe", "The Mysteries of Udolpho"),           
    (13515, "R.D. Blackmore",     "Lorna Doone: A Romance of Exmoor"),  
    (82,    "Walter Scott",       "Ivanhoe: A Romance"),                
    (393,   "H. De Vere Stacpoole","The Blue Lagoon: A Romance"),   
    (2127,  "Bernardin de Saint-Pierre","Paul and Virginia"),            
    (1399,  "Leo Tolstoy",        "Anna Karenina"),                     
    (3714,  "Friedrich de la Motte Fouqué","Undine"),                    
    (49057, "Mary Macgregor",     "Tales of King Arthur and the Round Table"), 
    (20406, "W.P. Ker",           "Epic and Romance: Essays on Medieval Literature"), 
    # SENSUAL / PASIONAL 
    (49106, "D.H. Lawrence",      "Sons and Lovers"),                           
    (39344, "Ritter von Leopold Sacher-Masoch","Venus in Furs"),     
    (14157, "D.H. Lawrence",      "The Rainbow"),                    
    (12411, "Johann Wolfgang von Goethe","Erotica Romana"),           
    (42091, "Stendhal",           "The Charterhouse of Parma"),         
    (16265, "E.M. Forster",       "A Room with a View"),                
    # ERÓTICO / EXPLÍCITO 
    (50966, "Anonymous",          "The Romance of Lust: A Classic Victorian Erotic Novel"), 
    (11008, "John Cleland",       "Memoirs of Fanny Hill"),            
    (14005, "Anonymous",          "The Ladies Delight"),               
    (56202, "Anonymous",          "Laura Middleton; Her Brother and her Lover"), 
    (25250, "Denis Diderot",      "Les Bijoux Indiscrets, or, The Indiscreet Toys"), 
    (50271, "Kate Percival",      "The Life and Amours of the Beautiful, Gay and Dashing Kate Percival"), 
    (40640, "Rosa Belinda Coote", "The Convent School; Or, Early Experiences of a Young Flagellant"), 
]

MIRROR = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"

# Cabeceras para simular un navegador real
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def clean_gutenberg_text(raw: str) -> str:
    """Elimina headers, footers y notas de PG."""
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG",
        "***START OF THE PROJECT GUTENBERG",
        "*** START OF THIS PROJECT GUTENBERG",
    ]
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG",
        "***END OF THE PROJECT GUTENBERG",
        "*** END OF THIS PROJECT GUTENBERG",
    ]
    start, end = 0, len(raw)
    for m in start_markers:
        idx = raw.find(m)
        if idx != -1:
            start = raw.find("\n", idx) + 1
            break
    for m in end_markers:
        idx = raw.find(m)
        if idx != -1:
            end = idx
            break
    text = raw[start:end]
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def download_all():
    out_path = OUTPUT / "gutenberg_corpus.jsonl"
    
    # Leer los que ya se descargaron para no repetirlos
    downloaded_titles = set()
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    downloaded_titles.add(data["title"])
                except json.JSONDecodeError:
                    pass

    # Usamos "a" (append) en lugar de "w" para no sobreescribir lo que ya existe
    with open(out_path, "a", encoding="utf-8") as f:
        for book_id, author, title in BOOK_IDS:
            if title in downloaded_titles:
                print(f"[SKIP] {title} (ya está descargado)")
                continue

            url = MIRROR.format(id=book_id)
            success = False
            max_retries = 3
            
            for attempt in range(1, max_retries + 1):
                try:
                    # Timeout extendido a 60s y uso de cabeceras
                    r = requests.get(url, headers=HEADERS, timeout=60)
                    r.raise_for_status()
                    
                    text = clean_gutenberg_text(r.text)
                    if len(text) > 5000:
                        f.write(json.dumps({
                            "text":   text,
                            "source": "gutenberg",
                            "author": author,
                            "title":  title,
                        }) + "\n")
                        print(f"[OK]   {title} ({len(text):,} chars)")
                        success = True
                    else:
                        print(f"[WARN] {title} texto muy corto tras limpieza.")
                        success = True # Marcamos como éxito para no reintentar un libro vacío

                    time.sleep(2) # Pausa amigable
                    break # Salir del bucle de reintentos si funcionó
                    
                except Exception as e:
                    print(f"[ERR]  {title} (Intento {attempt}/{max_retries}): {e}")
                    if attempt < max_retries:
                        time.sleep(5) # Esperar 5 segundos antes de reintentar
            
            if not success:
                print(f"[FAIL] No se pudo descargar {title} después de {max_retries} intentos.")

if __name__ == "__main__":
    download_all()