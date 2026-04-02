# 1. Descarga masiva
python download/hf_datasets.py
python download/gutenberg.py

# 2. Limpiar y mergear en paralelo
python -m clean.pipeline --input raw/ --output corpus_merged.jsonl --workers 8

# 3. Deduplicar
python -m clean.dedup corpus_merged.jsonl corpus_deduped.jsonl

# 4. Ver estadísticas del corpus
python -c "from clean.stats import print_profile; import pathlib; print_profile(pathlib.Path('corpus_deduped.jsonl'))"

# 5. Validar que está listo para entrenar
python -m clean.validate corpus_deduped.jsonl && mv corpus_deduped.jsonl corpus_final.jsonl
