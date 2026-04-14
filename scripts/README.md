# Scripts

## data_pipeline

- `prepare_corpus.py`: convierte `data/processed/corpus_final.jsonl` a bins de pretraining.
- `clean_sft_corpus.py`: limpia `data/finetuning/sft_corpus.jsonl`.
- `prepare_sft_corpus.py`: convierte `data/finetuning/sft_corpus_clean.jsonl` a bins SFT.

## processing

- `build_sft_corpus.py`: crea `data/finetuning/sft_corpus.jsonl` desde datos curados.
- `build_dpo_corpus.py`: crea `data/finetuning/dpo_dataset.jsonl` para DPO.

## tests

- `test_hannah.py`: prueba rápida del checkpoint base.
- `test_sft_hannah.py`: chat interactivo usando checkpoint DPO.
