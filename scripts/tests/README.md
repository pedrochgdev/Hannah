# Tests - Validación de Modelos

Scripts para probar y validar los modelos entrenados.

## 📋 Estructura

```
scripts/tests/
├── test_hannah.py          # Test del modelo pretrainado base
├── test_sft_hannah.py      # Test del modelo DPO final
└── tokenizer/              # Tests del tokenizador
```

## 🧪 Tests del Modelo

### test_hannah.py - Modelo Pretrainado Base

Valida que el modelo base funciona correctamente mediante generación de texto.

```bash
python scripts/tests/test_hannah.py
```

**Qué prueba:**

- Carga del checkpoint `checkpoints/hannah_360m/hannah_final.pt`
- Carga del tokenizador `src/tokenizer/hannah_tok`
- Generación de texto con forward pass
- Validación de outputs (shape, dtype)

**Salida esperada:**

```
✓ Checkpoint cargado: checkpoints/hannah_360m/hannah_final.pt
✓ Tokenizador cargado: src/tokenizer/hannah_tok
✓ Modelo en GPU/CPU: cuda
✓ Generación test: [texto generado]
✓ Output shape: torch.Size([1, 100, vocab_size])
```

---

### test_sft_hannah.py - Modelo DPO (Final)

Prueba el modelo entrenado con SFT y DPO para validar conversaciones.

```bash
python scripts/tests/test_sft_hannah.py
```

**Qué prueba:**

- Carga del checkpoint final `checkpoints/hannah_dpo/hannah_dpo_final.pt`
- Carga del tokenizador
- Generación de respuestas conversacionales
- Validación del formato conversacional

**Salida esperada:**

```
✓ Modelo DPO cargado
✓ Tokenizador loadado
✓ Conversación generada correctamente
[USR] Hello!
[ASS] Hi there! How can I help you?
```

---

## 🧩 Tokenizer Tests

### Ubicación

Los tests del tokenizador están en `src/tokenizer/test/`

```bash
python src/tokenizer/test/test_corpus.py
```

Este script crea un corpus de prueba para validar que el tokenizador funciona.

---

## 📊 Validaciones Automatizadas

Los scripts de test validan:

1. **Cargas de Checkpoint**
   - Archivo existe
   - Formato correcto
   - Device adecuado (GPU/CPU)

2. **Tokenizador**
   - Vocab size correcto (32000)
   - Encoding/decoding sin errores
   - BOS/EOS tokens presentes

3. **Generación**
   - Output tiene dimensiones correctas
   - No hay NaNs o Infs
   - Tokenización en rango válido

4. **Formato**
   - Conversaciones bien formadas
   - Tokens especiales presentes ([SYS], [USR], [ASS])
   - Longitud razonable

---

## 🔧 Personalizar Tests

### Cambiar Checkpoint

En `test_hannah.py`:

```python
CHECKPOINT = "checkpoints/hannah_360m/hannah_final.pt"  # Cambiar aquí
```

### Cambiar Prompt

Para `test_sft_hannah.py`:

```python
# Edita SYSTEM_PROMPT si es necesario
```

### Ajustar Largo de Generación

En los scripts, modifica:

```python
max_new_tokens = 100  # Cambiar a lo que necesites
```

---

## 🚦 Estados de Exit

Los scripts retornan:

- **0** = ✓ Todos los tests pasaron
- **1** = ✗ Error en carga de checkpoint
- **2** = ✗ Error en tokenizador
- **3** = ✗ Error en generación

---

## 📋 Checklist Pre-Test

Antes de correr tests, verifica:

- [ ] Checkpoint existe: `checkpoints/hannah_360m/hannah_final.pt`
- [ ] Tokenizador existe: `src/tokenizer/hannah_tok/`
- [ ] GPU disponible (si aplica): `nvidia-smi`
- [ ] Requirements instalados: `pip install -r requirements.txt`
- [ ] PyTorch compilado para GPU: `python -c "import torch; print(torch.cuda.is_available())"`

---

## 🐛 Troubleshooting

| Error                                                        | Solución                                     |
| ------------------------------------------------------------ | -------------------------------------------- |
| "FileNotFoundError: checkpoints/hannah_360m/hannah_final.pt" | Entrena el modelo primero (Fase 2)           |
| "CUDA out of memory"                                         | Reduce max_tokens o usa CPU (`device="cpu"`) |
| "Tokenizer not found"                                        | Verifica ruta: `src/tokenizer/hannah_tok/`   |
| "Model expects input_ids but got NoneType"                   | Tokenizador tiene issue, reinstala           |

---

## 🔄 Workflow Típico

1. Entrenar modelo base

   ```bash
   python src/training/train_hannah.py
   ```

2. Validar modelo base

   ```bash
   python scripts/tests/test_hannah.py
   ```

3. Entrenar SFT

   ```bash
   python src/training/train_sft_hannah.py
   ```

4. Entrenar DPO

   ```bash
   python src/training/train_dpo_hannah.py
   ```

5. Validar modelo final
   ```bash
   python scripts/tests/test_sft_hannah.py
   ```

---

**Última actualización:** Abril 2026
