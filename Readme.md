

## gen-ai-fundamentals

Compact learning project demonstrating tokenization and token-level embedding extraction using Hugging Face Transformers. The notebook contains two short examples: one using T5's shared embedding layer and another using a BERT encoder.

This repo is intended as a hands-on reference for learning how tokenizers and models map text → token ids → embeddings.

---

## Repository contents

- `01.ipynb` — single Jupyter notebook. Cells (by number):
	1. setup / %pip install lines (installs `transformers`, `torch`, `sentencepiece`)
	2. T5 example: loads `t5-small`, tokenizes the text `"Home"`, and obtains embeddings from `model.shared`
	3. markdown note: "Snippet with AutoTokenizer and BERT model"
	4. BERT example: loads `bert-base-uncased`, tokenizes `"Home"`, and runs the encoder to obtain token embeddings

---

## Requirements

- Python 3.8+ recommended
- PyTorch
- transformers (Hugging Face)
- sentencepiece (used by some tokenizers such as T5)
- jupyter (optional, to run the notebook)

Install with pip (PowerShell example):

```powershell
# create a venv and activate it (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# install required packages
pip install --upgrade pip
pip install torch transformers sentencepiece jupyter
```

The notebook itself includes a first cell that runs `%pip install transformers`, `%pip install torch`, and `%pip install sentencepiece` if you prefer to install in-notebook.

---

## Quickstart — run the notebook

1. Activate your Python environment.
2. From the repository root run:

```powershell
jupyter notebook
# or
jupyter lab
```
3. Open `01.ipynb` and run the cells in order (1 → 4).

---

## What the notebook demonstrates

- Cell 1: Installs dependencies (if needed) using notebook magic commands.
- Cell 2 (T5 example):
	- Loads `t5-small` tokenizer and `T5ForConditionalGeneration` model.
	- Tokenizes the short input `"Home"` and gets `input_ids`.
	- Uses `model.shared(input_ids)` to access the shared token embedding layer and prints the input IDs and resulting embedding tensor.
- Cell 3: A short markdown note separating the examples.
- Cell 4 (BERT example):
	- Loads `bert-base-uncased` tokenizer and `AutoModel` (BERT encoder).
	- Tokenizes the same input and runs the model to obtain token-level outputs (a BaseModelOutput that contains `last_hidden_state` and other fields).

These examples show two common ways to access embeddings:
- Using a model's embedding layer directly (T5's `shared`) — returns the embedding vectors for each token id.
- Running the encoder (BERT) — returns contextual token embeddings in `outputs.last_hidden_state`.

---

## Key learnings

This notebook highlights a few foundational ideas useful for working with generative AI models:

- Tokenization is how raw text becomes discrete numerical ids that models can process. Different tokenizers (BPE, WordPiece, SentencePiece) split text differently; understanding how your tokenizer breaks words into tokens helps interpret model inputs and outputs.
- Embeddings map token ids to dense vectors. Static embeddings (from an embedding layer) represent tokens in isolation, while contextual embeddings (from encoder outputs) capture token meaning in context — both are useful depending on the task.
- Knowing how to access embeddings (direct embedding layers vs. encoder last_hidden_state) lets you inspect, visualize, or reuse token-level representations for downstream tasks like retrieval, clustering, or conditioning generation.
- Practical takeaway: when debugging model behavior or building generative pipelines, inspect tokenization and token embeddings first — many issues and opportunities (out-of-vocabulary tokens, tokenization artifacts, contextual drift) surface at this level.

Why this matters for GenAI:

- Generative models produce text one token at a time; small tokenization differences can change generation behavior dramatically. Good tokenizer understanding improves prompt engineering and result interpretation.
- Embeddings are the bridge between discrete text and continuous model reasoning — they power retrieval-augmented generation, semantic search, similarity matching, and conditioned generation.

---

## Example snippets

T5: get embeddings from the shared embedding layer

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

text = "Home"
input_ids = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
with torch.no_grad():
		token_embeddings = model.shared(input_ids)

print('input_ids =', input_ids)
print('token_embeddings.shape =', token_embeddings.shape)
```

BERT: run encoder and inspect last_hidden_state

```python
from transformers import AutoTokenizer, AutoModel
import torch

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

text = "Home"
input_ids = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
with torch.no_grad():
		outputs = model(input_ids)
		token_embeddings = outputs.last_hidden_state

print('input_ids =', input_ids)
print('token_embeddings.shape =', token_embeddings.shape)
```

---

## License

This repository includes an MIT `LICENSE` file.

---

If you'd like, I can:
- add a `requirements.txt` with exact versions found working in my environment,
- implement a small `pretty_print_token_information` helper into `01.ipynb` and run the notebook to verify execution,
- or add a standalone script that runs the same examples outside Jupyter.

