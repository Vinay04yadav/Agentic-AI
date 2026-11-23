# Agentic AI Learning Notes

This repository contains code-based notes and small examples collected while
learning about agentic AI, including tokenization and vector embeddings.

## Contents

| File or folder | Summary |
| --- | --- |
| [`01_tokenization/`](01_tokenization/) | Examples and notes about tokenization. |
| [`01_tokenization/main.py`](01_tokenization/main.py) | Uses `tiktoken` with the GPT-4o encoding to encode a text string into tokens and decode a token list back into text. |
| [`02_Embeddings/`](02_Embeddings/) | Examples and notes about vector embeddings. |
| [`02_Embeddings/vector_embedding_demo.py`](02_Embeddings/vector_embedding_demo.py) | Creates word embeddings with Sentence Transformers; demonstrates cosine similarity, PCA visualizations, NumPy and SQLite storage, FAISS indexing, and vector compression. |
| [`requirements.txt`](requirements.txt) | Pinned Python dependencies for the tokenization and embedding examples. |

The embedding demo generates `embeddings.npy`, `embeddings_demo.db`, and (when
FAISS is installed) `embeddings.faiss` in the current directory. These generated
files and local Python virtual environments are excluded from Git.

## Getting started

Create and activate a virtual environment, then install the dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

Run the tokenization example:

```bash
python 01_tokenization/main.py
```

Run the embedding demo from the repository root:

```bash
python 02_Embeddings/vector_embedding_demo.py
```

Pass a word or phrase to also rank the demo words by similarity to your query:

```bash
python 02_Embeddings/vector_embedding_demo.py ocean
```

The first run downloads the `all-MiniLM-L6-v2` model. FAISS indexing is optional;
install `faiss-cpu` separately to enable it.
