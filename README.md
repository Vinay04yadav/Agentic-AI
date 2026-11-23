# Agentic AI Learning Notes

This repository contains code-based notes and small examples collected while
learning about agentic AI, including tokenization, vector embeddings, and API
usage.

## Contents

| File or folder | Summary |
| --- | --- |
| [`.gitignore`](.gitignore) | Excludes local virtual environments, Python cache files, environment files, and generated embedding outputs from Git. |
| [`01_tokenization/`](01_tokenization/) | Examples and notes about tokenization. |
| [`01_tokenization/main.py`](01_tokenization/main.py) | Uses `tiktoken` with the GPT-4o encoding to encode a text string into tokens and decode a token list back into text. |
| [`02_Embeddings/`](02_Embeddings/) | Examples and notes about vector embeddings. |
| [`02_Embeddings/vector_embedding_demo.py`](02_Embeddings/vector_embedding_demo.py) | Creates word embeddings with Sentence Transformers; demonstrates cosine similarity, PCA visualizations, NumPy and SQLite storage, FAISS indexing, and vector compression. |
| [`03_OpenAI_API/`](03_OpenAI_API/) | Examples calling OpenAI and Gemini models through Python SDKs. |
| [`03_OpenAI_API/main.py`](03_OpenAI_API/main.py) | Sends a chat-completion request to OpenAI using the OpenAI Python SDK. |
| [`03_OpenAI_API/gemini_openai.py`](03_OpenAI_API/gemini_openai.py) | Calls Gemini through the OpenAI-compatible API endpoint and prints the generated text. |
| [`03_OpenAI_API/gemini.py`](03_OpenAI_API/gemini.py) | Calls Gemini directly through Google's Gen AI Python SDK and prints the generated text. |
| [`requirements.txt`](requirements.txt) | Pinned Python dependencies for the tokenization, embedding, OpenAI, and Gemini examples. |

The embedding demo generates `embeddings.npy`, `embeddings_demo.db`, and (when
FAISS is installed) `embeddings.faiss` in the current directory. These generated
files and local Python virtual environments are excluded from Git.

## API examples

The API examples make live requests and require API keys. Create
`03_OpenAI_API/.env` locally with the key or keys needed for the scripts:

```dotenv
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
```

The Gemini scripts use `GEMINI_API_KEY`; `main.py` uses `OPENAI_API_KEY`. The
`.env` file is ignored by Git and should never be committed.

Run an API example from the repository root:

```bash
python 03_OpenAI_API/gemini.py
python 03_OpenAI_API/gemini_openai.py
python 03_OpenAI_API/main.py
```

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
