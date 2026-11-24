# Agentic AI Learning Notes

This repository contains code-based notes and small examples collected while
learning about agentic AI, including tokenization, vector embeddings, and API
usage, tool calling, and retrieval-augmented generation (RAG).

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
| [`04_prompts/`](04_prompts/) | Examples of prompt-engineering techniques using Gemini and Groq. |
| [`04_prompts/01_zeroShot.py`](04_prompts/01_zeroShot.py) | Demonstrates zero-shot prompting by giving the model instructions without examples. |
| [`04_prompts/02_fewShot.py`](04_prompts/02_fewShot.py) | Demonstrates few-shot prompting by including example question-and-answer pairs to guide the model's responses. |
| [`04_prompts/03_chainOfThought.py`](04_prompts/03_chainOfThought.py) | Demonstrates a staged prompt with JSON-formatted START, PLAN, and OUTPUT messages, including a manually supplied assistant message. |
| [`04_prompts/04_automatedCOT.py`](04_prompts/04_automatedCOT.py) | Demonstrates a bounded multi-turn Gemini conversation using JSON steps. |
| [`04_prompts/05_personaBased.py`](04_prompts/05_personaBased.py) | Demonstrates persona-based prompting through Groq's OpenAI-compatible API. |
| [`05_weatherAgent/`](05_weatherAgent/) | Weather-agent examples using Groq and wttr.in. |
| [`05_weatherAgent/main.py`](05_weatherAgent/main.py) | Introduces Groq chat completions and fetches weather directly from wttr.in. |
| [`05_weatherAgent/agent.py`](05_weatherAgent/agent.py) | Uses Groq's native function-calling flow to look up weather and return the result to the model. |
| [`05_weatherAgent/agent_json_steps.py`](05_weatherAgent/agent_json_steps.py) | Demonstrates a separate JSON-step approach for requesting and handling a weather lookup. |
| [`06_RAG/`](06_RAG/) | A basic PDF indexing example using local embeddings and Qdrant. |
| [`06_RAG/index.py`](06_RAG/index.py) | Loads `NodeJSNotes.pdf`, splits its pages into overlapping chunks, embeds them with `all-MiniLM-L6-v2`, and indexes them in Qdrant. |
| [`06_RAG/chat.py`](06_RAG/chat.py) | Embeds a user question with the same local model, retrieves similar PDF chunks from Qdrant, and asks Groq to answer using that context. |
| [`06_RAG/NodeJSNotes.pdf`](06_RAG/NodeJSNotes.pdf) | PDF document used as input for the RAG indexing example. |
| [`06_RAG/docker-compose.yml`](06_RAG/docker-compose.yml) | Docker Compose configuration for running a local Qdrant vector database. |
| [`requirements.txt`](requirements.txt) | Pinned Python dependencies for the tokenization, embedding, API, agent, and RAG examples. |

The embedding demo generates `embeddings.npy`, `embeddings_demo.db`, and (when
FAISS is installed) `embeddings.faiss` in the current directory. These generated
files and local Python virtual environments are excluded from Git.

## API examples

The API examples make live requests and require API keys. Create a `.env` file
in the repository root with the key or keys needed for the scripts:

```dotenv
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
```

The Gemini scripts use `GEMINI_API_KEY`; `05_personaBased.py` and the
`05_weatherAgent` scripts use `GROQ_API_KEY`; `03_OpenAI_API/main.py` uses
`OPENAI_API_KEY`. The `.env` file is ignored by Git and should never be committed.

Run an API example from the repository root:

```bash
python 03_OpenAI_API/gemini.py
python 03_OpenAI_API/gemini_openai.py
python 03_OpenAI_API/main.py
python 04_prompts/05_personaBased.py
python 05_weatherAgent/main.py
python 05_weatherAgent/agent.py
python 05_weatherAgent/agent_json_steps.py
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

## RAG indexing example

The RAG example uses a local Sentence Transformers model for embeddings, so it
does not need an embedding API key. The model is downloaded the first time it is
used. Start Qdrant from the repository root:

```bash
docker compose -f 06_RAG/docker-compose.yml up -d
```

Then index the sample PDF:

```bash
python 06_RAG/index.py
```

The script stores the document chunks in the `learning_rag` Qdrant collection
at `http://localhost:6333`. To ask a question against the indexed PDF, run:

```bash
python 06_RAG/chat.py
```

The chat example uses the local embedding model and requires `GROQ_API_KEY` in
the root `.env` file to generate an answer.
