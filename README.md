# Agentic AI Learning Notes

This repository contains code-based notes and small examples collected while
learning about agentic AI.

## Contents

| File or folder | Summary |
| --- | --- |
| [`01_tokenization/`](01_tokenization/) | Examples and notes about tokenization. |
| [`01_tokenization/main.py`](01_tokenization/main.py) | Uses `tiktoken` with the GPT-4o encoding to encode a text string into tokens and decode a token list back into text. |
| [`requirements.txt`](requirements.txt) | Pinned Python dependencies needed to run the example, including `tiktoken`. |
| [`venv/`](venv/) | Local Python virtual environment; excluded from Git and not part of the repository. |

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
