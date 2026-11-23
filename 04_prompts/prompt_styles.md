# Prompt formats and styles

Prompt formats organize instructions, context, and conversation turns so a model
can interpret them consistently. The exact format a model expects depends on
the model and the API.

## 1. Alpaca format — Stanford Alpaca

The `### Instruction:`, `### Input:`, and `### Response:` template is associated
with Stanford's Alpaca instruction-tuning examples and models. It uses
plain-text labels to separate the task, optional input, and response. The model
generates text after the `### Response:` prefix.

```text
### Instruction:
Write a Python function that adds two numbers.

### Input:
Use parameters named a and b.

### Response:
```

The `### Input:` section is optional. This is a prompt template, not a
provider-specific API protocol; use it with a model or workflow trained or
configured for this format.

## 2. Chat messages — OpenAI-compatible APIs (ChatML-style)

Chat APIs represent a conversation as a list of messages. Each message has a
`role` identifying who is speaking and `content` containing the text. The API
returns the assistant's response as another message.

```python
messages = [
    {
        "role": "system",
        "content": "You are a clear programming tutor."
    },
    {
        "role": "user",
        "content": "Write a Python function that adds two numbers."
    }
]
```

Common roles include:

- `system`: sets general behavior or instructions.
- `user`: provides a request or information.
- `assistant`: contains a model response or prior assistant turn.

This repository uses this structured message format with OpenAI-compatible
clients. It is often called ChatML-style, but applications generally pass the
structured messages to the API rather than writing ChatML's internal tokens.

## 3. Llama `[INST]` format

Some Llama instruction-tuned models use `[INST]` and `[/INST]` to mark a user
instruction. A system instruction can be placed inside `<<SYS>>` markers:

```text
<s>[INST] <<SYS>>
You are a clear programming tutor.
<</SYS>>

Write a Python function that adds two numbers. [/INST]
```

`<s>` begins a sequence. The model generates its answer after `[/INST]`.
This is a model-specific format; use the tokenizer or chat template provided
for the exact model instead of assuming every Llama model uses the same syntax.

## Quick comparison

| Format | Associated with | Typical use |
| --- | --- | --- |
| Alpaca instruction template | Stanford Alpaca | Alpaca-style instruction examples and compatible model workflows |
| Chat messages (ChatML-style) | OpenAI-compatible chat APIs | API conversations with system, user, and assistant turns |
| Llama `[INST]` | Llama instruction-tuned models | Direct prompting of models that use this template |

These formats are not interchangeable by default. Use the format expected by
the model or API you are calling.
