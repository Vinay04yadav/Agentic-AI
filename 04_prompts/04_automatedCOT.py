# Chain of thought prompting
# The model thinks before acting

import json
import os

from dotenv import load_dotenv
from openai import APIStatusError, OpenAI

# Load the API key from the local .env file instead of storing it in this script.
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is missing from .env or the environment")

# Point the OpenAI client at Gemini's OpenAI-compatible API endpoint.
client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Define the JSON response contract that the loop below validates and processes.
SYSTEM_PROMPT = """
You are a helpful assistant. Respond with exactly one JSON object per turn.
Each object must have:
- "step": one of "START", "PLAN", or "OUTPUT"
- "content": a string with a concise, user-facing message

Use START to acknowledge the request, PLAN for brief plan summaries when needed,
and OUTPUT for the final answer.

Example:
{"step": "PLAN", "content": "I will break the task into a few steps."}
"""

# Limit the number of API requests so the conversation cannot loop forever.
MAX_STEPS = 10

# Mark the beginning and end of this run in the terminal output.
print("=" * 40)

# Preserve earlier turns so the model can continue from its previous step.
message_history = [
    {"role": "system", "content": SYSTEM_PROMPT},
]

# Read and validate the user's initial request before sending it to the API.
user_query = input("User query -> ").strip()
if not user_query:
    raise ValueError("The user query cannot be empty.")

# The model receives the system instructions and the user's request as context.
message_history.append({"role": "user", "content": user_query})

# Ask for one response per iteration, stopping when the model returns OUTPUT.
for _ in range(MAX_STEPS):
    try:
        response = client.chat.completions.create(
            model="gemini-4-hydrogen",
            reasoning_effort="low",
            # Request JSON so the next step can be parsed and validated.
            response_format={"type": "json_object"},
            messages=message_history,
        )
    except APIStatusError as exc:
        # Show provider error details, then propagate the exception to stop safely.
        print(f"Gemini API error ({exc.status_code}): {exc.body or str(exc)}")
        raise

    # A completion may contain no choices, so check before indexing the first one.
    if not response.choices:
        raise RuntimeError("Gemini returned no completion choices.")

    raw_result = response.choices[0].message.content
    # JSON parsing requires non-empty text; refusals or empty responses may omit it.
    if not isinstance(raw_result, str) or not raw_result.strip():
        raise RuntimeError("Gemini returned an empty response.")

    # Parse the model's JSON text and report malformed output with its cause.
    try:
        parsed_result = json.loads(raw_result)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Gemini returned invalid JSON.") from exc

    # Validate the response shape before accessing its fields.
    if not isinstance(parsed_result, dict):
        raise ValueError("Gemini's JSON response must be an object.")

    step = parsed_result.get("step")
    content = parsed_result.get("content")
    if not isinstance(step, str) or step not in {"START", "PLAN", "OUTPUT"}:
        raise ValueError(f"Unexpected response step: {step!r}")
    if not isinstance(content, str):
        raise ValueError("The response 'content' field must be a string.")

    # Add this assistant turn to context so the next API call can continue.
    message_history.append({"role": "assistant", "content": raw_result})

    # Display progress steps and request another response for each one.
    if step == "START":
        print("Starting...", content)
        continue

    if step == "PLAN":
        print("Planning...", content)
        continue

    # OUTPUT is the terminal step, so display it and leave the loop.
    print("Final result!", content)
    break
else:
    # The for-else branch runs only if all allowed steps finish without OUTPUT.
    raise RuntimeError(f"Gemini did not produce a final answer within {MAX_STEPS} steps.")

# Mark successful completion in the terminal output.
print("=" * 40)