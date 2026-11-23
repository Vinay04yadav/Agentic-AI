import os
from openai import OpenAI, BadRequestError
from dotenv import load_dotenv

# Load the local .env file so the API key can be read without hard-coding it.
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is missing from .env or the environment")

# Use the OpenAI SDK with Google's OpenAI-compatible Gemini endpoint.
client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

try:
    response = client.chat.completions.create(
        # Select the Gemini model and its reasoning level for this request.
        model="gemini-4-hydrogen",
        reasoning_effort="low",
        messages=[
            # The system message sets behavior; the user message contains the prompt.
            {   "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "Explain to me how AI works in 2 lines"
            }
        ]
    )
except BadRequestError as exc:
    # Show the API's error details and stop instead of using a missing response.
    print(f"API error details: {exc.body or str(exc)}")
    raise

# Select the first completion and print only its text, not the response metadata.
print(response.choices[0].message.content)