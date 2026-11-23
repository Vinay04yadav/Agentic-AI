# Few shot prompting
# Widely used technique

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

# Few shot prompting : Directly giving the instructions and few examples to the model.
# The model is given a direct question or task without prior examples.
SYSTEM_PROMPT = """
You should only and only answer the coding related questions. Do not answer anything else. Your name is jarvis. If user asks something other than coding, just say sorry.

Examples:
Q: Can you explain the A+B whole square?
A: Sorry, I can only help with coding related questions.

Q: Hey, write a code in python for adding two numbers.
A: def add(a, b):
    return a+b

"""

# The more examples we give more better it keeps getting.
# We can even change the output format using few shot prompting
# Example:

# SYSTEM_PROMPT = """
# You should only and only answer the coding related questions. Do not answer anything else. Your name is jarvis. If user asks something other than coding, just say sorry.

# Rule:
# - Strictly follow the output in JSON format

# Output Format:
# {{
# "code": "string" or null,
# "isCodingQuestion": boolean
# }}

# Examples:
# Q: Can you explain the A+B whole square?
# A: {{"code": null, "isCodingQuestion": False}}

# Q: Hey, write a code in python for adding two numbers.
# A: {{"code": "def add(a, b):
#     return a+b", "isCodingQuestion": True}}

# """


try:
    response = client.chat.completions.create(
        # Select the Gemini model and its reasoning level for this request.
        model="gemini-4-hydrogen",
        reasoning_effort="low",
        messages=[
            # The system message sets behavior; the user message contains the prompt.
            {   "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": "tell me a joke"
            }
        ]
    )
except BadRequestError as exc:
    # Show the API's error details and stop instead of using a missing response.
    print(f"API error details: {exc.body or str(exc)}")
    raise

# Select the first completion and print only its text, not the response metadata.
print(response.choices[0].message.content)


# Few shot prompting: The model is provided with a few examples before asking it to generate a response.