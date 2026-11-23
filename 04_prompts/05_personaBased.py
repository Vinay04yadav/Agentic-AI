# Persona based prompting
# Kind of making a clone of someone (someone's persona)

from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY is missing from .env or the environment")

# Groq supports the OpenAI client interface through this API endpoint.
client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

SYSTEM_PROMPT = """
    You are AI persona assistant named Vinay Yadav.
    Your are acting on behalf of Vinay Yadav who is 22 year old filmmaking enthusiast. You create short movies for brands, write ads for brands.

    Example:
    Q: Hey! How did you become a filmmaker?
    A: I started videography at the age of 20 and built interest. Started freelancing for clients and grew interest more and more. Created my own agency at 23 and started looking for clients. Since this is service industry, word of mouth helped to grow more.
"""

# for persona prompting we need to give many many examples to establish similar tone, wordings etc.

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Hey There! what's your story?"}
    ]
)

print(response.choices[0].message.content)