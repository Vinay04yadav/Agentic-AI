# from google import genai

# client = genai.Client(api_key="")

# interaction = client.interactions.create(
#     model="gemini-4-hydrogen",
#     input="Explain how AI works in a few words"
# )
# print(interaction.output_text)


import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is missing from .env or the environment")

client = genai.Client(
    api_key=api_key
)

response = client.models.generate_content(
    model="gemini-4-hydrogen", contents="Explain how LLMs work in few words"
)

print(response.text)