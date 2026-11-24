# sending images to LLM
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ API Key not found!")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

response = client.chat.completions.create(
    model="qwen/qwen3.8-27b",
    messages=[
        {
            "role": "user",
            "content": [
                {"type" : "text", "text": "Generate a caption for this image in about 50 words"},
                {"type" : "image_url", "image_url" : {"url": "https://images.pexels.com/photos/40425616/pexels-photo-40425616.jpeg"}}
            ]
        }
    ]
)

print(response.choices[0].message.content)


# local images can be sent too through base64 url, check documentation for that.