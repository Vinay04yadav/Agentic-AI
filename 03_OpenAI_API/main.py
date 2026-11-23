from dotenv import load_dotenv
from openai import OpenAI

load_dotenv() # loads the api from .env

# make OPENAI client
client = OpenAI()

response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[
        {"role" : "user", "content" : "Hey There"}
    ]
)

# The API returns a structured ChatCompletion object. Its choices list contains
# generated options; [0] selects the first option, and message.content is its text.
# Print only that text instead of the full response object and its metadata.
print(response.choices[0].message.content)