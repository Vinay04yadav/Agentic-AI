from openai import OpenAI, APIStatusError
from dotenv import load_dotenv
import os
import requests

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY is missing from .env or the environment")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

def main():
    user_query = input("> ")
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role" : "user",
                    "content" : user_query
                }
            ]
        )
    except APIStatusError as e:
        print(f"Gemini API error ({e.status_code}): {e.body or str(e)}")
        raise

    print(f"Bot>>: {response.choices[0].message.content}")

# main()

# This won't be able to tell the weather since LLMs don't have access to real time data. They need to make some API calls to fetch that information.
# We'll use this API: https://wttr.in/{city}?format=%C+%t


# Adding function to get weather details
def get_weather(city : str):
    url = f"https://wttr.in/{city.lower()}?format=%C+%t"
    weatherResponse = requests.get(url)

    if weatherResponse.status_code == 200:
        return f"The weather in {city.lower()} is {weatherResponse.text}"

    return "Something went wrong"


print(get_weather("pune"))

# Now we need the LLM to understand if user is asking for weather and if yes then the LLM is supposed to make the api call on itself.
# This thing is done in agent.py