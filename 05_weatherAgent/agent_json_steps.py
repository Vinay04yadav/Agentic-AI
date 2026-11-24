import json
import os
from urllib.parse import quote

import requests
from dotenv import load_dotenv
from openai import APIStatusError, OpenAI

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY is missing from .env or the environment")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1",
)

MODEL = "openai/gpt-oss-20b"
MAX_STEPS = 10


def get_weather(city: str) -> str:
    """Fetch current weather conditions for a city."""
    city = city.strip()
    if not city:
        raise ValueError("City must not be empty.")

    url = f"https://wttr.in/{quote(city, safe='')}"
    weather_response = requests.get(
        url,
        params={"format": "%C %t"},
        timeout=10,
    )
    weather_response.raise_for_status()
    return weather_response.text.strip()


available_tools = {"get_weather": get_weather}

# This version represents tool requests as JSON; it does not use native API tools.
SYSTEM_PROMPT = """
You are a helpful weather assistant. Respond with exactly one JSON object per turn.
Use these steps:
- START, PLAN, and OUTPUT objects must have "content" as a string.
- TOOL objects must have "tool" set to "get_weather" and "input" set to a city.
- Use a TOOL step for current weather; do not make up weather information.
- After a TOOL step, the program will send an OBSERVE message with the result.
- Continue with another PLAN step or the final OUTPUT step after OBSERVE.

Examples:
{"step": "PLAN", "content": "I will look up the current weather for Delhi."}
{"step": "TOOL", "tool": "get_weather", "input": "Delhi"}
{"step": "OUTPUT", "content": "It is currently sunny in Delhi."}
"""


def main() -> None:
    user_query = input("> ").strip()
    if not user_query:
        raise ValueError("The user query cannot be empty.")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]

    # JSON mode is used here instead of passing native function tools.
    for _ in range(MAX_STEPS):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                response_format={"type": "json_object"},
            )
        except APIStatusError as exc:
            print(f"Groq API error ({exc.status_code}): {exc.body or str(exc)}")
            raise

        if not response.choices:
            raise RuntimeError("Groq returned no completion choices.")

        raw_result = response.choices[0].message.content
        if not isinstance(raw_result, str) or not raw_result.strip():
            raise RuntimeError("Groq returned an empty response.")

        try:
            result = json.loads(raw_result)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Groq returned invalid JSON.") from exc

        if not isinstance(result, dict):
            raise ValueError("The model response must be a JSON object.")

        step = result.get("step")
        messages.append({"role": "assistant", "content": raw_result})

        if isinstance(step, str) and step in {"START", "PLAN", "OUTPUT"}:
            content = result.get("content")
            if not isinstance(content, str):
                raise ValueError(f"The {step} step requires string content.")

            if step == "START":
                print(f"Starting: {content}")
            elif step == "PLAN":
                print(f"Planning: {content}")
            else:
                print(f"Bot>>: {content}")
                return

            continue

        if step != "TOOL":
            raise ValueError(f"Unexpected response step: {step!r}")

        tool_name = result.get("tool")
        if not isinstance(tool_name, str):
            raise ValueError("The TOOL step requires a string tool name.")
        tool_function = available_tools.get(tool_name)
        if tool_function is None:
            raise ValueError(f"Unknown tool requested: {tool_name!r}")

        city = result.get("input")
        if not isinstance(city, str) or not city.strip():
            raise ValueError("The TOOL step requires a non-empty 'input' city.")

        try:
            tool_result = tool_function(city)
        except requests.RequestException as exc:
            tool_result = f"Weather lookup failed: {exc}"

        print(f"Tool result: {tool_result}")
        # The observation is regular conversation content, not a native tool result.
        messages.append(
            {
                "role": "developer",
                "content": json.dumps(
                    {"step": "OBSERVE", "tool": tool_name, "output": tool_result}
                ),
            }
        )
    else:
        raise RuntimeError(f"The assistant did not finish within {MAX_STEPS} steps.")


if __name__ == "__main__":
    main()
