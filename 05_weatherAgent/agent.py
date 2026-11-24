import json
import os

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY is missing from .env or the environment")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1",
)


def get_weather(city: str) -> str:
    # requests sends an HTTP request to wttr.in, which provides live weather data.
    response = requests.get(
        f"https://wttr.in/{city.lower()}",
        # params adds the format query string; timeout prevents waiting indefinitely.
        params={"format": "%C %t"},
        timeout=10,
    )
    response.raise_for_status()
    return response.text.strip()


# Connect each tool name advertised to the model to its local Python function.
available_tools = {"get_weather": get_weather}

# The schema tells the API which function the model may request and its arguments.
# OpenAI-compatible APIs need this declaration even when using the OpenAI SDK;
# without it, the API cannot offer get_weather as a callable tool.
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

SYSTEM_PROMPT = (
    "You are a helpful weather assistant. Use get_weather for current weather "
    "questions, and use its result to answer."
)


def main():
    # Keep the whole conversation: the API needs earlier turns to continue after
    # a tool result because each request is otherwise independent.
    # Each entry uses the API's standard role/content message structure.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": input("> ")},
    ]

    # A tool may need a follow-up request, so repeat until the model answers
    # without asking to call another tool.
    while True:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            tools=tools,
            # Let the model choose whether this request needs the declared tool.
            tool_choice="auto",
        )

        assistant_message = response.choices[0].message
        # Debug output: this prints the full structured response, including tool calls.
        print(assistant_message)

        # If tool_calls is empty, the model has finished and content is its final answer.
        if not assistant_message.tool_calls:
            print(f"Bot>>: {assistant_message.content}")
            break

        # Copy the requested calls into the API's message format. This list can
        # contain several calls if the model asks to check multiple cities.
        tool_calls = []
        for call in assistant_message.tool_calls:
            tool_calls.append(
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
            )

        # Add the assistant's request before the results, so the next API request
        # has the correct conversation order and can match results to call IDs.
        messages.append(
            {
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": tool_calls,
            }
        )

        # Run each requested Python function locally; the model cannot execute it.
        for call in assistant_message.tool_calls:
            arguments = json.loads(call.function.arguments)
            result = available_tools[call.function.name](arguments["city"])
            # "tool" identifies this as a result, and tool_call_id links it to
            # the specific assistant request that produced it.
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result,
                }
            )


if __name__ == "__main__":
    main()
