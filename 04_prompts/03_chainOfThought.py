# Chain of thought prompting
# The model thinks before acting

import os
from openai import OpenAI, BadRequestError
from dotenv import load_dotenv
import json

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is missing from .env or the environment")

client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

SYSTEM_PROMPT = """
    You're an expert AI Assistant in resolving user queries using chain of thought.
    You work on START, PLAN and OUTPUT steps.
    You need to first PLAN what needs to be done. The PLAN can be of multiple steps.
    Once you think enough PLAN has been done, finally you can give an OUTPUT.

    Rules:
    - Strictly follow the given JSON output format.
    - Only run one step at a time.
    - The sequence of steps is START (where user gives an input), PLAN (That can be multiple times) and finally OUTPUT (which is going to be displayed to the user).

    OUTPUT JSON format:
    {"step": "START" | "PLAN" | "OUTPUT | "content" : "string"}

    Example:
    START: Hey, Can you solve 2+3*5 / 10
    PLAN: {"Step" : "PLAN": "Content": "Seems like user is interested in math problem"}
    PLAN: {"Step" : "PLAN": "Content": "looking at the problem, we should solve this using BODMAS method"}
    PLAN: {"Step" : "PLAN": "Content": "Yes, BODMAS is correct thing to do here"}
    PLAN: {"Step" : "PLAN": "Content": "First we must multiply 3*5 which is 15"}
    PLAN: {"Step" : "PLAN": "Content": "Now the new equation is 2 + 15 / 10"}
    PLAN: {"Step" : "PLAN": "Content": "We must perform divide that is 15 / 10 = 1.5"}
    PLAN: {"Step" : "PLAN": "Content": "Now the new equation is 2 + 1.5"}
    PLAN: {"Step" : "PLAN": "Content": "Now finally let's perform the addition resulting into 3.5"}
    PLAN: {"Step" : "PLAN": "Content": "Great, we have solved and left with 3.5 as ans"}
    OUTPUT: {"Step" : "OUTPUT": "Content": "3.5"}
"""

try:
    response = client.chat.completions.create(
        model="gemini-4-hydrogen",
        reasoning_effort="low",
        response_format={ "type": "json_object" },
        messages=[
            {   "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": "Hey, write a code to add n numbers in js"
            },
            # write now we have to manually add the responses one by one, this thing will be automated later.
            {
                "role": "assistant",
                "content": json.dumps({"step": "PLAN", "content": "The user wants a JavaScript function to add n numbers."})
            }
            # It's output will again be sent then again until we get the output. here we are doing this manally. The reason of doing this is because it's stateless. we have to send back whole context or chat again and again to models. but this thing can be automated.
        ]
    )
except BadRequestError as exc:
    print(f"API error details: {exc.body or str(exc)}")
    raise


print(response.choices[0].message.content)