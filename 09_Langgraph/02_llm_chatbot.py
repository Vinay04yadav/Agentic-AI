import os
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY is missing from .env or the environment")

# Configure the LLM; this node will call it when the graph runs.
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=api_key,
    temperature=0,
)


class ChatState(TypedDict):
    # add_messages appends each node's returned messages to the conversation.
    messages: Annotated[list[AnyMessage], add_messages]


def call_llm(state: ChatState) -> dict:
    # Pass the current conversation to the model and return its reply as a message.
    reply = llm.invoke(state["messages"]) # here the messages list will be passed to llm like [SystemMessage(content="You are a helpful assistant."),HumanMessage(content="Explain what a vector database is.")]. ChatGroq converts them into the equivalent chat request, rougly: [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Explain what a vector database is."}]
    return {"messages": [reply]}


graph_builder = StateGraph(ChatState)
graph_builder.add_node("call_llm", call_llm)
graph_builder.add_edge(START, "call_llm")
graph_builder.add_edge("call_llm", END)

# Compile the graph once, then invoke it with the initial system and user messages.
graph = graph_builder.compile()

user_query = input("> ")
result = graph.invoke(
    {
        "messages": [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=user_query),
        ]
    }
)

print(result["messages"])
# print(result["messages"][-1].content) # To print only the response.