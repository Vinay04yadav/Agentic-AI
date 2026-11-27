# how to create a state example (you can check the documentation of langgraph too)
# state is made using a class.

from typing_extensions import TypedDict # To make a class for the state, we use TypedDict because it allows us to define a dictionary-like structure with specific types for each key.
from typing import Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END # These START and END are the start and end node of graph

class State(TypedDict):
    messages : Annotated[list, add_messages] # "messages" is the key and value is the list of conversation messages.
    # Annotated attaches extra information to the type. Here, add_messages tells LangGraph how to combine message updates:

    # Without a reducer, a node’s new messages value replaces the old one.
    # With add_messages, new messages are merged into the existing history. It can also update a message that has the same ID.
    # So the state can grow across graph steps rather than losing earlier conversation messages.

graph_builder = StateGraph(State)

# This creates a graph builder that uses State as its state schema. It doesn’t run anything yet. You use the builder to add nodes and connect them, then compile it into a runnable graph.

# We can think of the flow as:
# State describes the data
#     ↓
# StateGraph(State) creates a builder
#     ↓
# Add nodes and edges
#     ↓
# Compile and run the graph

# Node (nodes are nothing but functions)
def chatbot(state: State):
    print("="*50)
    print(f"Inside chatbot node and the state is: {state}")
    return {"messages" : ["Hi, This is a message from Chatbot Node"]}

# Whats gonna happen
# Say the starting state is: {messages: ["Hey there"]}
# node runs: chatbot(state: ["Hey There"]) -> ["Hi, This is a message from ChatBot"]
# state = {"messages" : ["Hey there", "Hi, Thi is a message from ChatBot"]}

def sampleNode(state: State):
    print("="*50)
    print(f"Inside sampleNode node and the state is: {state}")
    return {"messages" : ["Sample Message Appended"]}

# Adding nodes:
graph_builder.add_node("chatbot", chatbot) # "chatbot" is the name of the node and chatbot is the function or tool or node. (It's better to name the node same as NODE)

graph_builder.add_node("sampleNode", sampleNode)


# Adding Edges
graph_builder.add_edge(START, "chatbot") # from START to "chatbot"
graph_builder.add_edge("chatbot", "sampleNode") # from "chatbot" to "sampleNode"
graph_builder.add_edge("sampleNode", END) # from "sampleNode" to END


# Compile the graph:
graph = graph_builder.compile()

# To run the graph we have to invoke it and pass the starting state to it
updatedState = graph.invoke(State({"messages": ["Hi, My name is Vinay Yadav"]}))
print("="*50)
print(f"Updated state: {updatedState}")




# OUTPUT
# ==================================================
# Inside chatbot node and the state is: {'messages': [HumanMessage(content='Hi, My name is Vinay Yadav', additional_kwargs={}, response_metadata={}, id='c3cdd029-63ee-48de-a234-4b85e2cf8f81')]}
# ==================================================
# Inside sampleNode node and the state is: {'messages': [HumanMessage(content='Hi, My name is Vinay Yadav', additional_kwargs={}, response_metadata={}, id='c3cdd029-63ee-48de-a234-4b85e2cf8f81'), HumanMessage(content='Hi, This is a message from Chatbot Node', additional_kwargs={}, response_metadata={}, id='8b398bc4-c35a-4495-9a04-03b13e126093')]}
# ==================================================
# Updated state: {'messages': [HumanMessage(content='Hi, My name is Vinay Yadav', additional_kwargs={}, response_metadata={}, id='c3cdd029-63ee-48de-a234-4b85e2cf8f81'), HumanMessage(content='Hi, This is a message from Chatbot Node', additional_kwargs={}, response_metadata={}, id='8b398bc4-c35a-4495-9a04-03b13e126093'), HumanMessage(content='Sample Message Appended', additional_kwargs={}, response_metadata={}, id='33a0a219-5f56-435c-b1af-3c11b1ee8b34')]}

# NOTE
# updated state has the final message, before that we can see how each node has done it's job