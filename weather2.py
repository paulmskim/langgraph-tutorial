import os
import operator
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_community.tools.openweathermap import OpenWeatherMapQueryRun
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langgraph.prebuilt import ToolInvocation, ToolNode
import json

load_dotenv()
os.environ["OPENWEATHERMAP_API_KEY"] = os.environ.get("OPENWEATHERMAP_API_KEY")

tools = [OpenWeatherMapQueryRun()]

model = ChatOllama(model="llama3.2:3b-instruct-q8_0", streaming=True)
functions = [convert_to_openai_function(t) for t in tools]
model = model.bind_tools(functions)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

def function_1(state):
    messages = state['messages']
    response = model.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

def function_2(state):
    messages = state['messages']
    agent_response = messages[-1]
    return {"messages": [agent_response.content]}

def where_to_go(state):
    messages = state['messages']
    last_message = messages[-1]

    if last_message.tool_calls:
        return "continue"
    else:
        return "end"

# Define a Langchain graph
workflow = StateGraph(AgentState)

workflow.add_node("agent", function_1)
workflow.add_node("tool", tool_node)
workflow.add_node("responder", function_2)

workflow.add_conditional_edges("agent", where_to_go, {"continue": "tool", "end": "responder"})

workflow.add_edge("tool", "agent")

workflow.set_entry_point("agent")

app = workflow.compile()

input = {"messages": [
    SystemMessage(content="Your task is to provide only the city name based on the user query. Nothing more, just the city name mentioned. If there is no city mentioned, then skip the tools to get the weather."),
    HumanMessage(content="What is the temperature in Las Vegas?"),
]}

for output in app.stream(input):
    # stream() yields dictionaries with output keyed by node name
    for key, value in output.items():
        print(f"Output from node: '{key}':")
        print("---")
        print(value)
    print("\n---\n")
