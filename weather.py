import os
from dotenv import load_dotenv
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_ollama import OllamaLLM
from langgraph.graph import Graph

load_dotenv()
os.environ["OPENWEATHERMAP_API_KEY"] = os.environ.get("OPENWEATHERMAP_API_KEY")

model = OllamaLLM(model="llama3.2:3b-instruct-q8_0")

def function_1(state):
    messages = state['messages']
    user_input = messages[-1]
    complete_query = "Your task is to provide only the city name based on the user query. \
                    Nothing more, just the city name mentioned. Following is the user query: " + user_input
    response = model.invoke(complete_query)
    state['messages'].append(response) # appending AIMessage response to the AgentState
    return state

def function_2(state):
    messages = state['messages']
    agent_response = messages[-1]
    weather = OpenWeatherMapAPIWrapper()
    weather_data = weather.run(agent_response)
    state['messages'].append(weather_data)
    return state

def function_3(state):
    messages = state['messages']
    user_input = messages[0]
    available_info = messages[-1]
    agent2_query = "Your task is to provide info concisely based on the user query and the available information from the internet. \
        Following is the user query: " + user_input + "Available information: " + available_info
    response = model.invoke(agent2_query)
    return response

# Define a Langchain graph
workflow = Graph()

workflow.add_node("agent", function_1)
workflow.add_node("tool", function_2)
workflow.add_node("responder", function_3)

workflow.add_edge("agent", "tool")
workflow.add_edge("tool", "responder")

workflow.set_entry_point("agent")
workflow.set_finish_point("responder")

app = workflow.compile()

# input = {"messages": ["What is the temperature in Las Vegas?"]}
input = {"messages": ["What's the temperature in Las Vegas?"]}

for output in app.stream(input):
    # stream() yields dictionaries with output keyed by node name
    for key, value in output.items():
        print(f"Output from node: '{key}':")
        print("---")
        print(value)
    print("\n---\n")
