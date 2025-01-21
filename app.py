from langchain_ollama import OllamaLLM
from langgraph.graph import Graph

model = OllamaLLM(model="llama3.2:3b-instruct-q8_0")

def function_1(input_1):
    response = model.invoke(input_1)
    return response

def function_2(input_2):
    return "Agent Says: " + input_2

# Define a Langchain graph
workflow = Graph()

workflow.add_node("agent", function_1)
workflow.add_node("node_2", function_2)

workflow.add_edge("agent", "node_2")

workflow.set_entry_point("agent")
workflow.set_finish_point("node_2")

app = workflow.compile()

input = "Hello"

for output in app.stream(input):
    # stream() yields dictionaries with output keyed by node name
    for key, value in output.items():
        print(f"Output from node: '{key}':")
        print("---")
        print(value)
    print("\n---\n")
