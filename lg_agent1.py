from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

llm = ChatOpenAI(
    model="qwen2.5",
    temperature=1.1,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key="null",  # if you prefer to pass api key in directly instaed of using env vars
    base_url="http://127.0.0.1:8080",
)


graph_builder = StateGraph(State)

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")

graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

input_state = {"messages": [{"role": "user", "content": "Hello!"}]}

output = graph.invoke(input_state)

print(output)
