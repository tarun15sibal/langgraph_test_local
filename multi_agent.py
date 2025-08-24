from langgraph.prebuilt import create_react_agent
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from pydantic.v1 import BaseModel, Field
from typing import Optional
import requests
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import HumanInputRun

#memory = SqliteSaver.from_conn_string(":memory:")

class MultiAgentState(TypedDict):
    question: str
    question_type: str
    answer: str
    feedback: str

question_category_prompt = '''You are a senior specialist of analytical support. Your task is to classify the incoming questions. 
Depending on your answer, question will be routed to the right team, so your task is crucial for our team. 
There are 3 possible question types: 
- DATABASE - questions related to our database (tables or fields)
- LANGCHAIN- questions related to LangGraph or LangChain libraries
- GENERAL - general questions
Return in the output only one word (DATABASE, LANGCHAIN or  GENERAL).
'''

def router_node(state: MultiAgentState):
  messages = [
    SystemMessage(content=question_category_prompt), 
    HumanMessage(content=state['question'])
  ]
  llm = ChatOpenAI(
    model="qwen2.5",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key="null",  # if you prefer to pass api key in directly instaed of using env vars
    base_url="http://127.0.0.1:8080",
)
  model = create_react_agent(llm, tools = [])
  response = model.invoke({"messages": messages})
#  print(response["messages"])
  return {"question_type": response["messages"][-1].content}



###Tvly starts here

os.environ["TAVILY_API_KEY"] = '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'
tavily_tool = TavilySearchResults(max_results=5)

search_expert_system_prompt = '''
You are an expert in LangChain and other technologies. 
Your goal is to answer questions based on results provided by search.
You don't add anything yourself and provide only information baked by other sources. 
'''

def search_expert_node(state: MultiAgentState):
    llm = ChatOpenAI(
    model="qwen2.5",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key="null",  # if you prefer to pass api key in directly instaed of using env vars
    base_url="http://127.0.0.1:8080",
)
    tvly_agent = create_react_agent(llm, [tavily_tool],prompt = search_expert_system_prompt)
    messages = [HumanMessage(content=state['question'])]
    result = tvly_agent.invoke({"messages": messages})
    return {'answer': result['messages'][-1].content}

###tvly ends here


### LLm general question starts here

general_prompt = '''You're a friendly assistant and your goal is to answer general questions.
Please, don't provide any unchecked information and just tell that you don't know if you don't have enough info.
'''
def general_assistant_node(state: MultiAgentState):
    messages = [
        SystemMessage(content=general_prompt), 
        HumanMessage(content=state['question'])
    ]
    llm = ChatOpenAI(
    model="qwen2.5",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key="null",  # if you prefer to pass api key in directly instaed of using env vars
    base_url="http://127.0.0.1:8080",
)
    gen_agent = create_react_agent(llm, [],prompt = general_prompt)
    response = gen_agent.invoke({"messages": messages})
    return {"answer": response['messages'][-1].content}

###LLM general question ends here


### simple router start
def route_question(state: MultiAgentState):
    return state['question_type']
### simple router end


### add human input


human_tool = HumanInputRun()

editor_agent_prompt = '''You're an editor and your goal is to provide the final answer to the customer, taking into the initial question.
If you need any clarifications or need feedback, please, use human. Always reach out to human to get the feedback before final answer.
You don't add any information on your own. You use friendly and professional tone. 
In the output please provide the final answer to the customer without additional comments.
Here's all the information you need.

Question from customer: 
----
{question}
----
Draft answer:
----
{answer}
----
'''

def editor_agent_node(state: MultiAgentState):
  llm = ChatOpenAI(
    model="qwen2.5",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key="null",  # if you prefer to pass api key in directly instaed of using env vars
    base_url="http://127.0.0.1:8080",
)
  editor_agent = create_react_agent(llm, [human_tool])
  messages = [SystemMessage(content=editor_agent_prompt.format(question = state['question'], answer = state['answer']))]
  result = editor_agent.invoke({"messages": messages})
  return {'answer': result['messages'][-1].content}






builder = StateGraph(MultiAgentState)
builder.add_node("router", router_node)
builder.add_node('langchain_expert', search_expert_node)
builder.add_node('general_assistant', general_assistant_node)
builder.add_node('editor', editor_agent_node)
builder.add_conditional_edges(
    "router", 
    route_question,
    { 
     'LANGCHAIN': 'langchain_expert', 
     'GENERAL': 'general_assistant'}
)
builder.set_entry_point("router")
#builder.add_edge('database_expert', END)
builder.add_edge('langchain_expert', 'editor')
builder.add_edge('general_assistant', 'editor')
builder.add_edge('editor', END)





with SqliteSaver.from_conn_string(":memory:") as memory:
     graph = builder.compile(checkpointer=memory)
     config = {"configurable": {"thread_id": "thread-21"}}
     s = graph.invoke({"question": "How many hours are in a year"}, config)
     print(s["answer"])
