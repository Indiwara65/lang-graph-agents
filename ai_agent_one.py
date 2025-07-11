from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Union
import os

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(name="o3-mini-2025-01-31",
                 api_key=openai_api_key)

class AgentState(TypedDict):
    messages : List[Union[HumanMessage, AIMessage]]

def process(state:AgentState)->AgentState:
    """"This node will call the LLM"""
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    print(f"AI : {response.content}\n")
    return state

graph = StateGraph(state_schema=AgentState)
graph.add_node("process", process)
graph.add_edge(START, 'process')
graph.add_edge('process',END)

app = graph.compile()

user_input = input('Enter : ')
while True:
    app.invoke({'messages':[HumanMessage(content=user_input)]})
    user_input = input('Enter : ')