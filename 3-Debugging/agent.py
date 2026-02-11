from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import END, START
from langgraph.graph.state import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import tools_condition
import os
from dotenv import load_dotenv
load_dotenv()


os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"]= os.getenv('LANGSMITH_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"]="true"


os.environ["LANGCHAIN_PROJECT"]="LANGCHAIN_PROJECT"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"


from langchain.chat_models import init_chat_model
llm=init_chat_model("groq:llama-3.3-70b-versatile")

class State(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]


## Graph With tool Call
def make_tool_graph():
    @tool
    def add(a:float,b:float):
        """Add two number"""
        return a+b
    tools=[add]
    tool_node=ToolNode([add])

    llm_with_tool=llm.bind_tools([add])

    def call_llm_model(state:State):
        return {"messages":[llm_with_tool.invoke(state['messages'])]}
    

    ## Grpah
    builder=StateGraph(State)
    builder.add_node("tool_calling_llm",call_llm_model)
    builder.add_node("tools",ToolNode(tools))

    ## Add Edges
    builder.add_edge(START, "tool_calling_llm")
    builder.add_conditional_edges(
        "tool_calling_llm",
        # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
        # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
        tools_condition
    )
    builder.add_edge("tools","tool_calling_llm")

    ## compile the graph
    graph=builder.compile()
    return graph

tool_agent=make_tool_graph()