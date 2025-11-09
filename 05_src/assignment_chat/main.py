from langgraph.graph import StateGraph, MessagesState, START, END
from langchain.chat_models import init_chat_model
from langgraph.prebuilt.tool_node import ToolNode, tools_condition
from langchain_core.messages import SystemMessage,  ToolMessage, HumanMessage, AnyMessage
from typing_extensions import TypedDict, Annotated
from typing import Literal
from instructions import system_prompt
import operator
from dotenv import load_dotenv
from tools import get_coords, search_web, get_weather, get_book_reco
from logger import get_logger

# Start logging and also load the env
_logs = get_logger(__name__)
load_dotenv("../.env")
load_dotenv("../.secrets")

# Specify the chat model
chat_agent = init_chat_model(
    "openai:gpt-4o-mini"
    
    ,
)
# Define tools & bind the tools to the chat_agent
tools = [get_coords, search_web, get_weather, get_book_reco]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = chat_agent.bind_tools(tools)

# Get the prompt(s)
instructions = system_prompt()

# Define state
class MessagesState(TypedDict): 
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

# Call the LLM, and make a decision as to whether to rely on our tools or not.
def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""
    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content = instructions
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }

# This is where we actually call the tools
def tool_node(state: dict):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END

# Build workflow
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

# Compile the agent
agent = agent_builder.compile()

# This is for testing
# messages = [HumanMessage(content="I'm going to Warsaw, what should I do there?")]
# messages = agent.invoke({"messages": messages})
# for m in messages["messages"]:
#     m.pretty_print()

def get_graph():
    builder = StateGraph(MessagesState)
    builder.add_node(llm_call)
    builder.add_node(ToolNode(tools))
    builder.add_edge(START, "llm_call")
    builder.add_conditional_edges(
        "llm_call",
        tools_condition,
    )
    builder.add_edge("tools", "llm_call")
    graph = builder.compile()
    return graph
