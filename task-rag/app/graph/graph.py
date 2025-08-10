"""LangGraph graph orchestration for the task management system."""

from typing import Literal

from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage

from .state import AgentState
from .nodes import call_llm, take_action


def should_continue(state: AgentState) -> Literal["retriever_agent", "__end__"]:
    """Determine whether to continue with tool execution or end the conversation.

    Args:
        state: The current agent state

    Returns:
        "retriever_agent" if there are tool calls to execute, "__end__" otherwise
    """
    messages = state["messages"]

    # Check if the last message is an AI message with tool calls
    if messages and isinstance(messages[-1], AIMessage):
        last_message = messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "retriever_agent"

    return "__end__"


def build_graph() -> StateGraph:
    """Build and compile the LangGraph state graph.

    Returns:
        Compiled state graph ready for execution
    """
    # Create the state graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("llm", call_llm)
    workflow.add_node("retriever_agent", take_action)

    # Set entry point
    workflow.set_entry_point("llm")

    # Add conditional edges
    workflow.add_conditional_edges(
        "llm",
        should_continue,
        {
            "retriever_agent": "retriever_agent",
            "__end__": END,
        },
    )

    # Add edge from retriever_agent back to llm
    workflow.add_edge("retriever_agent", "llm")

    # Compile the graph
    return workflow.compile()


# Global graph instance - will be initialized during app startup
_compiled_graph = None


def get_graph() -> StateGraph:
    """Get the compiled graph instance.

    Returns:
        The compiled state graph
    """
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def initialize_graph() -> None:
    """Initialize the graph during app startup."""
    global _compiled_graph
    _compiled_graph = build_graph()
