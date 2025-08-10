"""LangGraph state management for the task management system."""

from typing import Annotated, Sequence
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State for the LangGraph agent."""

    # Messages are the core of the conversation state
    # The add_messages function handles message deduplication and ordering
    messages: Annotated[Sequence[BaseMessage], add_messages]
