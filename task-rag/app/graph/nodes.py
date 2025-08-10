"""LangGraph nodes for the task management system."""

from typing import Dict, Any, Optional, Callable
import asyncio
import json

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler

from .state import AgentState
from .tools import TOOLS


# Global WebSocket broadcaster - will be set during app initialization
_websocket_broadcaster: Optional[Callable[[str, str], None]] = None


def set_websocket_broadcaster(broadcaster: Callable[[str, str], None]) -> None:
    """Set the WebSocket broadcaster for token streaming."""
    global _websocket_broadcaster
    _websocket_broadcaster = broadcaster


class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming tokens to WebSocket."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
    
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Called when a new token is generated."""
        if _websocket_broadcaster:
            _websocket_broadcaster(self.session_id, token)


def call_llm(state: AgentState) -> Dict[str, Any]:
    """Call the LLM with the current state and system prompt.
    
    Args:
        state: The current agent state containing messages
        
    Returns:
        Updated state with the LLM response
    """
    # System prompt for the task manager
    system_prompt = """You are a helpful AI task manager assistant. You can help users with:

1. **Document Search**: Search through uploaded documents to find relevant information
2. **Task Management**: Create, update, and list tasks based on user requests
3. **Natural Language Processing**: Understand user intent and take appropriate actions

Available tools:
- retriever_tool: Search uploaded documents for relevant information
- task_create_tool: Create new tasks from user descriptions
- task_update_tool: Update task status (pending, in_progress, completed, cancelled)
- task_list_tool: List tasks, optionally filtered by status

When users ask about documents or need information from uploaded files, use the retriever_tool.
When users want to create, update, or manage tasks, use the appropriate task tools.
Always be helpful, clear, and provide actionable responses.

If you need to use tools, call them with the appropriate parameters. The tools will return formatted results that you can use in your response to the user."""

    # Get the current messages
    messages = list(state["messages"])
    
    # Add system message if not already present
    if not messages or not isinstance(messages[0], SystemMessage):
        messages.insert(0, SystemMessage(content=system_prompt))
    
    # Extract session_id from the last human message metadata if available
    session_id = "default"
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and hasattr(msg, 'additional_kwargs'):
            session_id = msg.additional_kwargs.get('session_id', 'default')
            break
    
    # Create LLM instance with tools
    # Note: In a real implementation, this would be injected via dependencies
    llm = ChatOpenAI(
        model="gpt-4-turbo-preview",
        temperature=0.7,
        streaming=True,
        callbacks=[StreamingCallbackHandler(session_id)] if _websocket_broadcaster else []
    )
    
    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(TOOLS)
    
    # Invoke the LLM
    response = llm_with_tools.invoke(messages)
    
    # Return updated state
    return {"messages": [response]}


def take_action(state: AgentState) -> Dict[str, Any]:
    """Execute tool calls and return tool messages.
    
    Args:
        state: The current agent state containing messages
        
    Returns:
        Updated state with tool execution results
    """
    # Get the last message (should be an AI message with tool calls)
    messages = list(state["messages"])
    last_message = messages[-1]
    
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        # No tool calls to execute
        return {"messages": []}
    
    # Execute each tool call
    tool_messages = []
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]
        
        # Find the tool function
        tool_func = None
        for tool in TOOLS:
            if tool.name == tool_name:
                tool_func = tool
                break
        
        if tool_func is None:
            # Tool not found
            result = f"Error: Tool '{tool_name}' not found"
        else:
            try:
                # Execute the tool
                result = tool_func.invoke(tool_args)
            except Exception as e:
                result = f"Error executing {tool_name}: {str(e)}"
        
        # Create tool message
        tool_message = ToolMessage(
            content=result,
            tool_call_id=tool_call_id
        )
        tool_messages.append(tool_message)
    
    return {"messages": tool_messages}

