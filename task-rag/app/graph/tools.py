"""LangGraph tools for the task management system."""

import json
from typing import List, Optional
from uuid import UUID

from langchain_core.tools import tool
from langchain_core.vectorstores import VectorStoreRetriever

from ..deps import get_retriever, get_settings
from ..models.task import Task, TaskStatus
from ..services.task_service import TaskService


# Global instances - these will be injected properly in the graph
_retriever: Optional[VectorStoreRetriever] = None
_task_service: Optional[TaskService] = None


def set_tool_dependencies(retriever: VectorStoreRetriever, task_service: TaskService) -> None:
    """Set dependencies for tools (called during app initialization)."""
    global _retriever, _task_service
    _retriever = retriever
    _task_service = task_service


@tool
def retriever_tool(query: str) -> str:
    """Search the RAG corpus and return relevant passages with source/page references.
    
    Args:
        query: The search query to find relevant documents
        
    Returns:
        A formatted string containing relevant passages with source information
    """
    if not _retriever:
        return "Error: Retriever not initialized. Please ensure documents have been ingested."
    
    try:
        # Retrieve relevant documents
        docs = _retriever.invoke(query)
        
        if not docs:
            return "No relevant documents found for your query."
        
        # Format the results with source information
        results = []
        for i, doc in enumerate(docs, 1):
            content = doc.page_content.strip()
            metadata = doc.metadata
            
            # Extract source information
            source = metadata.get('source', 'Unknown')
            page = metadata.get('page', 'Unknown')
            
            result = f"**Passage {i}:**\n{content}\n*Source: {source}, Page: {page}*"
            results.append(result)
        
        return "\n\n".join(results)
    
    except Exception as e:
        return f"Error retrieving documents: {str(e)}"


@tool
def task_create_tool(text: str) -> str:
    """Create a new task from user text input.
    
    Args:
        text: The text describing the task to create
        
    Returns:
        A confirmation message with the created task details
    """
    if not _task_service:
        return "Error: Task service not initialized."
    
    try:
        # Parse the text to extract title and description
        # For now, we'll use the first sentence as title and rest as description
        sentences = text.strip().split('.')
        title = sentences[0].strip()
        description = '. '.join(sentences[1:]).strip() if len(sentences) > 1 else None
        
        # Ensure title is not empty
        if not title:
            return "Error: Cannot create task with empty title."
        
        # Create the task
        task = _task_service.create_task(title=title, description=description)
        
        return f"✅ Task created successfully!\n**ID:** {task.id}\n**Title:** {task.title}\n**Description:** {task.description or 'None'}\n**Status:** {task.status}"
    
    except Exception as e:
        return f"Error creating task: {str(e)}"


@tool
def task_update_tool(task_id: str, status: str) -> str:
    """Update the status of an existing task.
    
    Args:
        task_id: The UUID of the task to update
        status: The new status (pending, in_progress, completed, cancelled)
        
    Returns:
        A confirmation message with the updated task details
    """
    if not _task_service:
        return "Error: Task service not initialized."
    
    try:
        # Validate task_id format
        try:
            uuid_obj = UUID(task_id)
        except ValueError:
            return f"Error: Invalid task ID format: {task_id}"
        
        # Validate status
        try:
            task_status = TaskStatus(status.lower())
        except ValueError:
            valid_statuses = [s.value for s in TaskStatus]
            return f"Error: Invalid status '{status}'. Valid statuses are: {', '.join(valid_statuses)}"
        
        # Update the task
        task = _task_service.update_task_status(uuid_obj, task_status)
        
        if not task:
            return f"Error: Task with ID {task_id} not found."
        
        return f"✅ Task updated successfully!\n**ID:** {task.id}\n**Title:** {task.title}\n**Status:** {task.status}\n**Updated:** {task.updated_at.isoformat()}"
    
    except Exception as e:
        return f"Error updating task: {str(e)}"


@tool
def task_list_tool(status: Optional[str] = None) -> str:
    """List tasks, optionally filtered by status.
    
    Args:
        status: Optional status filter (pending, in_progress, completed, cancelled)
        
    Returns:
        A formatted list of tasks
    """
    if not _task_service:
        return "Error: Task service not initialized."
    
    try:
        # Validate status if provided
        status_filter = None
        if status:
            try:
                status_filter = TaskStatus(status.lower())
            except ValueError:
                valid_statuses = [s.value for s in TaskStatus]
                return f"Error: Invalid status '{status}'. Valid statuses are: {', '.join(valid_statuses)}"
        
        # Get tasks
        tasks = _task_service.list_tasks(status=status_filter)
        
        if not tasks:
            filter_msg = f" with status '{status}'" if status else ""
            return f"No tasks found{filter_msg}."
        
        # Format the task list
        task_lines = []
        for task in tasks:
            task_line = f"• **{task.title}** (ID: {str(task.id)[:8]}...) - Status: {task.status}"
            if task.description:
                task_line += f"\n  Description: {task.description[:100]}{'...' if len(task.description) > 100 else ''}"
            task_lines.append(task_line)
        
        header = f"📋 **Tasks{' (' + status + ')' if status else ''}** ({len(tasks)} total):\n\n"
        return header + "\n\n".join(task_lines)
    
    except Exception as e:
        return f"Error listing tasks: {str(e)}"


# List of all available tools
TOOLS = [
    retriever_tool,
    task_create_tool,
    task_update_tool,
    task_list_tool,
]

