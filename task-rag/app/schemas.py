"""API request/response schemas for the task management system."""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from .models.task import TaskStatus


# Task-related schemas
class TaskCreate(BaseModel):
    """Schema for creating a new task."""
    title: str = Field(..., min_length=1, max_length=200, description="Task title")
    description: Optional[str] = Field(None, max_length=2000, description="Task description")


class TaskUpdate(BaseModel):
    """Schema for updating an existing task."""
    title: Optional[str] = Field(None, min_length=1, max_length=200, description="Task title")
    description: Optional[str] = Field(None, max_length=2000, description="Task description")
    status: Optional[TaskStatus] = Field(None, description="Task status")


class TaskResponse(BaseModel):
    """Schema for task API responses."""
    id: UUID = Field(..., description="Unique task identifier")
    title: str = Field(..., description="Task title")
    description: Optional[str] = Field(None, description="Task description")
    status: TaskStatus = Field(..., description="Task status")
    created_at: datetime = Field(..., description="Task creation timestamp")
    updated_at: datetime = Field(..., description="Task last update timestamp")
    
    class Config:
        """Pydantic configuration."""
        from_attributes = True


class TaskListResponse(BaseModel):
    """Schema for task list API responses."""
    tasks: List[TaskResponse] = Field(..., description="List of tasks")
    total: int = Field(..., description="Total number of tasks")


# Chat-related schemas
class UserMessage(BaseModel):
    """Schema for user chat messages."""
    message: str = Field(..., min_length=1, max_length=4000, description="User message content")
    session_id: Optional[str] = Field(None, description="Chat session identifier")


class ChatResponse(BaseModel):
    """Schema for chat API responses."""
    message_id: str = Field(..., description="Unique message identifier")
    session_id: str = Field(..., description="Chat session identifier")
    status: str = Field(default="processing", description="Message processing status")


# Ingestion-related schemas
class IngestResponse(BaseModel):
    """Schema for document ingestion responses."""
    success: bool = Field(..., description="Whether ingestion was successful")
    message: str = Field(..., description="Status message")
    document_count: Optional[int] = Field(None, description="Number of documents processed")
    chunk_count: Optional[int] = Field(None, description="Number of text chunks created")
    filename: Optional[str] = Field(None, description="Original filename")


class BulkTaskCreate(BaseModel):
    """Schema for bulk task creation."""
    tasks: List[TaskCreate] = Field(..., min_items=1, max_items=100, description="List of tasks to create")


class BulkTaskResponse(BaseModel):
    """Schema for bulk task creation responses."""
    success: bool = Field(..., description="Whether bulk creation was successful")
    created_count: int = Field(..., description="Number of tasks successfully created")
    failed_count: int = Field(default=0, description="Number of tasks that failed to create")
    tasks: List[TaskResponse] = Field(..., description="List of created tasks")


# WebSocket-related schemas
class WebSocketMessage(BaseModel):
    """Schema for WebSocket messages."""
    type: str = Field(..., description="Message type (token, event, error, complete)")
    content: str = Field(..., description="Message content")
    session_id: str = Field(..., description="Session identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")


# Health check schema
class HealthResponse(BaseModel):
    """Schema for health check responses."""
    status: str = Field(default="healthy", description="Service health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")
    version: str = Field(default="1.0.0", description="Application version")

