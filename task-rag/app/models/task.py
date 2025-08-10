"""Domain models for the task management system."""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Task status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class Task(BaseModel):
    """Task domain model."""
    
    id: UUID = Field(default_factory=uuid4, description="Unique task identifier")
    title: str = Field(..., min_length=1, max_length=200, description="Task title")
    description: Optional[str] = Field(None, max_length=2000, description="Task description")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Task status")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Task creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Task last update timestamp")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }
    
    def update_timestamp(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.utcnow()
    
    def mark_completed(self) -> None:
        """Mark task as completed and update timestamp."""
        self.status = TaskStatus.COMPLETED
        self.update_timestamp()
    
    def mark_in_progress(self) -> None:
        """Mark task as in progress and update timestamp."""
        self.status = TaskStatus.IN_PROGRESS
        self.update_timestamp()
    
    def mark_cancelled(self) -> None:
        """Mark task as cancelled and update timestamp."""
        self.status = TaskStatus.CANCELLED
        self.update_timestamp()

