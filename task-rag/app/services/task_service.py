"""Task service for CRUD operations and task management."""

import logging
from datetime import datetime, date
from threading import Lock
from typing import Dict, List, Optional, UUID
from uuid import uuid4

from ..models.task import Task, TaskStatus
from ..schemas import TaskCreate, TaskUpdate

logger = logging.getLogger(__name__)


class TaskService:
    """Service for task CRUD operations with in-memory storage."""
    
    def __init__(self):
        """Initialize the task service."""
        self._tasks: Dict[UUID, Task] = {}
        self._lock = Lock()  # Thread-safe operations
        logger.info("Task service initialized with in-memory storage")
    
    def create_task(self, title: str, description: Optional[str] = None) -> Task:
        """Create a new task.
        
        Args:
            title: Task title
            description: Optional task description
            
        Returns:
            Created task
            
        Raises:
            ValueError: If title is empty or invalid
        """
        if not title or not title.strip():
            raise ValueError("Task title cannot be empty")
        
        with self._lock:
            task = Task(
                title=title.strip(),
                description=description.strip() if description else None
            )
            
            self._tasks[task.id] = task
            
            logger.info(f"Created task {task.id}: {task.title}")
            return task
    
    def create_task_from_schema(self, task_data: TaskCreate) -> Task:
        """Create a new task from schema.
        
        Args:
            task_data: Task creation data
            
        Returns:
            Created task
        """
        return self.create_task(
            title=task_data.title,
            description=task_data.description
        )
    
    def get_task(self, task_id: UUID) -> Optional[Task]:
        """Get a task by ID.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task if found, None otherwise
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                logger.debug(f"Retrieved task {task_id}: {task.title}")
            else:
                logger.debug(f"Task {task_id} not found")
            return task
    
    def update_task(self, task_id: UUID, task_data: TaskUpdate) -> Optional[Task]:
        """Update a task.
        
        Args:
            task_id: Task ID
            task_data: Task update data
            
        Returns:
            Updated task if found, None otherwise
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                logger.warning(f"Task {task_id} not found for update")
                return None
            
            # Update fields if provided
            if task_data.title is not None:
                if not task_data.title.strip():
                    raise ValueError("Task title cannot be empty")
                task.title = task_data.title.strip()
            
            if task_data.description is not None:
                task.description = task_data.description.strip() if task_data.description else None
            
            if task_data.status is not None:
                task.status = task_data.status
            
            # Update timestamp
            task.update_timestamp()
            
            logger.info(f"Updated task {task_id}: {task.title}")
            return task
    
    def update_task_status(self, task_id: UUID, status: TaskStatus) -> Optional[Task]:
        """Update task status.
        
        Args:
            task_id: Task ID
            status: New status
            
        Returns:
            Updated task if found, None otherwise
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                logger.warning(f"Task {task_id} not found for status update")
                return None
            
            old_status = task.status
            task.status = status
            task.update_timestamp()
            
            logger.info(f"Updated task {task_id} status: {old_status} -> {status}")
            return task
    
    def delete_task(self, task_id: UUID) -> bool:
        """Delete a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            True if task was deleted, False if not found
        """
        with self._lock:
            task = self._tasks.pop(task_id, None)
            if task:
                logger.info(f"Deleted task {task_id}: {task.title}")
                return True
            else:
                logger.warning(f"Task {task_id} not found for deletion")
                return False
    
    def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        date_filter: Optional[date] = None,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Task]:
        """List tasks with optional filters.
        
        Args:
            status: Filter by status
            date_filter: Filter by creation date
            limit: Maximum number of tasks to return
            offset: Number of tasks to skip
            
        Returns:
            List of tasks matching the filters
        """
        with self._lock:
            tasks = list(self._tasks.values())
            
            # Apply status filter
            if status is not None:
                tasks = [task for task in tasks if task.status == status]
            
            # Apply date filter
            if date_filter is not None:
                tasks = [
                    task for task in tasks
                    if task.created_at.date() == date_filter
                ]
            
            # Sort by creation date (newest first)
            tasks.sort(key=lambda t: t.created_at, reverse=True)
            
            # Apply pagination
            if offset > 0:
                tasks = tasks[offset:]
            
            if limit is not None:
                tasks = tasks[:limit]
            
            logger.debug(f"Listed {len(tasks)} tasks (status={status}, date={date_filter})")
            return tasks
    
    def get_task_count(self, status: Optional[TaskStatus] = None) -> int:
        """Get count of tasks.
        
        Args:
            status: Filter by status
            
        Returns:
            Number of tasks matching the filter
        """
        with self._lock:
            if status is None:
                return len(self._tasks)
            
            return sum(1 for task in self._tasks.values() if task.status == status)
    
    def get_tasks_by_status(self) -> Dict[TaskStatus, int]:
        """Get task counts by status.
        
        Returns:
            Dictionary mapping status to count
        """
        with self._lock:
            counts = {status: 0 for status in TaskStatus}
            
            for task in self._tasks.values():
                counts[task.status] += 1
            
            return counts
    
    def search_tasks(self, query: str, limit: Optional[int] = None) -> List[Task]:
        """Search tasks by title and description.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching tasks
        """
        if not query or not query.strip():
            return []
        
        query_lower = query.strip().lower()
        
        with self._lock:
            matching_tasks = []
            
            for task in self._tasks.values():
                # Search in title
                if query_lower in task.title.lower():
                    matching_tasks.append(task)
                    continue
                
                # Search in description
                if task.description and query_lower in task.description.lower():
                    matching_tasks.append(task)
            
            # Sort by relevance (title matches first, then by creation date)
            def sort_key(task):
                title_match = query_lower in task.title.lower()
                return (not title_match, -task.created_at.timestamp())
            
            matching_tasks.sort(key=sort_key)
            
            if limit is not None:
                matching_tasks = matching_tasks[:limit]
            
            logger.debug(f"Found {len(matching_tasks)} tasks matching query: {query}")
            return matching_tasks
    
    def bulk_create_tasks(self, tasks_data: List[TaskCreate]) -> List[Task]:
        """Create multiple tasks in bulk.
        
        Args:
            tasks_data: List of task creation data
            
        Returns:
            List of created tasks
            
        Raises:
            ValueError: If any task data is invalid
        """
        if not tasks_data:
            return []
        
        created_tasks = []
        
        with self._lock:
            for task_data in tasks_data:
                try:
                    task = Task(
                        title=task_data.title.strip(),
                        description=task_data.description.strip() if task_data.description else None
                    )
                    
                    self._tasks[task.id] = task
                    created_tasks.append(task)
                    
                except Exception as e:
                    logger.error(f"Error creating task '{task_data.title}': {str(e)}")
                    # Continue with other tasks
                    continue
            
            logger.info(f"Bulk created {len(created_tasks)} tasks")
            return created_tasks
    
    def clear_all_tasks(self) -> int:
        """Clear all tasks (for testing/development).
        
        Returns:
            Number of tasks that were cleared
        """
        with self._lock:
            count = len(self._tasks)
            self._tasks.clear()
            logger.warning(f"Cleared all {count} tasks")
            return count
    
    def get_statistics(self) -> Dict[str, any]:
        """Get task statistics.
        
        Returns:
            Dictionary containing various statistics
        """
        with self._lock:
            total_tasks = len(self._tasks)
            status_counts = self.get_tasks_by_status()
            
            # Calculate completion rate
            completed_count = status_counts[TaskStatus.COMPLETED]
            completion_rate = (completed_count / total_tasks * 100) if total_tasks > 0 else 0
            
            # Find oldest and newest tasks
            oldest_task = None
            newest_task = None
            
            if self._tasks:
                tasks_by_date = sorted(self._tasks.values(), key=lambda t: t.created_at)
                oldest_task = tasks_by_date[0]
                newest_task = tasks_by_date[-1]
            
            return {
                'total_tasks': total_tasks,
                'status_counts': status_counts,
                'completion_rate': round(completion_rate, 2),
                'oldest_task': {
                    'id': str(oldest_task.id),
                    'title': oldest_task.title,
                    'created_at': oldest_task.created_at.isoformat()
                } if oldest_task else None,
                'newest_task': {
                    'id': str(newest_task.id),
                    'title': newest_task.title,
                    'created_at': newest_task.created_at.isoformat()
                } if newest_task else None
            }


# Global task service instance - will be initialized during app startup
_task_service: Optional[TaskService] = None


def get_task_service() -> Optional[TaskService]:
    """Get the global task service instance.
    
    Returns:
        Task service instance or None if not initialized
    """
    return _task_service


def initialize_task_service() -> TaskService:
    """Initialize the global task service instance.
    
    Returns:
        Initialized task service
    """
    global _task_service
    _task_service = TaskService()
    return _task_service

