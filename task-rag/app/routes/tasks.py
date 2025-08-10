"""Task management CRUD routes."""

import logging
from datetime import date
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ..deps import get_task_service
from ..models.task import TaskStatus
from ..schemas import TaskCreate, TaskResponse, TaskUpdate
from ..services.task_service import TaskService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.post("/", response_model=TaskResponse, status_code=status.HTTP_201_CREATED)
async def create_task(
    task_data: TaskCreate,
    task_service: TaskService = Depends(get_task_service)
) -> TaskResponse:
    """Create a new task.
    
    Args:
        task_data: Task creation data
        task_service: Task service instance
        
    Returns:
        Created task response
        
    Raises:
        HTTPException: If task creation fails
    """
    try:
        logger.info(f"Creating new task: {task_data.title}")
        
        task = task_service.create_task_from_schema(task_data)
        
        return TaskResponse(
            id=task.id,
            title=task.title,
            description=task.description,
            status=task.status,
            created_at=task.created_at,
            updated_at=task.updated_at
        )
    
    except ValueError as e:
        logger.error(f"Validation error creating task: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error creating task: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during task creation"
        )


@router.get("/", response_model=List[TaskResponse])
async def list_tasks(
    status_filter: Optional[TaskStatus] = Query(None, alias="status"),
    date_filter: Optional[date] = Query(None, alias="date"),
    limit: Optional[int] = Query(None, ge=1, le=100),
    offset: int = Query(0, ge=0),
    task_service: TaskService = Depends(get_task_service)
) -> List[TaskResponse]:
    """List tasks with optional filters.
    
    Args:
        status_filter: Filter by task status
        date_filter: Filter by creation date
        limit: Maximum number of tasks to return
        offset: Number of tasks to skip
        task_service: Task service instance
        
    Returns:
        List of task responses
    """
    try:
        logger.debug(f"Listing tasks with filters: status={status_filter}, date={date_filter}")
        
        tasks = task_service.list_tasks(
            status=status_filter,
            date_filter=date_filter,
            limit=limit,
            offset=offset
        )
        
        return [
            TaskResponse(
                id=task.id,
                title=task.title,
                description=task.description,
                status=task.status,
                created_at=task.created_at,
                updated_at=task.updated_at
            )
            for task in tasks
        ]
    
    except Exception as e:
        logger.error(f"Error listing tasks: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while listing tasks"
        )


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: UUID,
    task_service: TaskService = Depends(get_task_service)
) -> TaskResponse:
    """Get a specific task by ID.
    
    Args:
        task_id: Task ID
        task_service: Task service instance
        
    Returns:
        Task response
        
    Raises:
        HTTPException: If task not found
    """
    try:
        logger.debug(f"Getting task: {task_id}")
        
        task = task_service.get_task(task_id)
        
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )
        
        return TaskResponse(
            id=task.id,
            title=task.title,
            description=task.description,
            status=task.status,
            created_at=task.created_at,
            updated_at=task.updated_at
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task {task_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while retrieving task"
        )


@router.patch("/{task_id}", response_model=TaskResponse)
async def update_task(
    task_id: UUID,
    task_data: TaskUpdate,
    task_service: TaskService = Depends(get_task_service)
) -> TaskResponse:
    """Update a task.
    
    Args:
        task_id: Task ID
        task_data: Task update data
        task_service: Task service instance
        
    Returns:
        Updated task response
        
    Raises:
        HTTPException: If task not found or update fails
    """
    try:
        logger.info(f"Updating task: {task_id}")
        
        task = task_service.update_task(task_id, task_data)
        
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )
        
        return TaskResponse(
            id=task.id,
            title=task.title,
            description=task.description,
            status=task.status,
            created_at=task.created_at,
            updated_at=task.updated_at
        )
    
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error updating task {task_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error updating task {task_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during task update"
        )


@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_task(
    task_id: UUID,
    task_service: TaskService = Depends(get_task_service)
):
    """Delete a task.
    
    Args:
        task_id: Task ID
        task_service: Task service instance
        
    Raises:
        HTTPException: If task not found
    """
    try:
        logger.info(f"Deleting task: {task_id}")
        
        success = task_service.delete_task(task_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting task {task_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during task deletion"
        )


@router.get("/search/", response_model=List[TaskResponse])
async def search_tasks(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: Optional[int] = Query(10, ge=1, le=50),
    task_service: TaskService = Depends(get_task_service)
) -> List[TaskResponse]:
    """Search tasks by title and description.
    
    Args:
        q: Search query
        limit: Maximum number of results
        task_service: Task service instance
        
    Returns:
        List of matching task responses
    """
    try:
        logger.debug(f"Searching tasks with query: {q}")
        
        tasks = task_service.search_tasks(q, limit=limit)
        
        return [
            TaskResponse(
                id=task.id,
                title=task.title,
                description=task.description,
                status=task.status,
                created_at=task.created_at,
                updated_at=task.updated_at
            )
            for task in tasks
        ]
    
    except Exception as e:
        logger.error(f"Error searching tasks: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while searching tasks"
        )


@router.get("/stats/", response_model=dict)
async def get_task_statistics(
    task_service: TaskService = Depends(get_task_service)
) -> dict:
    """Get task statistics.
    
    Args:
        task_service: Task service instance
        
    Returns:
        Task statistics
    """
    try:
        logger.debug("Getting task statistics")
        
        stats = task_service.get_statistics()
        
        return stats
    
    except Exception as e:
        logger.error(f"Error getting task statistics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while retrieving statistics"
        )

