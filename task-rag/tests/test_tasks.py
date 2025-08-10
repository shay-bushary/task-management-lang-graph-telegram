"""Tests for task CRUD operations and LangGraph tool integration."""

import pytest
from datetime import datetime, date
from unittest.mock import patch, MagicMock, AsyncMock
from uuid import uuid4

from app.models.task import Task, TaskStatus
from app.services.task_service import TaskService
from app.schemas import TaskCreate, TaskUpdate
from app.graph.tools import task_create_tool, task_update_tool, task_list_tool


class TestTaskModel:
    """Test Task domain model."""
    
    def test_task_creation(self):
        """Test task creation with default values."""
        task = Task(title="Test Task", description="Test description")
        
        assert task.title == "Test Task"
        assert task.description == "Test description"
        assert task.status == TaskStatus.PENDING
        assert task.id is not None
        assert task.created_at is not None
        assert task.updated_at is not None
        assert task.created_at == task.updated_at
    
    def test_task_creation_minimal(self):
        """Test task creation with minimal data."""
        task = Task(title="Minimal Task")
        
        assert task.title == "Minimal Task"
        assert task.description is None
        assert task.status == TaskStatus.PENDING
    
    def test_task_update_timestamp(self):
        """Test task timestamp update."""
        task = Task(title="Test Task")
        original_updated_at = task.updated_at
        
        # Small delay to ensure timestamp difference
        import time
        time.sleep(0.001)
        
        task.update_timestamp()
        
        assert task.updated_at > original_updated_at
    
    def test_task_status_enum(self):
        """Test task status enumeration."""
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.IN_PROGRESS == "in_progress"
        assert TaskStatus.COMPLETED == "completed"
        assert TaskStatus.CANCELLED == "cancelled"
    
    def test_task_serialization(self):
        """Test task serialization to dict."""
        task = Task(title="Test Task", description="Test description")
        task_dict = task.model_dump()
        
        assert task_dict['title'] == "Test Task"
        assert task_dict['description'] == "Test description"
        assert task_dict['status'] == "pending"
        assert 'id' in task_dict
        assert 'created_at' in task_dict
        assert 'updated_at' in task_dict


class TestTaskService:
    """Test TaskService functionality."""
    
    def test_service_initialization(self):
        """Test task service initialization."""
        service = TaskService()
        
        assert service._tasks == {}
        assert service._lock is not None
    
    def test_create_task_success(self, task_service):
        """Test successful task creation."""
        task = task_service.create_task("Test Task", "Test description")
        
        assert task.title == "Test Task"
        assert task.description == "Test description"
        assert task.status == TaskStatus.PENDING
        assert task.id in task_service._tasks
    
    def test_create_task_minimal(self, task_service):
        """Test task creation with minimal data."""
        task = task_service.create_task("Minimal Task")
        
        assert task.title == "Minimal Task"
        assert task.description is None
        assert task.status == TaskStatus.PENDING
    
    def test_create_task_empty_title(self, task_service):
        """Test task creation with empty title."""
        with pytest.raises(ValueError, match="Task title cannot be empty"):
            task_service.create_task("")
        
        with pytest.raises(ValueError, match="Task title cannot be empty"):
            task_service.create_task("   ")
    
    def test_create_task_from_schema(self, task_service):
        """Test task creation from schema."""
        task_data = TaskCreate(title="Schema Task", description="From schema")
        task = task_service.create_task_from_schema(task_data)
        
        assert task.title == "Schema Task"
        assert task.description == "From schema"
    
    def test_get_task_success(self, task_service):
        """Test successful task retrieval."""
        task = task_service.create_task("Test Task")
        retrieved_task = task_service.get_task(task.id)
        
        assert retrieved_task is not None
        assert retrieved_task.id == task.id
        assert retrieved_task.title == "Test Task"
    
    def test_get_task_not_found(self, task_service):
        """Test task retrieval with non-existent ID."""
        non_existent_id = uuid4()
        retrieved_task = task_service.get_task(non_existent_id)
        
        assert retrieved_task is None
    
    def test_update_task_success(self, task_service):
        """Test successful task update."""
        task = task_service.create_task("Original Task")
        update_data = TaskUpdate(title="Updated Task", description="Updated description")
        
        updated_task = task_service.update_task(task.id, update_data)
        
        assert updated_task is not None
        assert updated_task.title == "Updated Task"
        assert updated_task.description == "Updated description"
        assert updated_task.updated_at > updated_task.created_at
    
    def test_update_task_partial(self, task_service):
        """Test partial task update."""
        task = task_service.create_task("Original Task", "Original description")
        update_data = TaskUpdate(title="Updated Task")  # Only title
        
        updated_task = task_service.update_task(task.id, update_data)
        
        assert updated_task.title == "Updated Task"
        assert updated_task.description == "Original description"  # Unchanged
    
    def test_update_task_not_found(self, task_service):
        """Test task update with non-existent ID."""
        non_existent_id = uuid4()
        update_data = TaskUpdate(title="Updated Task")
        
        updated_task = task_service.update_task(non_existent_id, update_data)
        
        assert updated_task is None
    
    def test_update_task_empty_title(self, task_service):
        """Test task update with empty title."""
        task = task_service.create_task("Original Task")
        update_data = TaskUpdate(title="")
        
        with pytest.raises(ValueError, match="Task title cannot be empty"):
            task_service.update_task(task.id, update_data)
    
    def test_update_task_status(self, task_service):
        """Test task status update."""
        task = task_service.create_task("Test Task")
        
        updated_task = task_service.update_task_status(task.id, TaskStatus.COMPLETED)
        
        assert updated_task is not None
        assert updated_task.status == TaskStatus.COMPLETED
        assert updated_task.updated_at > updated_task.created_at
    
    def test_delete_task_success(self, task_service):
        """Test successful task deletion."""
        task = task_service.create_task("Test Task")
        
        success = task_service.delete_task(task.id)
        
        assert success is True
        assert task.id not in task_service._tasks
        assert task_service.get_task(task.id) is None
    
    def test_delete_task_not_found(self, task_service):
        """Test task deletion with non-existent ID."""
        non_existent_id = uuid4()
        
        success = task_service.delete_task(non_existent_id)
        
        assert success is False
    
    def test_list_tasks_empty(self, task_service):
        """Test listing tasks when none exist."""
        tasks = task_service.list_tasks()
        
        assert tasks == []
    
    def test_list_tasks_all(self, task_service):
        """Test listing all tasks."""
        task1 = task_service.create_task("Task 1")
        task2 = task_service.create_task("Task 2")
        task3 = task_service.create_task("Task 3")
        
        tasks = task_service.list_tasks()
        
        assert len(tasks) == 3
        # Should be sorted by creation date (newest first)
        assert tasks[0].title == "Task 3"
        assert tasks[1].title == "Task 2"
        assert tasks[2].title == "Task 1"
    
    def test_list_tasks_with_status_filter(self, task_service):
        """Test listing tasks with status filter."""
        task1 = task_service.create_task("Task 1")
        task2 = task_service.create_task("Task 2")
        task_service.update_task_status(task2.id, TaskStatus.COMPLETED)
        task3 = task_service.create_task("Task 3")
        
        pending_tasks = task_service.list_tasks(status=TaskStatus.PENDING)
        completed_tasks = task_service.list_tasks(status=TaskStatus.COMPLETED)
        
        assert len(pending_tasks) == 2
        assert len(completed_tasks) == 1
        assert completed_tasks[0].id == task2.id
    
    def test_list_tasks_with_date_filter(self, task_service):
        """Test listing tasks with date filter."""
        today = date.today()
        task1 = task_service.create_task("Today Task")
        
        today_tasks = task_service.list_tasks(date_filter=today)
        yesterday_tasks = task_service.list_tasks(date_filter=date(2023, 1, 1))
        
        assert len(today_tasks) == 1
        assert len(yesterday_tasks) == 0
        assert today_tasks[0].id == task1.id
    
    def test_list_tasks_with_pagination(self, task_service):
        """Test listing tasks with pagination."""
        # Create 5 tasks
        for i in range(5):
            task_service.create_task(f"Task {i}")
        
        # Test limit
        limited_tasks = task_service.list_tasks(limit=3)
        assert len(limited_tasks) == 3
        
        # Test offset
        offset_tasks = task_service.list_tasks(offset=2, limit=2)
        assert len(offset_tasks) == 2
        
        # Test offset beyond available tasks
        beyond_tasks = task_service.list_tasks(offset=10)
        assert len(beyond_tasks) == 0
    
    def test_get_task_count(self, task_service):
        """Test task counting."""
        assert task_service.get_task_count() == 0
        
        task1 = task_service.create_task("Task 1")
        task2 = task_service.create_task("Task 2")
        task_service.update_task_status(task2.id, TaskStatus.COMPLETED)
        
        assert task_service.get_task_count() == 2
        assert task_service.get_task_count(status=TaskStatus.PENDING) == 1
        assert task_service.get_task_count(status=TaskStatus.COMPLETED) == 1
    
    def test_get_tasks_by_status(self, task_service):
        """Test task counting by status."""
        task1 = task_service.create_task("Task 1")
        task2 = task_service.create_task("Task 2")
        task3 = task_service.create_task("Task 3")
        
        task_service.update_task_status(task2.id, TaskStatus.COMPLETED)
        task_service.update_task_status(task3.id, TaskStatus.IN_PROGRESS)
        
        counts = task_service.get_tasks_by_status()
        
        assert counts[TaskStatus.PENDING] == 1
        assert counts[TaskStatus.IN_PROGRESS] == 1
        assert counts[TaskStatus.COMPLETED] == 1
        assert counts[TaskStatus.CANCELLED] == 0
    
    def test_search_tasks_success(self, task_service):
        """Test successful task search."""
        task1 = task_service.create_task("Project planning", "Plan the new project")
        task2 = task_service.create_task("Code review", "Review the project code")
        task3 = task_service.create_task("Meeting", "Team meeting")
        
        # Search in titles
        project_tasks = task_service.search_tasks("project")
        assert len(project_tasks) == 2
        
        # Search in descriptions
        review_tasks = task_service.search_tasks("review")
        assert len(review_tasks) == 1
        assert review_tasks[0].id == task2.id
        
        # Case insensitive search
        planning_tasks = task_service.search_tasks("PLANNING")
        assert len(planning_tasks) == 1
        assert planning_tasks[0].id == task1.id
    
    def test_search_tasks_empty_query(self, task_service):
        """Test task search with empty query."""
        task_service.create_task("Test Task")
        
        results = task_service.search_tasks("")
        assert results == []
        
        results = task_service.search_tasks("   ")
        assert results == []
    
    def test_search_tasks_with_limit(self, task_service):
        """Test task search with limit."""
        for i in range(5):
            task_service.create_task(f"Test task {i}", "Test description")
        
        results = task_service.search_tasks("test", limit=3)
        assert len(results) == 3
    
    def test_bulk_create_tasks(self, task_service):
        """Test bulk task creation."""
        tasks_data = [
            TaskCreate(title="Task 1", description="Description 1"),
            TaskCreate(title="Task 2", description="Description 2"),
            TaskCreate(title="Task 3", description="Description 3")
        ]
        
        created_tasks = task_service.bulk_create_tasks(tasks_data)
        
        assert len(created_tasks) == 3
        assert task_service.get_task_count() == 3
        
        for i, task in enumerate(created_tasks):
            assert task.title == f"Task {i + 1}"
            assert task.description == f"Description {i + 1}"
    
    def test_bulk_create_tasks_empty(self, task_service):
        """Test bulk task creation with empty list."""
        created_tasks = task_service.bulk_create_tasks([])
        
        assert created_tasks == []
        assert task_service.get_task_count() == 0
    
    def test_bulk_create_tasks_with_errors(self, task_service):
        """Test bulk task creation with some invalid tasks."""
        tasks_data = [
            TaskCreate(title="Valid Task", description="Valid description"),
            TaskCreate(title="", description="Invalid - empty title"),  # This will cause error
            TaskCreate(title="Another Valid Task", description="Another valid description")
        ]
        
        # Mock the Task creation to raise error for empty title
        with patch('app.services.task_service.Task') as mock_task:
            def side_effect(*args, **kwargs):
                if kwargs.get('title') == "":
                    raise ValueError("Empty title")
                return MagicMock(id=uuid4(), title=kwargs.get('title'), description=kwargs.get('description'))
            
            mock_task.side_effect = side_effect
            
            created_tasks = task_service.bulk_create_tasks(tasks_data)
            
            # Should create only valid tasks
            assert len(created_tasks) == 2
    
    def test_clear_all_tasks(self, task_service):
        """Test clearing all tasks."""
        task_service.create_task("Task 1")
        task_service.create_task("Task 2")
        task_service.create_task("Task 3")
        
        assert task_service.get_task_count() == 3
        
        cleared_count = task_service.clear_all_tasks()
        
        assert cleared_count == 3
        assert task_service.get_task_count() == 0
        assert task_service.list_tasks() == []
    
    def test_get_statistics(self, task_service):
        """Test task statistics."""
        # Empty statistics
        stats = task_service.get_statistics()
        assert stats['total_tasks'] == 0
        assert stats['completion_rate'] == 0
        assert stats['oldest_task'] is None
        assert stats['newest_task'] is None
        
        # Create tasks with different statuses
        task1 = task_service.create_task("Task 1")
        task2 = task_service.create_task("Task 2")
        task3 = task_service.create_task("Task 3")
        
        task_service.update_task_status(task2.id, TaskStatus.COMPLETED)
        task_service.update_task_status(task3.id, TaskStatus.COMPLETED)
        
        stats = task_service.get_statistics()
        
        assert stats['total_tasks'] == 3
        assert stats['completion_rate'] == 66.67  # 2/3 * 100, rounded
        assert stats['oldest_task']['id'] == str(task1.id)
        assert stats['newest_task']['id'] == str(task3.id)
        assert TaskStatus.COMPLETED in stats['status_counts']
        assert stats['status_counts'][TaskStatus.COMPLETED] == 2


class TestTaskServiceThreadSafety:
    """Test task service thread safety."""
    
    def test_concurrent_task_creation(self, task_service):
        """Test concurrent task creation."""
        import threading
        import time
        
        created_tasks = []
        errors = []
        
        def create_task_worker(task_id):
            try:
                task = task_service.create_task(f"Concurrent Task {task_id}")
                created_tasks.append(task)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=create_task_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All tasks should be created without errors
        assert len(errors) == 0
        assert len(created_tasks) == 10
        assert task_service.get_task_count() == 10
    
    def test_concurrent_task_updates(self, task_service):
        """Test concurrent task updates."""
        import threading
        
        # Create a task to update
        task = task_service.create_task("Test Task")
        
        update_results = []
        errors = []
        
        def update_task_worker(status):
            try:
                result = task_service.update_task_status(task.id, status)
                update_results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Create threads to update task status
        statuses = [TaskStatus.IN_PROGRESS, TaskStatus.COMPLETED, TaskStatus.CANCELLED]
        threads = []
        
        for status in statuses:
            thread = threading.Thread(target=update_task_worker, args=(status,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All updates should complete without errors
        assert len(errors) == 0
        assert len(update_results) == 3
        
        # Final task should have one of the statuses
        final_task = task_service.get_task(task.id)
        assert final_task.status in statuses


class TestLangGraphToolIntegration:
    """Test LangGraph tool integration."""
    
    @pytest.mark.asyncio
    async def test_task_create_tool_success(self):
        """Test successful task creation via LangGraph tool."""
        with patch('app.graph.tools.get_task_service') as mock_get_service:
            # Mock task service
            mock_service = MagicMock()
            mock_task = MagicMock()
            mock_task.id = "test-id"
            mock_task.title = "Review quarterly report"
            mock_task.status = TaskStatus.PENDING
            mock_service.create_task.return_value = mock_task
            mock_get_service.return_value = mock_service
            
            # Execute tool
            result = await task_create_tool("Create a task to review the quarterly report")
            
            assert "successfully created" in result.lower()
            assert "test-id" in result
            assert "Review quarterly report" in result
            mock_service.create_task.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_task_create_tool_service_unavailable(self):
        """Test task creation tool when service is unavailable."""
        with patch('app.graph.tools.get_task_service', return_value=None):
            result = await task_create_tool("Create a task")
            
            assert "Task service not available" in result
    
    @pytest.mark.asyncio
    async def test_task_create_tool_error(self):
        """Test task creation tool error handling."""
        with patch('app.graph.tools.get_task_service') as mock_get_service:
            # Mock service to raise error
            mock_service = MagicMock()
            mock_service.create_task.side_effect = Exception("Creation failed")
            mock_get_service.return_value = mock_service
            
            result = await task_create_tool("Create a task")
            
            assert "Error creating task" in result
            assert "Creation failed" in result
    
    @pytest.mark.asyncio
    async def test_task_update_tool_success(self):
        """Test successful task update via LangGraph tool."""
        with patch('app.graph.tools.get_task_service') as mock_get_service:
            # Mock task service
            mock_service = MagicMock()
            mock_task = MagicMock()
            mock_task.id = "test-id"
            mock_task.title = "Test Task"
            mock_task.status = TaskStatus.COMPLETED
            mock_service.get_task.return_value = mock_task
            mock_service.update_task_status.return_value = mock_task
            mock_get_service.return_value = mock_service
            
            # Execute tool
            result = await task_update_tool("test-id", "completed")
            
            assert "successfully updated" in result.lower()
            assert "test-id" in result
            assert "completed" in result
            mock_service.update_task_status.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_task_update_tool_invalid_status(self):
        """Test task update tool with invalid status."""
        with patch('app.graph.tools.get_task_service') as mock_get_service:
            mock_service = MagicMock()
            mock_get_service.return_value = mock_service
            
            result = await task_update_tool("test-id", "invalid_status")
            
            assert "Invalid status" in result
    
    @pytest.mark.asyncio
    async def test_task_update_tool_task_not_found(self):
        """Test task update tool with non-existent task."""
        with patch('app.graph.tools.get_task_service') as mock_get_service:
            # Mock service to return None for get_task
            mock_service = MagicMock()
            mock_service.get_task.return_value = None
            mock_get_service.return_value = mock_service
            
            result = await task_update_tool("nonexistent-id", "completed")
            
            assert "Task not found" in result
    
    @pytest.mark.asyncio
    async def test_task_list_tool_success(self):
        """Test successful task listing via LangGraph tool."""
        with patch('app.graph.tools.get_task_service') as mock_get_service:
            # Mock task service
            mock_service = MagicMock()
            mock_tasks = [
                MagicMock(id="1", title="Task 1", status=TaskStatus.PENDING, 
                         created_at=datetime.now()),
                MagicMock(id="2", title="Task 2", status=TaskStatus.COMPLETED, 
                         created_at=datetime.now())
            ]
            mock_service.list_tasks.return_value = mock_tasks
            mock_get_service.return_value = mock_service
            
            # Execute tool
            result = await task_list_tool("all")
            
            assert "Task 1" in result
            assert "Task 2" in result
            assert "pending" in result
            assert "completed" in result
            mock_service.list_tasks.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_task_list_tool_with_status_filter(self):
        """Test task listing tool with status filter."""
        with patch('app.graph.tools.get_task_service') as mock_get_service:
            # Mock task service
            mock_service = MagicMock()
            mock_tasks = [
                MagicMock(id="1", title="Pending Task", status=TaskStatus.PENDING, 
                         created_at=datetime.now())
            ]
            mock_service.list_tasks.return_value = mock_tasks
            mock_get_service.return_value = mock_service
            
            # Execute tool
            result = await task_list_tool("pending")
            
            assert "Pending Task" in result
            mock_service.list_tasks.assert_called_once_with(status=TaskStatus.PENDING, limit=10)
    
    @pytest.mark.asyncio
    async def test_task_list_tool_empty_results(self):
        """Test task listing tool with no tasks."""
        with patch('app.graph.tools.get_task_service') as mock_get_service:
            # Mock service to return empty list
            mock_service = MagicMock()
            mock_service.list_tasks.return_value = []
            mock_get_service.return_value = mock_service
            
            result = await task_list_tool("all")
            
            assert "No tasks found" in result


class TestTaskRoutes:
    """Test task API routes."""
    
    def test_create_task_success(self, client, sample_task_data):
        """Test successful task creation via API."""
        with patch('app.deps.get_task_service') as mock_get_service:
            # Mock task service
            mock_service = MagicMock()
            mock_task = MagicMock()
            mock_task.id = "test-id"
            mock_task.title = "Test Task"
            mock_task.description = "This is a test task description"
            mock_task.status = TaskStatus.PENDING
            mock_task.created_at = datetime.now()
            mock_task.updated_at = datetime.now()
            mock_service.create_task_from_schema.return_value = mock_task
            mock_get_service.return_value = mock_service
            
            response = client.post("/tasks/", json=sample_task_data)
            
            assert response.status_code == 201
            data = response.json()
            assert data['title'] == "Test Task"
            assert data['status'] == "pending"
    
    def test_create_task_validation_error(self, client):
        """Test task creation with validation error."""
        invalid_data = {"title": ""}  # Empty title
        
        response = client.post("/tasks/", json=invalid_data)
        
        assert response.status_code == 422
        assert "validation error" in response.json()['error'].lower()
    
    def test_list_tasks_success(self, client):
        """Test successful task listing via API."""
        with patch('app.deps.get_task_service') as mock_get_service:
            # Mock task service
            mock_service = MagicMock()
            mock_tasks = [
                MagicMock(id="1", title="Task 1", description="Desc 1", 
                         status=TaskStatus.PENDING, created_at=datetime.now(), 
                         updated_at=datetime.now()),
                MagicMock(id="2", title="Task 2", description="Desc 2", 
                         status=TaskStatus.COMPLETED, created_at=datetime.now(), 
                         updated_at=datetime.now())
            ]
            mock_service.list_tasks.return_value = mock_tasks
            mock_get_service.return_value = mock_service
            
            response = client.get("/tasks/")
            
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2
            assert data[0]['title'] == "Task 1"
            assert data[1]['title'] == "Task 2"
    
    def test_list_tasks_with_filters(self, client):
        """Test task listing with query parameters."""
        with patch('app.deps.get_task_service') as mock_get_service:
            mock_service = MagicMock()
            mock_service.list_tasks.return_value = []
            mock_get_service.return_value = mock_service
            
            response = client.get("/tasks/?status=pending&limit=5&offset=0")
            
            assert response.status_code == 200
            mock_service.list_tasks.assert_called_once_with(
                status=TaskStatus.PENDING, date_filter=None, limit=5, offset=0
            )
    
    def test_get_task_success(self, client):
        """Test successful task retrieval via API."""
        task_id = str(uuid4())
        
        with patch('app.deps.get_task_service') as mock_get_service:
            # Mock task service
            mock_service = MagicMock()
            mock_task = MagicMock()
            mock_task.id = task_id
            mock_task.title = "Test Task"
            mock_task.description = "Test description"
            mock_task.status = TaskStatus.PENDING
            mock_task.created_at = datetime.now()
            mock_task.updated_at = datetime.now()
            mock_service.get_task.return_value = mock_task
            mock_get_service.return_value = mock_service
            
            response = client.get(f"/tasks/{task_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert data['title'] == "Test Task"
    
    def test_get_task_not_found(self, client):
        """Test task retrieval with non-existent ID."""
        task_id = str(uuid4())
        
        with patch('app.deps.get_task_service') as mock_get_service:
            # Mock service to return None
            mock_service = MagicMock()
            mock_service.get_task.return_value = None
            mock_get_service.return_value = mock_service
            
            response = client.get(f"/tasks/{task_id}")
            
            assert response.status_code == 404
            assert "not found" in response.json()['detail']
    
    def test_update_task_success(self, client):
        """Test successful task update via API."""
        task_id = str(uuid4())
        update_data = {"title": "Updated Task", "status": "completed"}
        
        with patch('app.deps.get_task_service') as mock_get_service:
            # Mock task service
            mock_service = MagicMock()
            mock_task = MagicMock()
            mock_task.id = task_id
            mock_task.title = "Updated Task"
            mock_task.description = "Test description"
            mock_task.status = TaskStatus.COMPLETED
            mock_task.created_at = datetime.now()
            mock_task.updated_at = datetime.now()
            mock_service.update_task.return_value = mock_task
            mock_get_service.return_value = mock_service
            
            response = client.patch(f"/tasks/{task_id}", json=update_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data['title'] == "Updated Task"
            assert data['status'] == "completed"
    
    def test_delete_task_success(self, client):
        """Test successful task deletion via API."""
        task_id = str(uuid4())
        
        with patch('app.deps.get_task_service') as mock_get_service:
            # Mock task service
            mock_service = MagicMock()
            mock_service.delete_task.return_value = True
            mock_get_service.return_value = mock_service
            
            response = client.delete(f"/tasks/{task_id}")
            
            assert response.status_code == 204
    
    def test_delete_task_not_found(self, client):
        """Test task deletion with non-existent ID."""
        task_id = str(uuid4())
        
        with patch('app.deps.get_task_service') as mock_get_service:
            # Mock service to return False
            mock_service = MagicMock()
            mock_service.delete_task.return_value = False
            mock_get_service.return_value = mock_service
            
            response = client.delete(f"/tasks/{task_id}")
            
            assert response.status_code == 404
    
    def test_search_tasks_success(self, client):
        """Test successful task search via API."""
        with patch('app.deps.get_task_service') as mock_get_service:
            # Mock task service
            mock_service = MagicMock()
            mock_tasks = [
                MagicMock(id="1", title="Project Task", description="Project work", 
                         status=TaskStatus.PENDING, created_at=datetime.now(), 
                         updated_at=datetime.now())
            ]
            mock_service.search_tasks.return_value = mock_tasks
            mock_get_service.return_value = mock_service
            
            response = client.get("/tasks/search/?q=project&limit=5")
            
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]['title'] == "Project Task"
            mock_service.search_tasks.assert_called_once_with("project", limit=5)
    
    def test_get_task_statistics(self, client):
        """Test task statistics endpoint."""
        with patch('app.deps.get_task_service') as mock_get_service:
            # Mock task service
            mock_service = MagicMock()
            mock_stats = {
                'total_tasks': 10,
                'completion_rate': 60.0,
                'status_counts': {
                    'pending': 4,
                    'completed': 6
                }
            }
            mock_service.get_statistics.return_value = mock_stats
            mock_get_service.return_value = mock_service
            
            response = client.get("/tasks/stats/")
            
            assert response.status_code == 200
            data = response.json()
            assert data['total_tasks'] == 10
            assert data['completion_rate'] == 60.0

