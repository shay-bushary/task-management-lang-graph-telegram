"""Shared test fixtures and configuration for the test suite."""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Add the parent directory to the path for imports
import sys

sys.path.append(str(Path(__file__).parent.parent))

from app.config import Settings
from app.main import create_app
from app.models.task import Task, TaskStatus
from app.services.rag_service import RAGService
from app.services.task_service import TaskService


@pytest.fixture
def test_settings() -> Settings:
    """Create test settings with temporary directories."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test settings
        settings = Settings(
            openai_api_key="test-api-key",
            telegram_bot_token="test-bot-token",
            telegram_webhook_secret="test-webhook-secret",
            telegram_webhook_url="https://test.example.com",
            uploads_dir=temp_path / "uploads",
            chroma_dir=temp_path / "chroma",
            model_name="gpt-3.5-turbo",
            embeddings_model="text-embedding-3-small",
            log_level="DEBUG",
            environment="test",
        )

        # Create directories
        settings.uploads_dir.mkdir(parents=True, exist_ok=True)
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)

        yield settings


@pytest.fixture
def mock_openai():
    """Mock OpenAI API calls."""
    with patch("openai.OpenAI") as mock_client:
        # Mock embeddings
        mock_embeddings = MagicMock()
        mock_embeddings.create.return_value.data = [MagicMock(embedding=[0.1] * 1536)]
        mock_client.return_value.embeddings = mock_embeddings

        # Mock chat completions
        mock_chat = MagicMock()
        mock_chat.create.return_value.choices = [
            MagicMock(message=MagicMock(content="Test response"))
        ]
        mock_client.return_value.chat.completions = mock_chat

        yield mock_client


@pytest.fixture
def mock_langchain_openai():
    """Mock LangChain OpenAI components."""
    with (
        patch("app.services.rag_service.OpenAIEmbeddings") as mock_embeddings,
        patch("app.graph.nodes.ChatOpenAI") as mock_llm,
    ):
        # Mock embeddings
        mock_embeddings_instance = MagicMock()
        mock_embeddings_instance.embed_documents.return_value = [[0.1] * 1536]
        mock_embeddings_instance.embed_query.return_value = [0.1] * 1536
        mock_embeddings.return_value = mock_embeddings_instance

        # Mock LLM
        mock_llm_instance = AsyncMock()
        mock_llm_instance.ainvoke.return_value = MagicMock(content="Test LLM response")
        mock_llm.return_value = mock_llm_instance

        yield {"embeddings": mock_embeddings, "llm": mock_llm}


@pytest.fixture
def mock_chroma():
    """Mock Chroma vector store."""
    with patch("app.services.rag_service.Chroma") as mock_chroma_class:
        mock_chroma_instance = MagicMock()

        # Mock collection
        mock_collection = MagicMock()
        mock_collection.count.return_value = 5
        mock_collection.get.return_value = {"ids": ["1", "2", "3"]}
        mock_collection.delete.return_value = None
        mock_chroma_instance._collection = mock_collection

        # Mock vector store methods
        mock_chroma_instance.add_documents.return_value = ["doc1", "doc2", "doc3"]
        mock_chroma_instance.similarity_search.return_value = [
            MagicMock(
                page_content="Test content 1",
                metadata={"source": "test.pdf", "page": 1},
            ),
            MagicMock(
                page_content="Test content 2",
                metadata={"source": "test.pdf", "page": 2},
            ),
        ]
        mock_chroma_instance.similarity_search_with_score.return_value = [
            (
                MagicMock(
                    page_content="Test content 1", metadata={"source": "test.pdf"}
                ),
                0.9,
            ),
            (
                MagicMock(
                    page_content="Test content 2", metadata={"source": "test.pdf"}
                ),
                0.8,
            ),
        ]
        mock_chroma_instance.as_retriever.return_value = MagicMock()
        mock_chroma_instance.persist.return_value = None

        mock_chroma_class.return_value = mock_chroma_instance

        yield mock_chroma_instance


@pytest.fixture
def mock_pdf_loader():
    """Mock PyPDFLoader."""
    with patch("app.services.rag_service.PyPDFLoader") as mock_loader_class:
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = [
            MagicMock(
                page_content="This is page 1 content",
                metadata={"source": "test.pdf", "page": 0},
            ),
            MagicMock(
                page_content="This is page 2 content",
                metadata={"source": "test.pdf", "page": 1},
            ),
        ]
        mock_loader_class.return_value = mock_loader_instance

        yield mock_loader_instance


@pytest.fixture
def mock_text_splitter():
    """Mock RecursiveCharacterTextSplitter."""
    with patch(
        "app.services.rag_service.RecursiveCharacterTextSplitter"
    ) as mock_splitter_class:
        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_documents.return_value = [
            MagicMock(
                page_content="This is chunk 1",
                metadata={"source": "test.pdf", "page": 0, "chunk": 0},
            ),
            MagicMock(
                page_content="This is chunk 2",
                metadata={"source": "test.pdf", "page": 0, "chunk": 1},
            ),
            MagicMock(
                page_content="This is chunk 3",
                metadata={"source": "test.pdf", "page": 1, "chunk": 0},
            ),
        ]
        mock_splitter_class.return_value = mock_splitter_instance

        yield mock_splitter_instance


@pytest.fixture
def sample_pdf_content() -> bytes:
    """Create sample PDF content for testing."""
    # This is a minimal PDF content for testing
    pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj

4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
72 720 Td
(Test PDF content) Tj
ET
endstream
endobj

xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000206 00000 n 
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
300
%%EOF"""
    return pdf_content


@pytest.fixture
def sample_task() -> Task:
    """Create a sample task for testing."""
    return Task(title="Test Task", description="This is a test task")


@pytest.fixture
def task_service() -> TaskService:
    """Create a task service instance for testing."""
    return TaskService()


@pytest.fixture
def rag_service(test_settings, mock_langchain_openai, mock_chroma) -> RAGService:
    """Create a RAG service instance for testing."""
    return RAGService(test_settings)


@pytest.fixture
def client(test_settings) -> Generator[TestClient, None, None]:
    """Create a test client for the FastAPI app."""
    with patch("app.deps.get_settings", return_value=test_settings):
        app = create_app()
        with TestClient(app) as test_client:
            yield test_client


@pytest.fixture
def mock_telegram_bot():
    """Mock Telegram bot for testing."""
    with patch("aiogram.Bot") as mock_bot_class:
        mock_bot_instance = AsyncMock()
        mock_bot_instance.send_message.return_value = MagicMock(message_id=123)
        mock_bot_instance.edit_message_text.return_value = True
        mock_bot_instance.delete_message.return_value = True
        mock_bot_instance.get_file.return_value = MagicMock(file_path="test/path.pdf")
        mock_bot_instance.download_file.return_value = b"test pdf content"
        mock_bot_instance.set_webhook.return_value = True
        mock_bot_instance.delete_webhook.return_value = True
        mock_bot_instance.get_webhook_info.return_value = MagicMock(
            url="https://test.com"
        )
        mock_bot_instance.get_me.return_value = MagicMock(
            id=123456789, username="test_bot", first_name="Test Bot", is_bot=True
        )

        mock_bot_class.return_value = mock_bot_instance

        yield mock_bot_instance


@pytest.fixture
def mock_websocket():
    """Mock WebSocket connection for testing."""
    with patch("aiohttp.ClientSession") as mock_session_class:
        mock_session = AsyncMock()
        mock_ws = AsyncMock()

        # Mock WebSocket messages
        mock_messages = [
            MagicMock(
                type=1, data='{"type": "token", "content": "Hello"}'
            ),  # WSMsgType.TEXT = 1
            MagicMock(type=1, data='{"type": "token", "content": " world"}'),
            MagicMock(
                type=1,
                data='{"type": "final_result", "result": {"content": "Hello world"}}',
            ),
        ]

        mock_ws.__aiter__.return_value = iter(mock_messages)
        mock_session.ws_connect.return_value.__aenter__.return_value = mock_ws
        mock_session_class.return_value.__aenter__.return_value = mock_session

        yield mock_ws


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Test data fixtures
@pytest.fixture
def sample_task_data():
    """Sample task data for testing."""
    return {"title": "Test Task", "description": "This is a test task description"}


@pytest.fixture
def sample_user_message():
    """Sample user message for testing."""
    return {"message": "Create a task to review the quarterly report"}


@pytest.fixture
def sample_tasks_bulk():
    """Sample bulk tasks data for testing."""
    return [
        {"title": "Task 1", "description": "Description 1"},
        {"title": "Task 2", "description": "Description 2"},
        {"title": "Task 3", "description": "Description 3"},
    ]
