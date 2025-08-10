"""Tests for PDF upload, chunking, and embedding functionality."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import UploadFile
import io

from app.services.rag_service import RAGService
from app.utils.pdf import validate_pdf_file, safe_save_uploaded_file, get_pdf_metadata


class TestPDFValidation:
    """Test PDF validation utilities."""

    def test_validate_pdf_file_success(self, test_settings, sample_pdf_content):
        """Test successful PDF validation."""
        # Create a temporary PDF file
        pdf_path = test_settings.uploads_dir / "test.pdf"
        pdf_path.write_bytes(sample_pdf_content)

        with patch("app.utils.pdf.PdfReader") as mock_reader:
            mock_reader.return_value.pages = [MagicMock(), MagicMock()]

            is_valid, error_msg = validate_pdf_file(pdf_path)

            assert is_valid is True
            assert error_msg is None

    def test_validate_pdf_file_not_exists(self, test_settings):
        """Test PDF validation with non-existent file."""
        pdf_path = test_settings.uploads_dir / "nonexistent.pdf"

        is_valid, error_msg = validate_pdf_file(pdf_path)

        assert is_valid is False
        assert "does not exist" in error_msg

    def test_validate_pdf_file_wrong_extension(self, test_settings):
        """Test PDF validation with wrong file extension."""
        txt_path = test_settings.uploads_dir / "test.txt"
        txt_path.write_text("Not a PDF")

        is_valid, error_msg = validate_pdf_file(txt_path)

        assert is_valid is False
        assert "not a PDF file" in error_msg

    def test_validate_pdf_file_too_large(self, test_settings):
        """Test PDF validation with file too large."""
        pdf_path = test_settings.uploads_dir / "large.pdf"
        # Create a file larger than 50MB
        large_content = b"x" * (51 * 1024 * 1024)
        pdf_path.write_bytes(large_content)

        is_valid, error_msg = validate_pdf_file(pdf_path)

        assert is_valid is False
        assert "too large" in error_msg

    def test_safe_save_uploaded_file_success(self, test_settings, sample_pdf_content):
        """Test successful file saving."""
        with patch("app.utils.pdf.validate_pdf_file", return_value=(True, None)):
            file_path = safe_save_uploaded_file(
                file_content=sample_pdf_content,
                filename="test.pdf",
                upload_dir=test_settings.uploads_dir,
            )

            assert file_path.exists()
            assert file_path.name == "test.pdf"
            assert file_path.read_bytes() == sample_pdf_content

    def test_safe_save_uploaded_file_invalid_pdf(self, test_settings):
        """Test file saving with invalid PDF."""
        with patch(
            "app.utils.pdf.validate_pdf_file", return_value=(False, "Invalid PDF")
        ):
            with pytest.raises(Exception, match="Invalid PDF"):
                safe_save_uploaded_file(
                    file_content=b"invalid content",
                    filename="test.pdf",
                    upload_dir=test_settings.uploads_dir,
                )

    def test_get_pdf_metadata_success(self, test_settings, sample_pdf_content):
        """Test PDF metadata extraction."""
        pdf_path = test_settings.uploads_dir / "test.pdf"
        pdf_path.write_bytes(sample_pdf_content)

        with patch("app.utils.pdf.PdfReader") as mock_reader:
            mock_reader.return_value.pages = [MagicMock(), MagicMock()]
            mock_reader.return_value.metadata = {
                "/Title": "Test PDF",
                "/Author": "Test Author",
            }

            metadata = get_pdf_metadata(pdf_path)

            assert metadata["num_pages"] == 2
            assert metadata["file_size"] > 0
            assert "title" in metadata
            assert "author" in metadata


class TestRAGService:
    """Test RAG service functionality."""

    def test_rag_service_initialization(
        self, test_settings, mock_langchain_openai, mock_chroma
    ):
        """Test RAG service initialization."""
        rag_service = RAGService(test_settings)

        assert rag_service.settings == test_settings
        assert rag_service.embeddings is not None
        assert rag_service.text_splitter is not None
        assert rag_service.vectorstore is not None

    def test_process_pdf_success(
        self,
        rag_service,
        test_settings,
        sample_pdf_content,
        mock_pdf_loader,
        mock_text_splitter,
        mock_chroma,
    ):
        """Test successful PDF processing."""
        # Create a test PDF file
        pdf_path = test_settings.uploads_dir / "test.pdf"
        pdf_path.write_bytes(sample_pdf_content)

        with (
            patch("app.utils.pdf.validate_pdf_file", return_value=(True, None)),
            patch(
                "app.utils.pdf.get_pdf_metadata",
                return_value={"num_pages": 2, "file_size": 1024},
            ),
        ):
            result = rag_service.process_pdf(pdf_path)

            assert result["success"] is True
            assert result["filename"] == "test.pdf"
            assert result["document_count"] == 2
            assert result["chunk_count"] == 3
            assert "chunk_ids" in result
            assert "metadata" in result

    def test_process_pdf_invalid_file(self, rag_service, test_settings):
        """Test PDF processing with invalid file."""
        pdf_path = test_settings.uploads_dir / "invalid.pdf"

        with patch(
            "app.utils.pdf.validate_pdf_file", return_value=(False, "Invalid PDF")
        ):
            with pytest.raises(ValueError, match="Invalid PDF file"):
                rag_service.process_pdf(pdf_path)

    def test_process_pdf_no_content(
        self, rag_service, test_settings, sample_pdf_content, mock_pdf_loader
    ):
        """Test PDF processing with no extractable content."""
        pdf_path = test_settings.uploads_dir / "empty.pdf"
        pdf_path.write_bytes(sample_pdf_content)

        # Mock loader to return empty documents
        mock_pdf_loader.load.return_value = []

        with patch("app.utils.pdf.validate_pdf_file", return_value=(True, None)):
            with pytest.raises(ValueError, match="No content could be extracted"):
                rag_service.process_pdf(pdf_path)

    def test_get_retriever(self, rag_service, mock_chroma):
        """Test retriever creation."""
        retriever = rag_service.get_retriever(k=3)

        assert retriever is not None
        mock_chroma.as_retriever.assert_called_once()

    def test_search_documents(self, rag_service, mock_chroma):
        """Test document search."""
        query = "test query"

        results = rag_service.search_documents(query, k=2)

        assert len(results) == 2
        assert results[0].page_content == "Test content 1"
        assert results[1].page_content == "Test content 2"
        mock_chroma.similarity_search.assert_called_once_with(query, k=2)

    def test_search_documents_with_scores(self, rag_service, mock_chroma):
        """Test document search with scores."""
        query = "test query"

        results = rag_service.search_documents_with_scores(query, k=2)

        assert len(results) == 2
        assert results[0][1] == 0.9  # First result score
        assert results[1][1] == 0.8  # Second result score
        mock_chroma.similarity_search_with_score.assert_called_once_with(query, k=2)

    def test_get_collection_info(self, rag_service, mock_chroma):
        """Test collection information retrieval."""
        info = rag_service.get_collection_info()

        assert info["document_count"] == 5
        assert info["collection_name"] == "task_documents"
        assert "chroma_dir" in info

    def test_delete_documents_by_source(self, rag_service, mock_chroma):
        """Test document deletion by source."""
        source = "test.pdf"

        success = rag_service.delete_documents_by_source(source)

        assert success is True
        mock_chroma._collection.get.assert_called_once_with(where={"source": source})
        mock_chroma._collection.delete.assert_called_once()

    def test_clear_collection(self, rag_service, mock_chroma):
        """Test collection clearing."""
        success = rag_service.clear_collection()

        assert success is True
        mock_chroma._collection.get.assert_called_once()
        mock_chroma._collection.delete.assert_called_once()


class TestIngestRoutes:
    """Test ingestion API routes."""

    def test_ingest_pdf_success(
        self,
        client,
        sample_pdf_content,
        mock_langchain_openai,
        mock_chroma,
        mock_pdf_loader,
        mock_text_splitter,
    ):
        """Test successful PDF ingestion via API."""
        with (
            patch("app.routes.ingest.safe_save_uploaded_file") as mock_save,
            patch("app.deps.get_rag_service") as mock_get_rag,
            patch("app.services.rag_service.initialize_rag_service"),
        ):
            # Mock file saving
            mock_save.return_value = Path("/tmp/test.pdf")

            # Mock RAG service
            mock_rag_service = MagicMock()
            mock_rag_service.process_pdf.return_value = {
                "success": True,
                "filename": "test.pdf",
                "document_count": 2,
                "chunk_count": 3,
                "chunk_ids": ["1", "2", "3"],
                "metadata": {"num_pages": 2},
            }
            mock_get_rag.return_value = mock_rag_service

            # Create test file
            files = {
                "file": ("test.pdf", io.BytesIO(sample_pdf_content), "application/pdf")
            }

            response = client.post("/ingest/pdf", files=files)

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["filename"] == "test.pdf"
            assert data["document_count"] == 2
            assert data["chunk_count"] == 3

    def test_ingest_pdf_invalid_file_type(self, client):
        """Test PDF ingestion with invalid file type."""
        files = {"file": ("test.txt", io.BytesIO(b"not a pdf"), "text/plain")}

        response = client.post("/ingest/pdf", files=files)

        assert response.status_code == 400
        assert "Only PDF files are supported" in response.json()["detail"]

    def test_ingest_pdf_empty_file(self, client):
        """Test PDF ingestion with empty file."""
        files = {"file": ("test.pdf", io.BytesIO(b""), "application/pdf")}

        response = client.post("/ingest/pdf", files=files)

        assert response.status_code == 400
        assert "Empty file uploaded" in response.json()["detail"]

    def test_ingest_pdf_processing_error(self, client, sample_pdf_content):
        """Test PDF ingestion with processing error."""
        with (
            patch("app.routes.ingest.safe_save_uploaded_file") as mock_save,
            patch("app.deps.get_rag_service") as mock_get_rag,
        ):
            mock_save.return_value = Path("/tmp/test.pdf")

            # Mock RAG service to raise error
            mock_rag_service = MagicMock()
            mock_rag_service.process_pdf.side_effect = Exception("Processing failed")
            mock_get_rag.return_value = mock_rag_service

            files = {
                "file": ("test.pdf", io.BytesIO(sample_pdf_content), "application/pdf")
            }

            response = client.post("/ingest/pdf", files=files)

            assert response.status_code == 500
            assert "Error processing PDF" in response.json()["detail"]

    def test_ingest_tasks_success(self, client, sample_tasks_bulk):
        """Test successful bulk task ingestion."""
        with patch("app.deps.get_task_service") as mock_get_task:
            # Mock task service
            mock_task_service = MagicMock()
            mock_tasks = [
                MagicMock(
                    id="1",
                    title="Task 1",
                    description="Description 1",
                    status="pending",
                    created_at="2023-01-01T00:00:00",
                    updated_at="2023-01-01T00:00:00",
                ),
                MagicMock(
                    id="2",
                    title="Task 2",
                    description="Description 2",
                    status="pending",
                    created_at="2023-01-01T00:00:00",
                    updated_at="2023-01-01T00:00:00",
                ),
                MagicMock(
                    id="3",
                    title="Task 3",
                    description="Description 3",
                    status="pending",
                    created_at="2023-01-01T00:00:00",
                    updated_at="2023-01-01T00:00:00",
                ),
            ]
            mock_task_service.bulk_create_tasks.return_value = mock_tasks
            mock_get_task.return_value = mock_task_service

            response = client.post("/ingest/tasks", json=sample_tasks_bulk)

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 3
            assert data[0]["title"] == "Task 1"

    def test_ingest_tasks_empty_list(self, client):
        """Test bulk task ingestion with empty list."""
        response = client.post("/ingest/tasks", json=[])

        assert response.status_code == 400
        assert "No tasks provided" in response.json()["detail"]

    def test_ingest_tasks_too_many(self, client):
        """Test bulk task ingestion with too many tasks."""
        large_task_list = [
            {"title": f"Task {i}", "description": f"Desc {i}"} for i in range(101)
        ]

        response = client.post("/ingest/tasks", json=large_task_list)

        assert response.status_code == 400
        assert "Too many tasks" in response.json()["detail"]

    def test_ingest_status_success(self, client):
        """Test ingestion status endpoint."""
        with (
            patch("app.deps.get_rag_service") as mock_get_rag,
            patch("app.deps.get_task_service") as mock_get_task,
        ):
            # Mock services
            mock_rag_service = MagicMock()
            mock_rag_service.get_collection_info.return_value = {
                "document_count": 5,
                "collection_name": "test_collection",
            }
            mock_get_rag.return_value = mock_rag_service

            mock_task_service = MagicMock()
            mock_task_service.get_statistics.return_value = {
                "total_tasks": 10,
                "completion_rate": 50.0,
            }
            mock_get_task.return_value = mock_task_service

            response = client.get("/ingest/status")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "rag_collection" in data
            assert "task_statistics" in data
