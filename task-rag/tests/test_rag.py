"""Tests for retrieval and search functionality."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from langchain_core.documents import Document

from app.services.rag_service import RAGService
from app.graph.tools import retriever_tool
from app.graph.state import AgentState


class TestRAGRetrieval:
    """Test RAG retrieval functionality."""
    
    def test_retriever_creation(self, rag_service, mock_chroma):
        """Test retriever creation with default parameters."""
        retriever = rag_service.get_retriever()
        
        assert retriever is not None
        mock_chroma.as_retriever.assert_called_once_with(
            search_type="similarity",
            search_kwargs={"k": 5}  # Default k value
        )
    
    def test_retriever_creation_custom_k(self, rag_service, mock_chroma):
        """Test retriever creation with custom k parameter."""
        retriever = rag_service.get_retriever(k=3)
        
        assert retriever is not None
        mock_chroma.as_retriever.assert_called_once_with(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
    
    def test_similarity_search_success(self, rag_service, mock_chroma):
        """Test successful similarity search."""
        query = "What is the project timeline?"
        
        results = rag_service.search_documents(query, k=3)
        
        assert len(results) == 2  # Based on mock setup
        assert results[0].page_content == "Test content 1"
        assert results[0].metadata['source'] == 'test.pdf'
        assert results[1].page_content == "Test content 2"
        mock_chroma.similarity_search.assert_called_once_with(query, k=3)
    
    def test_similarity_search_with_scores(self, rag_service, mock_chroma):
        """Test similarity search with relevance scores."""
        query = "project management"
        
        results = rag_service.search_documents_with_scores(query, k=2)
        
        assert len(results) == 2
        doc1, score1 = results[0]
        doc2, score2 = results[1]
        
        assert score1 == 0.9
        assert score2 == 0.8
        assert doc1.page_content == "Test content 1"
        assert doc2.page_content == "Test content 2"
        mock_chroma.similarity_search_with_score.assert_called_once_with(query, k=2)
    
    def test_search_documents_error_handling(self, rag_service, mock_chroma):
        """Test error handling in document search."""
        mock_chroma.similarity_search.side_effect = Exception("Search failed")
        
        with pytest.raises(Exception, match="Search failed"):
            rag_service.search_documents("test query")
    
    def test_search_documents_empty_query(self, rag_service, mock_chroma):
        """Test search with empty query."""
        results = rag_service.search_documents("", k=5)
        
        # Should still call the search method
        mock_chroma.similarity_search.assert_called_once_with("", k=5)
    
    def test_collection_info_retrieval(self, rag_service, mock_chroma):
        """Test collection information retrieval."""
        info = rag_service.get_collection_info()
        
        assert info['document_count'] == 5
        assert info['collection_name'] == 'task_documents'
        assert 'chroma_dir' in info
        mock_chroma._collection.count.assert_called_once()
    
    def test_collection_info_error_handling(self, rag_service, mock_chroma):
        """Test error handling in collection info retrieval."""
        mock_chroma._collection.count.side_effect = Exception("Collection error")
        
        info = rag_service.get_collection_info()
        
        assert info['document_count'] == 0
        assert 'error' in info
        assert info['error'] == "Collection error"


class TestDocumentManagement:
    """Test document management functionality."""
    
    def test_delete_documents_by_source_success(self, rag_service, mock_chroma):
        """Test successful document deletion by source."""
        source = "test_document.pdf"
        
        success = rag_service.delete_documents_by_source(source)
        
        assert success is True
        mock_chroma._collection.get.assert_called_once_with(where={"source": source})
        mock_chroma._collection.delete.assert_called_once_with(ids=['1', '2', '3'])
    
    def test_delete_documents_by_source_not_found(self, rag_service, mock_chroma):
        """Test document deletion when source not found."""
        # Mock empty result
        mock_chroma._collection.get.return_value = {'ids': []}
        
        success = rag_service.delete_documents_by_source("nonexistent.pdf")
        
        assert success is False
        mock_chroma._collection.delete.assert_not_called()
    
    def test_delete_documents_by_source_error(self, rag_service, mock_chroma):
        """Test error handling in document deletion."""
        mock_chroma._collection.get.side_effect = Exception("Delete error")
        
        success = rag_service.delete_documents_by_source("test.pdf")
        
        assert success is False
    
    def test_clear_collection_success(self, rag_service, mock_chroma):
        """Test successful collection clearing."""
        success = rag_service.clear_collection()
        
        assert success is True
        mock_chroma._collection.get.assert_called_once()
        mock_chroma._collection.delete.assert_called_once_with(ids=['1', '2', '3'])
    
    def test_clear_collection_empty(self, rag_service, mock_chroma):
        """Test clearing empty collection."""
        mock_chroma._collection.get.return_value = {'ids': []}
        
        success = rag_service.clear_collection()
        
        assert success is True
        mock_chroma._collection.delete.assert_not_called()
    
    def test_clear_collection_error(self, rag_service, mock_chroma):
        """Test error handling in collection clearing."""
        mock_chroma._collection.get.side_effect = Exception("Clear error")
        
        success = rag_service.clear_collection()
        
        assert success is False


class TestRAGServiceIntegration:
    """Test RAG service integration with other components."""
    
    def test_rag_service_global_instance(self, test_settings):
        """Test global RAG service instance management."""
        from app.services.rag_service import initialize_rag_service, get_rag_service
        
        with patch('app.services.rag_service.OpenAIEmbeddings'), \
             patch('app.services.rag_service.Chroma'):
            
            # Initialize service
            service = initialize_rag_service(test_settings)
            
            # Get service
            retrieved_service = get_rag_service()
            
            assert service is retrieved_service
            assert retrieved_service is not None
    
    def test_rag_service_settings_integration(self, test_settings, mock_langchain_openai, mock_chroma):
        """Test RAG service integration with settings."""
        rag_service = RAGService(test_settings)
        
        assert rag_service.settings.chunk_size == 1000
        assert rag_service.settings.chunk_overlap == 200
        assert rag_service.settings.retrieval_k == 5
        assert rag_service.settings.embeddings_model == "text-embedding-3-small"
    
    def test_text_splitter_configuration(self, test_settings, mock_langchain_openai, mock_chroma):
        """Test text splitter configuration."""
        rag_service = RAGService(test_settings)
        
        splitter = rag_service.text_splitter
        assert splitter.chunk_size == test_settings.chunk_size
        assert splitter.chunk_overlap == test_settings.chunk_overlap


class TestRetrieverTool:
    """Test the retriever tool for LangGraph integration."""
    
    @pytest.mark.asyncio
    async def test_retriever_tool_success(self, mock_chroma):
        """Test successful retriever tool execution."""
        with patch('app.graph.tools.get_rag_service') as mock_get_rag:
            # Mock RAG service
            mock_rag_service = MagicMock()
            mock_rag_service.search_documents_with_scores.return_value = [
                (Document(page_content="Project timeline is 6 months", 
                         metadata={'source': 'project.pdf', 'page': 1}), 0.95),
                (Document(page_content="Milestones are defined quarterly", 
                         metadata={'source': 'project.pdf', 'page': 2}), 0.87)
            ]
            mock_get_rag.return_value = mock_rag_service
            
            # Execute tool
            result = await retriever_tool("What is the project timeline?")
            
            assert "Project timeline is 6 months" in result
            assert "Milestones are defined quarterly" in result
            assert "project.pdf" in result
            assert "page 1" in result
            assert "page 2" in result
            mock_rag_service.search_documents_with_scores.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_retriever_tool_no_results(self):
        """Test retriever tool with no search results."""
        with patch('app.graph.tools.get_rag_service') as mock_get_rag:
            # Mock RAG service with empty results
            mock_rag_service = MagicMock()
            mock_rag_service.search_documents_with_scores.return_value = []
            mock_get_rag.return_value = mock_rag_service
            
            # Execute tool
            result = await retriever_tool("nonexistent query")
            
            assert "No relevant documents found" in result
    
    @pytest.mark.asyncio
    async def test_retriever_tool_service_not_available(self):
        """Test retriever tool when RAG service is not available."""
        with patch('app.graph.tools.get_rag_service', return_value=None):
            # Execute tool
            result = await retriever_tool("test query")
            
            assert "RAG service not available" in result
    
    @pytest.mark.asyncio
    async def test_retriever_tool_error_handling(self):
        """Test retriever tool error handling."""
        with patch('app.graph.tools.get_rag_service') as mock_get_rag:
            # Mock RAG service to raise error
            mock_rag_service = MagicMock()
            mock_rag_service.search_documents_with_scores.side_effect = Exception("Search error")
            mock_get_rag.return_value = mock_rag_service
            
            # Execute tool
            result = await retriever_tool("test query")
            
            assert "Error searching documents" in result
            assert "Search error" in result


class TestRAGPerformance:
    """Test RAG performance and optimization."""
    
    def test_search_performance_with_large_results(self, rag_service, mock_chroma):
        """Test search performance with large result sets."""
        # Mock large result set
        large_results = []
        for i in range(100):
            doc = MagicMock()
            doc.page_content = f"Content {i}"
            doc.metadata = {'source': f'doc_{i}.pdf', 'page': i % 10}
            large_results.append((doc, 0.9 - i * 0.001))
        
        mock_chroma.similarity_search_with_score.return_value = large_results
        
        # Search with limit
        results = rag_service.search_documents_with_scores("test", k=10)
        
        assert len(results) == 100  # Mock returns all results
        mock_chroma.similarity_search_with_score.assert_called_once_with("test", k=10)
    
    def test_retriever_caching_behavior(self, rag_service, mock_chroma):
        """Test retriever caching behavior."""
        # Get retriever multiple times
        retriever1 = rag_service.get_retriever(k=5)
        retriever2 = rag_service.get_retriever(k=5)
        
        # Should create new retriever each time (no caching in current implementation)
        assert mock_chroma.as_retriever.call_count == 2
    
    def test_memory_efficient_search(self, rag_service, mock_chroma):
        """Test memory-efficient search operations."""
        # Test with various query sizes
        queries = [
            "short",
            "medium length query with several words",
            "very long query " * 50  # Very long query
        ]
        
        for query in queries:
            results = rag_service.search_documents(query, k=3)
            assert len(results) == 2  # Based on mock setup
        
        # Should handle all queries without issues
        assert mock_chroma.similarity_search.call_count == 3


class TestRAGErrorRecovery:
    """Test RAG error recovery and resilience."""
    
    def test_service_recovery_after_error(self, rag_service, mock_chroma):
        """Test service recovery after encountering errors."""
        # First call fails
        mock_chroma.similarity_search.side_effect = Exception("Temporary error")
        
        with pytest.raises(Exception):
            rag_service.search_documents("test query")
        
        # Reset mock to succeed
        mock_chroma.similarity_search.side_effect = None
        mock_chroma.similarity_search.return_value = [
            MagicMock(page_content="Recovery test", metadata={'source': 'test.pdf'})
        ]
        
        # Second call should succeed
        results = rag_service.search_documents("test query")
        assert len(results) == 1
        assert results[0].page_content == "Recovery test"
    
    def test_partial_failure_handling(self, rag_service, mock_chroma):
        """Test handling of partial failures in operations."""
        # Mock collection operations to partially fail
        mock_chroma._collection.get.return_value = {'ids': ['1', '2', '3']}
        mock_chroma._collection.delete.side_effect = Exception("Partial delete failure")
        
        # Should handle the error gracefully
        success = rag_service.delete_documents_by_source("test.pdf")
        assert success is False
    
    def test_concurrent_access_safety(self, rag_service, mock_chroma):
        """Test thread safety of RAG operations."""
        import threading
        import time
        
        results = []
        errors = []
        
        def search_worker():
            try:
                result = rag_service.search_documents("concurrent test", k=2)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=search_worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All operations should complete without errors
        assert len(errors) == 0
        assert len(results) == 5

