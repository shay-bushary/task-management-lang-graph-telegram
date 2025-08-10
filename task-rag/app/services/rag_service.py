"""RAG service for document processing and retrieval."""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import Settings
from ..utils.pdf import get_pdf_metadata, validate_pdf_file

logger = logging.getLogger(__name__)


class RAGService:
    """Service for RAG operations including document processing and retrieval."""
    
    def __init__(self, settings: Settings):
        """Initialize the RAG service.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.embeddings = OpenAIEmbeddings(
            model=settings.embeddings_model,
            api_key=settings.openai_api_key
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Ensure chroma directory exists
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize vector store
        self.vectorstore = Chroma(
            persist_directory=str(settings.chroma_dir),
            embedding_function=self.embeddings,
            collection_name="task_documents"
        )
        
        logger.info(f"RAG service initialized with Chroma at {settings.chroma_dir}")
    
    def process_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Process a PDF file and add it to the vector store.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary containing processing results
            
        Raises:
            Exception: If PDF processing fails
        """
        try:
            logger.info(f"Processing PDF: {file_path}")
            
            # Validate PDF file
            is_valid, error_msg = validate_pdf_file(file_path)
            if not is_valid:
                raise ValueError(f"Invalid PDF file: {error_msg}")
            
            # Get PDF metadata
            metadata = get_pdf_metadata(file_path)
            
            # Load PDF documents
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
            
            if not documents:
                raise ValueError("No content could be extracted from the PDF")
            
            logger.info(f"Loaded {len(documents)} pages from PDF")
            
            # Add file metadata to each document
            for doc in documents:
                doc.metadata.update({
                    'source': file_path.name,
                    'file_path': str(file_path),
                    'file_size': metadata.get('file_size', 0),
                    'total_pages': metadata.get('num_pages', 0)
                })
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            if not chunks:
                raise ValueError("No text chunks could be created from the PDF")
            
            logger.info(f"Created {len(chunks)} text chunks")
            
            # Add chunks to vector store
            chunk_ids = self.vectorstore.add_documents(chunks)
            
            # Persist the vector store
            self.vectorstore.persist()
            
            logger.info(f"Successfully processed PDF {file_path.name}: {len(chunks)} chunks added")
            
            return {
                'success': True,
                'filename': file_path.name,
                'document_count': len(documents),
                'chunk_count': len(chunks),
                'chunk_ids': chunk_ids,
                'metadata': metadata
            }
        
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            raise
    
    def get_retriever(self, k: Optional[int] = None) -> VectorStoreRetriever:
        """Get a retriever for similarity search.
        
        Args:
            k: Number of documents to retrieve (defaults to settings.retrieval_k)
            
        Returns:
            Vector store retriever
        """
        search_k = k or self.settings.retrieval_k
        
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": search_k}
        )
    
    def search_documents(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Search for documents similar to the query.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        try:
            search_k = k or self.settings.retrieval_k
            
            logger.info(f"Searching documents for query: '{query}' (k={search_k})")
            
            # Perform similarity search
            results = self.vectorstore.similarity_search(query, k=search_k)
            
            logger.info(f"Found {len(results)} relevant documents")
            
            return results
        
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            raise
    
    def search_documents_with_scores(self, query: str, k: Optional[int] = None) -> List[tuple]:
        """Search for documents with similarity scores.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of (document, score) tuples
        """
        try:
            search_k = k or self.settings.retrieval_k
            
            logger.info(f"Searching documents with scores for query: '{query}' (k={search_k})")
            
            # Perform similarity search with scores
            results = self.vectorstore.similarity_search_with_score(query, k=search_k)
            
            logger.info(f"Found {len(results)} relevant documents with scores")
            
            return results
        
        except Exception as e:
            logger.error(f"Error searching documents with scores: {str(e)}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the document collection.
        
        Returns:
            Dictionary containing collection information
        """
        try:
            # Get collection
            collection = self.vectorstore._collection
            
            # Get collection stats
            count = collection.count()
            
            return {
                'document_count': count,
                'collection_name': collection.name,
                'chroma_dir': str(self.settings.chroma_dir)
            }
        
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {
                'document_count': 0,
                'collection_name': 'task_documents',
                'chroma_dir': str(self.settings.chroma_dir),
                'error': str(e)
            }
    
    def delete_documents_by_source(self, source: str) -> bool:
        """Delete documents by source filename.
        
        Args:
            source: Source filename to delete
            
        Returns:
            True if documents were deleted, False otherwise
        """
        try:
            logger.info(f"Deleting documents from source: {source}")
            
            # Get collection
            collection = self.vectorstore._collection
            
            # Query documents by source
            results = collection.get(where={"source": source})
            
            if not results['ids']:
                logger.info(f"No documents found for source: {source}")
                return False
            
            # Delete documents
            collection.delete(ids=results['ids'])
            
            logger.info(f"Deleted {len(results['ids'])} documents from source: {source}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error deleting documents from source {source}: {str(e)}")
            return False
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection.
        
        Returns:
            True if collection was cleared, False otherwise
        """
        try:
            logger.warning("Clearing all documents from collection")
            
            # Get collection
            collection = self.vectorstore._collection
            
            # Get all document IDs
            results = collection.get()
            
            if not results['ids']:
                logger.info("Collection is already empty")
                return True
            
            # Delete all documents
            collection.delete(ids=results['ids'])
            
            logger.info(f"Cleared {len(results['ids'])} documents from collection")
            
            return True
        
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            return False


# Global RAG service instance - will be initialized during app startup
_rag_service: Optional[RAGService] = None


def get_rag_service() -> Optional[RAGService]:
    """Get the global RAG service instance.
    
    Returns:
        RAG service instance or None if not initialized
    """
    return _rag_service


def initialize_rag_service(settings: Settings) -> RAGService:
    """Initialize the global RAG service instance.
    
    Args:
        settings: Application settings
        
    Returns:
        Initialized RAG service
    """
    global _rag_service
    _rag_service = RAGService(settings)
    return _rag_service

