"""Ingestion routes for PDF and task data."""

import logging
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from ..config import Settings
from ..deps import get_rag_service, get_settings, get_task_service
from ..schemas import IngestResponse, TaskCreate, TaskResponse
from ..services.rag_service import RAGService
from ..services.task_service import TaskService
from ..utils.pdf import safe_save_uploaded_file

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("/pdf", response_model=IngestResponse)
async def ingest_pdf(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    rag_service: RAGService = Depends(get_rag_service)
) -> IngestResponse:
    """Ingest a PDF file for RAG processing.
    
    Args:
        file: Uploaded PDF file
        settings: Application settings
        rag_service: RAG service instance
        
    Returns:
        Ingestion response with processing results
        
    Raises:
        HTTPException: If file processing fails
    """
    try:
        logger.info(f"Starting PDF ingestion for file: {file.filename}")
        
        # Validate file type
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files are supported"
            )
        
        # Read file content
        file_content = await file.read()
        
        if not file_content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file uploaded"
            )
        
        # Save file to uploads directory
        try:
            file_path = safe_save_uploaded_file(
                file_content=file_content,
                filename=file.filename,
                upload_dir=settings.uploads_dir
            )
        except Exception as e:
            logger.error(f"Error saving uploaded file: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error saving file: {str(e)}"
            )
        
        # Process PDF with RAG service
        try:
            processing_result = rag_service.process_pdf(file_path)
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            # Clean up the saved file if processing fails
            try:
                file_path.unlink()
            except:
                pass
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing PDF: {str(e)}"
            )
        
        logger.info(f"Successfully processed PDF: {file.filename}")
        
        return IngestResponse(
            success=True,
            message=f"Successfully processed PDF '{file.filename}'",
            filename=processing_result['filename'],
            document_count=processing_result['document_count'],
            chunk_count=processing_result['chunk_count'],
            metadata=processing_result.get('metadata', {})
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during PDF ingestion: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during PDF processing"
        )


@router.post("/tasks", response_model=List[TaskResponse])
async def ingest_tasks(
    tasks_data: List[TaskCreate],
    task_service: TaskService = Depends(get_task_service)
) -> List[TaskResponse]:
    """Bulk create tasks from JSON list.
    
    Args:
        tasks_data: List of task creation data
        task_service: Task service instance
        
    Returns:
        List of created task responses
        
    Raises:
        HTTPException: If task creation fails
    """
    try:
        logger.info(f"Starting bulk task creation for {len(tasks_data)} tasks")
        
        if not tasks_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No tasks provided for creation"
            )
        
        if len(tasks_data) > 100:  # Reasonable limit for bulk operations
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Too many tasks (maximum 100 allowed)"
            )
        
        # Create tasks in bulk
        created_tasks = task_service.bulk_create_tasks(tasks_data)
        
        if not created_tasks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No tasks could be created (check task data validity)"
            )
        
        # Convert to response format
        task_responses = [
            TaskResponse(
                id=task.id,
                title=task.title,
                description=task.description,
                status=task.status,
                created_at=task.created_at,
                updated_at=task.updated_at
            )
            for task in created_tasks
        ]
        
        logger.info(f"Successfully created {len(created_tasks)} tasks")
        
        return task_responses
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during bulk task creation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during task creation"
        )


@router.get("/status")
async def get_ingestion_status(
    rag_service: RAGService = Depends(get_rag_service),
    task_service: TaskService = Depends(get_task_service)
) -> JSONResponse:
    """Get ingestion system status.
    
    Args:
        rag_service: RAG service instance
        task_service: Task service instance
        
    Returns:
        System status information
    """
    try:
        # Get RAG collection info
        collection_info = rag_service.get_collection_info()
        
        # Get task statistics
        task_stats = task_service.get_statistics()
        
        return JSONResponse(
            content={
                "status": "healthy",
                "rag_collection": collection_info,
                "task_statistics": task_stats,
                "timestamp": logger.handlers[0].formatter.formatTime(
                    logging.LogRecord("", 0, "", 0, "", (), None)
                ) if logger.handlers else None
            }
        )
    
    except Exception as e:
        logger.error(f"Error getting ingestion status: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "error": str(e)
            }
        )

