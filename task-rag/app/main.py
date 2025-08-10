"""FastAPI main application with app factory and route configuration."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import Settings
from .deps import get_settings
from .routes import chat, ingest, tasks
from .services.rag_service import initialize_rag_service
from .services.task_service import initialize_task_service
from .utils.logging import setup_logging
from .ws import websocket_endpoint

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup and shutdown events.

    Args:
        app: FastAPI application instance
    """
    # Startup
    logger.info("Starting up Task Management RAG application")

    try:
        # Get settings
        settings = get_settings()

        # Setup logging
        setup_logging(settings)
        logger.info("Logging configured")

        # Initialize services
        rag_service = initialize_rag_service(settings)
        logger.info("RAG service initialized")

        task_service = initialize_task_service()
        logger.info("Task service initialized")

        # Create required directories
        settings.uploads_dir.mkdir(parents=True, exist_ok=True)
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Required directories created")

        logger.info("Application startup completed successfully")

    except Exception as e:
        logger.error(f"Error during application startup: {str(e)}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down Task Management RAG application")

    try:
        # Cleanup services if needed
        logger.info("Application shutdown completed successfully")

    except Exception as e:
        logger.error(f"Error during application shutdown: {str(e)}")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance
    """
    # Create FastAPI app with lifespan manager
    app = FastAPI(
        title="Task Management RAG",
        description="A comprehensive task management system with RAG capabilities, LangGraph orchestration, and Telegram bot integration",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add custom middleware for request logging
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log all HTTP requests."""
        start_time = (
            logger.handlers[0].formatter.formatTime(
                logging.LogRecord("", 0, "", 0, "", (), None)
            )
            if logger.handlers
            else "unknown"
        )

        logger.info(f"Request: {request.method} {request.url}")

        response = await call_next(request)

        logger.info(
            f"Response: {response.status_code} for {request.method} {request.url}"
        )

        return response

    # Custom exception handlers
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle HTTP exceptions with proper logging."""
        logger.warning(
            f"HTTP {exc.status_code}: {exc.detail} for {request.method} {request.url}"
        )

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "status_code": exc.status_code,
                "path": str(request.url),
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        """Handle request validation errors with detailed information."""
        logger.warning(
            f"Validation error for {request.method} {request.url}: {exc.errors()}"
        )

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "Validation error",
                "details": exc.errors(),
                "status_code": 422,
                "path": str(request.url),
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions."""
        logger.error(
            f"Unexpected error for {request.method} {request.url}: {str(exc)}",
            exc_info=True,
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error",
                "status_code": 500,
                "path": str(request.url),
            },
        )

    # Health check endpoint
    @app.get("/healthz", tags=["health"])
    async def health_check():
        """Health check endpoint for monitoring and load balancers.

        Returns:
            Health status information
        """
        try:
            # Basic health checks
            settings = get_settings()

            # Check if required directories exist
            uploads_exists = settings.uploads_dir.exists()
            chroma_exists = settings.chroma_dir.exists()

            # Get service status (basic checks)
            from .services.rag_service import get_rag_service
            from .services.task_service import get_task_service

            rag_service = get_rag_service()
            task_service = get_task_service()

            health_status = {
                "status": "healthy",
                "timestamp": logger.handlers[0].formatter.formatTime(
                    logging.LogRecord("", 0, "", 0, "", (), None)
                )
                if logger.handlers
                else None,
                "version": "1.0.0",
                "services": {
                    "rag_service": "initialized" if rag_service else "not_initialized",
                    "task_service": "initialized"
                    if task_service
                    else "not_initialized",
                },
                "directories": {
                    "uploads": "exists" if uploads_exists else "missing",
                    "chroma": "exists" if chroma_exists else "missing",
                },
            }

            # Determine overall health
            if (
                not rag_service
                or not task_service
                or not uploads_exists
                or not chroma_exists
            ):
                health_status["status"] = "degraded"

            return health_status

        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": logger.handlers[0].formatter.formatTime(
                        logging.LogRecord("", 0, "", 0, "", (), None)
                    )
                    if logger.handlers
                    else None,
                },
            )

    # Root endpoint
    @app.get("/", tags=["root"])
    async def root():
        """Root endpoint with API information.

        Returns:
            API information and available endpoints
        """
        return {
            "name": "Task Management RAG API",
            "version": "1.0.0",
            "description": "A comprehensive task management system with RAG capabilities",
            "docs_url": "/docs",
            "health_check": "/healthz",
            "endpoints": {
                "ingestion": "/ingest",
                "tasks": "/tasks",
                "chat": "/chat",
                "websocket": "/ws/stream",
            },
        }

    # Include routers with proper prefixes and tags
    app.include_router(ingest.router, prefix="/ingest", tags=["ingestion"])

    app.include_router(tasks.router, prefix="/tasks", tags=["tasks"])

    app.include_router(chat.router, prefix="/chat", tags=["chat"])

    # Add WebSocket endpoint
    app.websocket("/ws/stream")(websocket_endpoint)

    logger.info("FastAPI application created and configured")

    return app


# Create the app instance
app = create_app()
