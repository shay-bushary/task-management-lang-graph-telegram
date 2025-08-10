"""Structured logging configuration for the Task Management RAG application."""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

from ..config import Settings


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        """Format log record with colors for console output."""
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}"
                f"{self.COLORS['RESET']}"
            )
        
        # Format the message
        formatted = super().format(record)
        
        return formatted


def setup_logging(settings: Settings) -> None:
    """Setup structured logging for the application.
    
    Args:
        settings: Application settings containing logging configuration
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level.upper()))
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, settings.log_level.upper()))
    
    console_formatter = ColoredFormatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler for all logs
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_dir / "app.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)  # Always log everything to file
    
    file_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Error file handler for errors and above
    error_handler = logging.handlers.RotatingFileHandler(
        filename=log_dir / "error.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    
    error_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s\n%(exc_info)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    error_handler.setFormatter(error_formatter)
    root_logger.addHandler(error_handler)
    
    # Configure specific loggers
    configure_module_loggers(settings)
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level: {settings.log_level.upper()}")
    logger.info(f"Log files will be written to: {log_dir.absolute()}")


def configure_module_loggers(settings: Settings) -> None:
    """Configure logging levels for specific modules.
    
    Args:
        settings: Application settings
    """
    # Application modules
    app_loggers = [
        'app.main',
        'app.routes',
        'app.services',
        'app.graph',
        'app.utils',
        'bots.telegram_bot'
    ]
    
    for logger_name in app_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, settings.log_level.upper()))
    
    # Third-party library loggers (usually more verbose)
    third_party_loggers = {
        'uvicorn': logging.INFO,
        'uvicorn.access': logging.WARNING,
        'fastapi': logging.INFO,
        'aiogram': logging.INFO,
        'aiohttp': logging.WARNING,
        'httpx': logging.WARNING,
        'openai': logging.WARNING,
        'chromadb': logging.WARNING,
        'langchain': logging.INFO,
        'langgraph': logging.INFO,
    }
    
    for logger_name, level in third_party_loggers.items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
    
    # Suppress overly verbose loggers in production
    if settings.environment == "production":
        verbose_loggers = [
            'uvicorn.access',
            'aiohttp.access',
            'httpx',
            'openai._base_client',
        ]
        
        for logger_name in verbose_loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.ERROR)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def log_function_call(func_name: str, args: Optional[dict] = None, kwargs: Optional[dict] = None):
    """Decorator to log function calls with arguments.
    
    Args:
        func_name: Function name to log
        args: Function arguments
        kwargs: Function keyword arguments
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            
            # Log function entry
            arg_str = ", ".join([str(arg) for arg in args])
            kwarg_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
            all_args = ", ".join(filter(None, [arg_str, kwarg_str]))
            
            logger.debug(f"Calling {func_name}({all_args})")
            
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func_name} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{func_name} failed with error: {str(e)}")
                raise
        
        return wrapper
    return decorator


def log_async_function_call(func_name: str):
    """Decorator to log async function calls.
    
    Args:
        func_name: Function name to log
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            
            # Log function entry
            arg_str = ", ".join([str(arg) for arg in args])
            kwarg_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
            all_args = ", ".join(filter(None, [arg_str, kwarg_str]))
            
            logger.debug(f"Calling async {func_name}({all_args})")
            
            try:
                result = await func(*args, **kwargs)
                logger.debug(f"Async {func_name} completed successfully")
                return result
            except Exception as e:
                logger.error(f"Async {func_name} failed with error: {str(e)}")
                raise
        
        return wrapper
    return decorator


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
    
    def log_info(self, message: str, *args, **kwargs):
        """Log info message."""
        self.logger.info(message, *args, **kwargs)
    
    def log_debug(self, message: str, *args, **kwargs):
        """Log debug message."""
        self.logger.debug(message, *args, **kwargs)
    
    def log_warning(self, message: str, *args, **kwargs):
        """Log warning message."""
        self.logger.warning(message, *args, **kwargs)
    
    def log_error(self, message: str, *args, **kwargs):
        """Log error message."""
        self.logger.error(message, *args, **kwargs)
    
    def log_exception(self, message: str, *args, **kwargs):
        """Log exception with traceback."""
        self.logger.exception(message, *args, **kwargs)


def configure_request_logging():
    """Configure request/response logging for FastAPI."""
    import time
    from fastapi import Request, Response
    
    async def log_requests(request: Request, call_next):
        """Middleware to log HTTP requests and responses."""
        logger = logging.getLogger("app.middleware.requests")
        
        # Log request
        start_time = time.time()
        logger.info(
            f"Request started: {request.method} {request.url} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        
        # Process request
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        logger.info(
            f"Request completed: {request.method} {request.url} "
            f"-> {response.status_code} in {process_time:.3f}s"
        )
        
        return response
    
    return log_requests


def log_startup_info(settings: Settings):
    """Log application startup information.
    
    Args:
        settings: Application settings
    """
    logger = logging.getLogger("app.startup")
    
    logger.info("=" * 60)
    logger.info("Task Management RAG Application Starting")
    logger.info("=" * 60)
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Log Level: {settings.log_level.upper()}")
    logger.info(f"OpenAI Model: {settings.model_name}")
    logger.info(f"Embeddings Model: {settings.embeddings_model}")
    logger.info(f"Uploads Directory: {settings.uploads_dir}")
    logger.info(f"Chroma Directory: {settings.chroma_dir}")
    
    if settings.telegram_bot_token:
        logger.info("Telegram Bot: Configured")
        if settings.telegram_webhook_url:
            logger.info(f"Telegram Webhook: {settings.telegram_webhook_url}")
        else:
            logger.info("Telegram Mode: Polling")
    else:
        logger.info("Telegram Bot: Not configured")
    
    logger.info("=" * 60)


def log_shutdown_info():
    """Log application shutdown information."""
    logger = logging.getLogger("app.shutdown")
    
    logger.info("=" * 60)
    logger.info("Task Management RAG Application Shutting Down")
    logger.info("=" * 60)


# Performance logging utilities
class PerformanceLogger:
    """Utility class for performance logging."""
    
    def __init__(self, logger_name: str):
        """Initialize performance logger.
        
        Args:
            logger_name: Name of the logger to use
        """
        self.logger = logging.getLogger(logger_name)
    
    def log_operation_time(self, operation: str, duration: float, **kwargs):
        """Log operation timing.
        
        Args:
            operation: Operation name
            duration: Duration in seconds
            **kwargs: Additional context
        """
        context = " ".join([f"{k}={v}" for k, v in kwargs.items()])
        self.logger.info(f"Performance: {operation} took {duration:.3f}s {context}")
    
    def log_memory_usage(self, operation: str, memory_mb: float):
        """Log memory usage.
        
        Args:
            operation: Operation name
            memory_mb: Memory usage in MB
        """
        self.logger.info(f"Memory: {operation} used {memory_mb:.2f}MB")


# Context manager for timing operations
class TimedOperation:
    """Context manager for timing operations."""
    
    def __init__(self, operation_name: str, logger_name: str = __name__):
        """Initialize timed operation.
        
        Args:
            operation_name: Name of the operation
            logger_name: Logger name to use
        """
        self.operation_name = operation_name
        self.logger = logging.getLogger(logger_name)
        self.start_time = None
    
    def __enter__(self):
        """Start timing."""
        import time
        self.start_time = time.time()
        self.logger.debug(f"Starting operation: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log result."""
        import time
        duration = time.time() - self.start_time
        
        if exc_type is None:
            self.logger.info(f"Operation completed: {self.operation_name} in {duration:.3f}s")
        else:
            self.logger.error(f"Operation failed: {self.operation_name} after {duration:.3f}s")


# Structured logging for specific events
def log_user_action(user_id: str, action: str, details: Optional[dict] = None):
    """Log user actions for analytics.
    
    Args:
        user_id: User identifier
        action: Action performed
        details: Additional action details
    """
    logger = logging.getLogger("app.analytics.user_actions")
    
    log_data = {
        "user_id": user_id,
        "action": action,
        "timestamp": logging.Formatter().formatTime(
            logging.LogRecord("", 0, "", 0, "", (), None)
        )
    }
    
    if details:
        log_data.update(details)
    
    logger.info(f"User action: {log_data}")


def log_system_event(event_type: str, details: Optional[dict] = None):
    """Log system events for monitoring.
    
    Args:
        event_type: Type of system event
        details: Additional event details
    """
    logger = logging.getLogger("app.system.events")
    
    log_data = {
        "event_type": event_type,
        "timestamp": logging.Formatter().formatTime(
            logging.LogRecord("", 0, "", 0, "", (), None)
        )
    }
    
    if details:
        log_data.update(details)
    
    logger.info(f"System event: {log_data}")


# Export commonly used functions
__all__ = [
    'setup_logging',
    'get_logger',
    'log_function_call',
    'log_async_function_call',
    'LoggerMixin',
    'configure_request_logging',
    'log_startup_info',
    'log_shutdown_info',
    'PerformanceLogger',
    'TimedOperation',
    'log_user_action',
    'log_system_event'
]

